"""
RAG System Interface for Pipeline Evaluation

Provides a standardized interface to communicate with RAG systems for evaluation.
"""

import logging
import requests
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class RAGInterface:
    """Interface for communicating with RAG systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG system interface.
        
        Args:
            config: RAG system configuration
        """
        self.config = config
        self.endpoint = config.get('endpoint', '')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        # Authentication setup
        self.auth_config = config.get('auth', {})
        self.session = self._setup_session()
        
        # Request format configuration
        self.request_format = config.get('request_format', {})
        self.question_field = self.request_format.get('question_field', 'question')
        self.response_fields = self.request_format.get('response_fields', {
            'answer': 'answer',
            'contexts': 'contexts',
            'confidence': 'confidence'
        })
        
        logger.info(f"RAG interface initialized for endpoint: {self.endpoint}")
    
    def _setup_session(self) -> requests.Session:
        """Setup HTTP session with authentication."""
        session = requests.Session()
        
        auth_type = self.auth_config.get('type', 'none')
        
        if auth_type == 'bearer':
            token = self.auth_config.get('token', '')
            if token:
                session.headers.update({'Authorization': f'Bearer {token}'})
        elif auth_type == 'api_key':
            api_key = self.auth_config.get('api_key', '')
            header_name = self.auth_config.get('header_name', 'X-API-Key')
            if api_key:
                session.headers.update({header_name: api_key})
        
        # Set common headers
        session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RAG-Evaluation-Pipeline/1.0'
        })
        
        return session
    
    def query_rag_system(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask the RAG system
            
        Returns:
            Dictionary containing RAG response with normalized fields
        """
        if not self.endpoint:
            raise ValueError("RAG endpoint not configured")
        
        # Prepare request
        request_data = {self.question_field: question}
        
        # Add any additional request parameters
        request_params = self.config.get('request_params', {})
        request_data.update(request_params)
        
        # Execute request with retries
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = self.session.post(
                    self.endpoint,
                    json=request_data,
                    timeout=self.timeout
                )
                
                response_time = time.time() - start_time
                
                # Check response status
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                
                # Normalize response format
                normalized_response = self._normalize_response(response_data, response_time)
                
                logger.debug(f"RAG query successful (attempt {attempt + 1}): {question[:50]}...")
                return normalized_response
                
            except requests.exceptions.Timeout:
                logger.warning(f"RAG query timeout (attempt {attempt + 1}): {question[:50]}...")
                if attempt == self.max_retries - 1:
                    return self._create_error_response("timeout", response_time=self.timeout)
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"RAG connection error (attempt {attempt + 1}): {question[:50]}...")
                if attempt == self.max_retries - 1:
                    return self._create_error_response("connection_error")
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"RAG HTTP error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return self._create_error_response(f"http_error_{response.status_code}")
                
            except Exception as e:
                logger.error(f"RAG query error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return self._create_error_response(f"unknown_error: {str(e)}")
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return self._create_error_response("max_retries_exceeded")
    
    def batch_query(self, questions: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Query RAG system with multiple questions.
        
        Args:
            questions: List of questions to ask
            batch_size: Number of questions to process in each batch
            
        Returns:
            List of RAG responses
        """
        logger.info(f"Starting batch query of {len(questions)} questions")
        
        results = []
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}: questions {i+1}-{min(i+batch_size, len(questions))}")
            
            batch_results = []
            for question in batch:
                result = self.query_rag_system(question)
                batch_results.append(result)
                
                # Add small delay between requests to be respectful
                time.sleep(0.1)
            
            results.extend(batch_results)
            
            # Log progress
            if i + batch_size < len(questions):
                logger.info(f"Completed {i + batch_size}/{len(questions)} questions")
        
        logger.info(f"Batch query completed: {len(results)} responses")
        return results
    
    def _normalize_response(self, response_data: Dict[str, Any], response_time: float) -> Dict[str, Any]:
        """
        Normalize RAG response to standard format.
        
        Args:
            response_data: Raw response from RAG system
            response_time: Time taken for the request
            
        Returns:
            Normalized response dictionary
        """
        # Extract fields based on configuration
        answer = self._extract_field(response_data, self.response_fields.get('answer', 'answer'))
        contexts = self._extract_field(response_data, self.response_fields.get('contexts', 'contexts'))
        confidence = self._extract_field(response_data, self.response_fields.get('confidence', 'confidence'))
        
        # Ensure contexts is a list
        if contexts is not None and not isinstance(contexts, list):
            contexts = [str(contexts)]
        elif contexts is None:
            contexts = []
        
        # Ensure confidence is a float
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = None
        
        return {
            'answer': str(answer) if answer is not None else '',
            'contexts': contexts,
            'confidence': confidence,
            'response_time': response_time,
            'success': True,
            'error': None,
            'raw_response': response_data
        }
    
    def _extract_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """
        Extract field from nested dictionary using dot notation.
        
        Args:
            data: Dictionary to extract from
            field_path: Field path (e.g., 'data.answer' or 'result.contexts')
            
        Returns:
            Extracted value or None if not found
        """
        if not field_path:
            return None
        
        current = data
        for key in field_path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _create_error_response(self, error_type: str, response_time: float = 0.0) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error_type: Type of error that occurred
            response_time: Time taken before error
            
        Returns:
            Error response dictionary
        """
        return {
            'answer': '',
            'contexts': [],
            'confidence': None,
            'response_time': response_time,
            'success': False,
            'error': error_type,
            'raw_response': {}
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to RAG system.
        
        Returns:
            Test result dictionary
        """
        test_question = "What is this system?"
        
        try:
            logger.info("Testing RAG system connection...")
            result = self.query_rag_system(test_question)
            
            if result['success']:
                logger.info("✅ RAG system connection test successful")
                return {
                    'success': True,
                    'response_time': result['response_time'],
                    'endpoint': self.endpoint
                }
            else:
                logger.warning(f"⚠️ RAG system responded but with error: {result['error']}")
                return {
                    'success': False,
                    'error': result['error'],
                    'endpoint': self.endpoint
                }
                
        except Exception as e:
            logger.error(f"❌ RAG system connection test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'endpoint': self.endpoint
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system interface configuration.
        
        Returns:
            System information dictionary
        """
        return {
            'endpoint': self.endpoint,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'auth_type': self.auth_config.get('type', 'none'),
            'question_field': self.question_field,
            'response_fields': self.response_fields
        }
