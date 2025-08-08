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
        
        # Load English prompt helpers
        self._load_english_prompt_helpers()
        
        logger.info(f"RAG interface initialized for endpoint: {self.endpoint}")
    
    def _load_english_prompt_helpers(self):
        """Load English prompt helper functions."""
        try:
            from .english_prompts import get_english_system_prompt, create_custom_english_prompt
            self.get_english_system_prompt = get_english_system_prompt
            self.create_custom_english_prompt = create_custom_english_prompt
        except ImportError:
            # Fallback functions if module not available
            def get_english_system_prompt(prompt_type: str = 'default') -> str:
                return """You are a helpful assistant for evaluation purposes that MUST respond only in English. 

MANDATORY REQUIREMENTS FOR EVALUATION:
- Always respond in English language only, regardless of the input language
- If the question is in Chinese, Japanese, Korean, or any other language, translate it internally and answer in English
- Use clear, professional English with proper grammar and vocabulary
- Do not mix languages in your response under any circumstances
- If you cannot understand the input language, ask for clarification in English only
- Provide comprehensive and accurate answers to enable proper evaluation

EVALUATION CONTEXT:
- Your responses are being evaluated for quality and accuracy
- Consistent English responses ensure fair evaluation across all test cases
- Focus on providing helpful, detailed answers in English

Remember: ALL responses must be in English only for proper evaluation."""
            
            def create_custom_english_prompt(domain: str = "", requirements: list = None) -> str:
                base = "You are a helpful assistant that MUST respond only in English."
                if domain:
                    base += f" You specialize in {domain} topics."
                
                base += """\n
LANGUAGE REQUIREMENTS:
- Always respond in English language only
- Translate non-English inputs internally before answering
- Use appropriate terminology for the domain
- Maintain professional tone
- Do not mix languages in your response

EVALUATION CONTEXT:
- Your responses are being evaluated for quality and accuracy
- Consistent English responses ensure fair evaluation across all test cases"""
                
                if requirements:
                    base += "\n\nADDITIONAL REQUIREMENTS:\n"
                    for req in requirements:
                        base += f"- {req}\n"
                
                base += "\nRemember: ALL responses must be in English only for proper evaluation."
                return base
            
            self.get_english_system_prompt = get_english_system_prompt
            self.create_custom_english_prompt = create_custom_english_prompt
    
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
        
        # Prepare request - check if payload_template is configured
        payload_template = self.request_format.get('payload_template')
        
        # Enhanced system prompt to ensure responses are in English
        system_prompt = self._build_english_system_prompt()
        
        if payload_template:
            # Use payload template and substitute question
            request_data = {}
            for key, value in payload_template.items():
                if isinstance(value, str) and "{question}" in value:
                    request_data[key] = value.format(question=question)
                else:
                    request_data[key] = value
            
            # Add system prompt to payload in multiple possible fields
            self._add_system_prompt_to_payload(request_data, system_prompt)
            logger.debug(f"Using payload template: {request_data}")
        else:
            # Use simple question field format
            request_data = {
                self.question_field: question,
                'system_prompt': system_prompt
            }
            logger.debug(f"Using simple format with field '{self.question_field}': {request_data}")
        
        # Add any additional request parameters
        request_params = self.config.get('request_params', {})
        request_data.update(request_params)
        
        logger.info(f"Final RAG request payload: {request_data}")
        
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
        
        # Debug logging for response parsing
        logger.debug(f"RAG response keys: {list(response_data.keys())}")
        logger.debug(f"RAG response structure: {response_data}")
        logger.debug(f"Extracted answer: {answer}")
        logger.debug(f"Answer field path: {self.response_fields.get('answer', 'answer')}")
        
        # If answer extraction failed, try alternative common fields
        if not answer:
            alternative_fields = ['response', 'result', 'text', 'content', 'answer']
            for alt_field in alternative_fields:
                alt_answer = self._extract_field(response_data, alt_field)
                if alt_answer:
                    logger.info(f"Found answer using alternative field: {alt_field}")
                    answer = alt_answer
                    break
        
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
        Extract field from nested dictionary using dot notation and array indexing.
        
        Args:
            data: Dictionary to extract from
            field_path: Field path (e.g., 'data.answer', 'message[0].content', 'results[0].text')
            
        Returns:
            Extracted value or None if not found
        """
        if not field_path:
            return None
        
        import re
        
        current = data
        # Split path and handle both dot notation and array indexing
        parts = field_path.split('.')
        
        for part in parts:
            if current is None:
                return None
                
            # Check if this part has array indexing like 'message[0]'
            array_match = re.match(r'^(\w+)\[(\d+)\]$', part)
            if array_match:
                key, index = array_match.groups()
                index = int(index)
                
                if isinstance(current, dict) and key in current:
                    current = current[key]
                    if isinstance(current, list) and 0 <= index < len(current):
                        current = current[index]
                    else:
                        return None
                else:
                    return None
            else:
                # Regular dictionary key access
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
        
        return current
    
    def _build_english_system_prompt(self) -> str:
        """
        Build a comprehensive system prompt to ensure English responses.
        
        Returns:
            Comprehensive system prompt for English-only responses
        """
        # Check for custom prompt in config
        custom_prompt = self.config.get('english_system_prompt')
        if custom_prompt:
            return custom_prompt
        
        # Check for prompt type preference
        prompt_type = self.config.get('english_prompt_type', 'default')
        
        # Check for domain-specific requirements
        domain = self.config.get('domain', '')
        additional_requirements = self.config.get('english_requirements', [])
        
        if domain or additional_requirements:
            return self.create_custom_english_prompt(domain, additional_requirements)
        else:
            return self.get_english_system_prompt(prompt_type)
    
    def _add_system_prompt_to_payload(self, request_data: dict, system_prompt: str) -> None:
        """
        Add system prompt to request payload, trying multiple common field names.
        
        Args:
            request_data: The request payload dictionary
            system_prompt: The system prompt to add
        """
        # Common field names for system prompts in different RAG systems
        system_prompt_fields = [
            'system_prompt',
            'system_message', 
            'system',
            'instructions',
            'prompt',
            'context_instructions'
        ]
        
        # Try to add system prompt to existing fields or create new one
        prompt_added = False
        
        # Check if any system prompt field already exists
        for field in system_prompt_fields:
            if field in request_data:
                # Append to existing system prompt
                existing_prompt = request_data[field]
                if existing_prompt:
                    request_data[field] = f"{existing_prompt}\n\n{system_prompt}"
                else:
                    request_data[field] = system_prompt
                prompt_added = True
                break
        
        # If no existing field found, add as 'system_prompt'
        if not prompt_added:
            request_data['system_prompt'] = system_prompt
        
        # For SMT assistant specifically, also add to context if available
        if 'context' in request_data and isinstance(request_data['context'], dict):
            request_data['context']['language_instruction'] = "Respond only in English"
            request_data['context']['evaluation_mode'] = "English-only responses required for evaluation"
        
        # Also try adding to common nested structures
        if 'messages' in request_data and isinstance(request_data['messages'], list):
            # Add system message to messages array if it exists
            system_message = {
                'role': 'system',
                'content': system_prompt
            }
            request_data['messages'].insert(0, system_message)
            prompt_added = True
        
        logger.debug(f"Added English system prompt to request payload")
    
    def _create_error_response(self, error_type: str, response_time: float = 0.0) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            error_type: Type of error that occurred
            response_time: Time taken for the failed request
            
        Returns:
            Standardized error response dictionary
        """
        return {
            'success': False,
            'error': error_type,
            'answer': f'Error: {error_type}',
            'contexts': [],
            'confidence': 0.0,
            'response_time': response_time,
            'metadata': {
                'error_type': error_type,
                'timestamp': time.time()
            }
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
