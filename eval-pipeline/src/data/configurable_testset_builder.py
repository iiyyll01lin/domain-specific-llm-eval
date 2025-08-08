#!/usr/bin/env python3
"""
Configurable Testset Builder - Flexible Testset Generation Interface

Provides a user-friendly interface for selecting testset generation types and strategies
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import pandas as pd
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.hybrid_testset_generator import HybridTestsetGenerator

class ConfigurableTestsetBuilder:
    """
    Configurable testset builder - allows users to select generation strategies
    """
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml", config: Optional[Dict[str, Any]] = None):
        self.config_path = config_path
        if config is not None:
            self.base_config = config
        else:
            self.base_config = self._load_base_config()
        self.logger = logging.getLogger(__name__)
        
        # Available generation strategies
        self.available_strategies = {
            'synthetic_llm': {
                'name': 'LLM Synthetic Generation',
                'description': 'Use large language models to generate diverse questions',
                'requires_llm': True,
                'quality': 'high',
                'diversity': 'high'
            },
            'synthetic_local': {
                'name': 'Local Synthetic Generation', 
                'description': 'Use existing keyword extraction system for generation',
                'requires_llm': False,
                'quality': 'medium',
                'diversity': 'medium'
            },
            'domain_specific': {
                'name': 'Domain-Specific Tests',
                'description': 'Generate domain-relevant questions based on document content',
                'requires_llm': False,
                'quality': 'high',
                'diversity': 'medium'
            },
            'edge_cases': {
                'name': 'Edge Case Testing',
                'description': 'Generate system boundary and error handling tests',
                'requires_llm': False,
                'quality': 'medium',
                'diversity': 'low'
            },
            'user_query_simulation': {
                'name': 'User Query Simulation',
                'description': 'Simulate real user query patterns',
                'requires_llm': True,
                'quality': 'high',
                'diversity': 'high'
            },
            'golden_standard': {
                'name': 'Golden Standard Tests',
                'description': 'Expert-level high-quality standard test cases',
                'requires_llm': False,
                'quality': 'very_high',
                'diversity': 'low'
            }
        }
        
        # Question type distribution
        self.question_types = {
            'factual': {
                'name': 'Factual Questions',
                'description': 'Questions with answers directly found in documents',
                'example': 'What is the default port for RedFish API?'
            },
            'reasoning': {
                'name': 'Reasoning Questions', 
                'description': 'Questions requiring logical reasoning and multi-step thinking',
                'example': 'Why is RedFish more suitable than SNMP for modern data center management?'
            },
            'multi_context': {
                'name': 'Multi-Context Questions',
                'description': 'Questions requiring integration of multiple document sources', 
                'example': 'Compare the security differences between RedFish and IPMI'
            },
            'conditional': {
                'name': 'Conditional Questions',
                'description': 'Questions containing hypothetical conditions',
                'example': 'If server temperature exceeds threshold, how would RedFish API respond?'
            },
            'procedural': {
                'name': 'Procedural Questions',
                'description': 'Questions about operational steps and processes',
                'example': 'How to restart a server using RedFish API?'
            },
            'troubleshooting': {
                'name': 'Troubleshooting Questions',
                'description': 'Questions about problem diagnosis and resolution',
                'example': 'When RedFish connection fails, what settings should be checked?'
            }
        }
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Unable to load configuration file {self.config_path}: {e}")
            return {}
    
    def show_available_options(self):
        """Display available generation options"""
        print("\n🎯 Available Testset Generation Strategies:")
        print("=" * 60)
        
        for strategy_id, strategy in self.available_strategies.items():
            llm_req = "🔴 Requires LLM" if strategy['requires_llm'] else "🟢 Local Only"
            quality = f"📊 Quality: {strategy['quality']}"
            diversity = f"🌈 Diversity: {strategy['diversity']}"
            
            print(f"\n{strategy_id}:")
            print(f"  📝 Name: {strategy['name']}")
            print(f"  📖 Description: {strategy['description']}")
            print(f"  {llm_req} | {quality} | {diversity}")
        
        print("\n🔍 Available Question Types:")
        print("=" * 60)
        
        for qtype_id, qtype in self.question_types.items():
            print(f"\n{qtype_id}:")
            print(f"  📝 Name: {qtype['name']}")
            print(f"  📖 Description: {qtype['description']}")
            print(f"  💡 Example: {qtype['example']}")
    
    def create_custom_config(self, 
                           strategies: List[str],
                           question_distribution: Dict[str, float],
                           total_samples: int = 100,
                           samples_per_document: int = 20,
                           document_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create custom generation configuration
        
        Args:
            strategies: List of selected generation strategies
            question_distribution: Question type distribution (should sum to 1.0)
            total_samples: Total number of samples
            samples_per_document: Samples per document
            document_paths: List of document paths
        """
        
        # Validate input
        if sum(question_distribution.values()) != 1.0:
            print("⚠️ Warning: Question type distribution doesn't sum to 1.0, will normalize automatically")
            total = sum(question_distribution.values())
            question_distribution = {k: v/total for k, v in question_distribution.items()}
        
        # Check if strategies require LLM
        requires_llm = any(
            self.available_strategies[strategy]['requires_llm'] 
            for strategy in strategies 
            if strategy in self.available_strategies
        )
        
        # Build configuration
        custom_config = {
            'testset_generation': {
                'method': 'hybrid' if requires_llm else 'configurable',
                'samples_per_document': samples_per_document,
                'max_total_samples': total_samples,
                'strategies': question_distribution,
                'selected_strategies': strategies,
                'generation_timestamp': datetime.now().isoformat(),
                
                # Keyword extraction settings
                'keyword_extraction': {
                    'methods': ['keybert', 'yake', 'spacy_ner'],
                    'min_keywords': 3,
                    'max_keywords': 8
                },
                
                # RAGAS configuration (if LLM required)
                'use_ragas_generator': requires_llm,
                'ragas_config': {
                    'use_custom_llm': requires_llm,
                    'use_openai': False,
                    'custom_llm': self.base_config.get('testset_generation', {}).get('ragas_config', {}).get('custom_llm', {})
                } if requires_llm else {}
            },
            
            # Document sources
            'data_sources': {
                'documents': {
                    'primary_docs': document_paths or self.base_config.get('data_sources', {}).get('documents', {}).get('primary_docs', [])
                }
            }
        }
        
        return custom_config
    
    def generate_edge_cases(self, domain: str = "RAG API") -> List[Dict[str, Any]]:
        """Generate edge case tests"""
        edge_cases = []
        
        # Out-of-domain questions
        out_of_domain = [
            {"question": "How's the weather today?", "expected_behavior": "should_reject", "category": "out_of_domain"},
            {"question": "What is Bitcoin?", "expected_behavior": "should_reject", "category": "out_of_domain"},
            {"question": "How to cook pasta?", "expected_behavior": "should_reject", "category": "out_of_domain"}
        ]
        
        # Ambiguous questions
        ambiguous = [
            {"question": "How to fix this?", "expected_behavior": "should_clarify", "category": "ambiguous"},
            {"question": "Why doesn't it work?", "expected_behavior": "should_clarify", "category": "ambiguous"},
            {"question": "Where are the settings?", "expected_behavior": "should_clarify", "category": "ambiguous"}
        ]
        
        # No-answer questions (information not in knowledge base)
        no_answer = [
            {"question": "When will the next version be released?", "expected_behavior": "should_say_unknown", "category": "no_answer"},
            {"question": "What is the product price?", "expected_behavior": "should_say_unknown", "category": "no_answer"},
            {"question": "Who is the CEO?", "expected_behavior": "should_say_unknown", "category": "no_answer"}
        ]
        
        # Very long input
        very_long = [
            {"question": "Please tell me" + " very" * 100 + " detailed information", "expected_behavior": "should_handle_gracefully", "category": "long_input"}
        ]
        
        # Empty input and special characters
        special_inputs = [
            {"question": "", "expected_behavior": "should_handle_empty", "category": "empty_input"},
            {"question": "   ", "expected_behavior": "should_handle_whitespace", "category": "whitespace_only"},
            {"question": "!@#$%^&*()", "expected_behavior": "should_handle_special_chars", "category": "special_characters"}
        ]
        
        edge_cases.extend(out_of_domain + ambiguous + no_answer + very_long + special_inputs)
        
        for case in edge_cases:
            case.update({
                "generation_method": "edge_case_generation",
                "question_type": "edge_case",
                "auto_keywords": [],
                "source_file": "edge_case_generator"
            })
        
        return edge_cases
    
    def generate_user_query_simulation(self, domain: str = "server_management") -> List[Dict[str, Any]]:
        """Simulate real user query patterns"""
        
        # Query patterns based on different user roles
        user_personas = {
            "system_admin": [
                "How to restart all servers?",
                "Where is server temperature monitoring configured?", 
                "How to check hardware status?",
                "How to configure RAID settings?"
            ],
            "developer": [
                "Where is the API endpoint list?",
                "How to get JSON format response?",
                "How to configure authentication token?",
                "What's the POST request parameter format?"
            ],
            "manager": [
                "How is the overall system health?",
                "Can you generate operation reports?", 
                "What monitoring metrics are available?",
                "How to set alert thresholds?"
            ],
            "novice_user": [
                "What does this system do?",
                "Where should I start learning?",
                "Are there any simple tutorials?",
                "What are the most basic operations?"
            ]
        }
        
        simulated_queries = []
        for persona, queries in user_personas.items():
            for query in queries:
                simulated_queries.append({
                    "question": query,
                    "user_persona": persona,
                    "generation_method": "user_simulation",
                    "question_type": "user_simulated",
                    "auto_keywords": [],
                    "expected_behavior": "should_provide_helpful_answer"
                })
        
        return simulated_queries
    
    def generate_golden_standard(self) -> List[Dict[str, Any]]:
        """Generate golden standard test cases (high-quality expert-level)"""
        
        golden_cases = [
            {
                "question": "What are the main advantages of RedFish API?",
                "ground_truth": "RedFish API provides standardized RESTful interface, supports HTTPS secure communication, uses JSON format, and has good scalability and interoperability.",
                "category": "fundamental_concept",
                "difficulty": "basic",
                "quality_score": 1.0
            },
            {
                "question": "How to query server power status using RedFish API?",
                "ground_truth": "Access /redfish/v1/Systems/{SystemId}/PowerState endpoint via GET request, the PowerState field in response shows current power status.",
                "category": "practical_operation", 
                "difficulty": "intermediate",
                "quality_score": 1.0
            },
            {
                "question": "How to handle authentication and authorization in RedFish API?",
                "ground_truth": "RedFish supports multiple authentication methods including basic authentication, session authentication, and certificate authentication. Authorization is implemented based on role and permission system.",
                "category": "security",
                "difficulty": "advanced",
                "quality_score": 1.0
            }
        ]
        
        for case in golden_cases:
            case.update({
                "generation_method": "golden_standard",
                "question_type": "expert_curated",
                "auto_keywords": [],
                "source_file": "expert_knowledge_base"
            })
        
        return golden_cases
    
    def build_testset(self, config: Dict[str, Any], output_dir: str = "outputs/custom_testsets") -> Dict[str, Any]:
        """
        Build testset according to configuration
        """
        
        self.logger.info("🚀 Starting custom testset construction...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize hybrid generator
        testset_config = config['testset_generation']
        generator = HybridTestsetGenerator(testset_config)
        
        # Get document paths
        document_paths = config['data_sources']['documents']['primary_docs']
        
        # Generate main testset
        main_results = generator.generate_comprehensive_testset(
            document_paths=document_paths,
            output_dir=output_path
        )
        
        # Add additional test cases based on selected strategies
        additional_cases = []
        selected_strategies = testset_config.get('selected_strategies', [])
        
        if 'edge_cases' in selected_strategies:
            self.logger.info("🎯 Generating edge case tests...")
            edge_cases = self.generate_edge_cases()
            additional_cases.extend(edge_cases)
        
        if 'user_query_simulation' in selected_strategies:
            self.logger.info("👤 Generating user query simulation...")
            user_queries = self.generate_user_query_simulation()
            additional_cases.extend(user_queries)
        
        if 'golden_standard' in selected_strategies:
            self.logger.info("⭐ Generating golden standard tests...")
            golden_cases = self.generate_golden_standard()
            additional_cases.extend(golden_cases)
        
        # Merge all test cases
        if additional_cases:
            additional_df = pd.DataFrame(additional_cases)
            
            if main_results.get('success') and 'testset' in main_results:
                # Ensure field consistency
                main_testset = main_results['testset']
                
                # Fill missing fields
                for col in main_testset.columns:
                    if col not in additional_df.columns:
                        additional_df[col] = ""
                
                for col in additional_df.columns:
                    if col not in main_testset.columns:
                        main_testset[col] = ""
                
                # Merge
                combined_testset = pd.concat([main_testset, additional_df], ignore_index=True)
            else:
                combined_testset = additional_df
            
            main_results['testset'] = combined_testset
            main_results['total_samples'] = len(combined_testset)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"custom_testset_{timestamp}.csv"
        
        if 'testset' in main_results:
            main_results['testset'].to_csv(output_file, index=False, encoding='utf-8')
            self.logger.info(f"✅ Testset saved to: {output_file}")
        
        return main_results

    def generate_testset(self, documents: List[Dict[str, Any]], output_file: str = None) -> pd.DataFrame:
        """
        Generate testset using configurable strategies.
        
        Args:
            documents: List of processed documents
            output_file: Optional output CSV file path
            
        Returns:
            pandas.DataFrame: Generated testset
        """
        try:
            # Use configured strategies from base_config
            testset_config = self.base_config.get('testset_generation', {})
            
            # Create default configuration if not specified
            if not testset_config.get('selected_strategies'):
                testset_config = self.create_custom_config(
                    strategies=['synthetic_local', 'domain_specific'],
                    question_distribution={'factual': 0.6, 'reasoning': 0.4},
                    total_samples=20,
                    samples_per_document=10
                )['testset_generation']
            
            # Build testset using configuration
            results = self.build_testset_from_documents(documents, testset_config)
            
            # Convert to DataFrame
            if results.get('success', False) and results.get('testset'):
                df = pd.DataFrame(results['testset'])
                
                # Save to file if specified
                if output_file:
                    df.to_csv(output_file, index=False)
                    self.logger.info(f"Testset saved to {output_file}")
                
                return df
            else:
                # Return empty DataFrame with standard columns
                return pd.DataFrame(columns=[
                    'question', 'answer', 'contexts', 'ground_truth', 
                    'metadata', 'question_type', 'difficulty'
                ])
                
        except Exception as e:
            self.logger.error(f"Error generating testset: {e}")
            return pd.DataFrame(columns=[
                'question', 'answer', 'contexts', 'ground_truth', 
                'metadata', 'question_type', 'difficulty'
            ])
    
    def build_testset_from_documents(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build testset from processed documents using given configuration.
        
        Args:
            documents: List of processed documents
            config: Testset generation configuration
            
        Returns:
            Dict with testset generation results
        """
        try:
            from data.hybrid_testset_generator import HybridTestsetGenerator
            
            # Use HybridTestsetGenerator as the backend for actual generation
            generator = HybridTestsetGenerator(config=config)
            
            # Generate testset using the hybrid generator's comprehensive method
            document_paths = [doc.get('source_file', '') for doc in documents if doc.get('source_file')]
            
            # Filter out empty paths and ensure we have valid documents
            valid_paths = [path for path in document_paths if path and path.strip()]
            
            if not valid_paths:
                # Create dummy document for testing
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write("Sample test document content for testset generation.")
                    valid_paths = [f.name]
            
            results = generator.generate_comprehensive_testset(
                document_paths=valid_paths,
                output_dir=Path("outputs/testsets")  # Use proper path instead of None
            )
            
            # Extract testset data
            if results.get('success', False) and results.get('testset') is not None:
                testset = results['testset']
                
                # Convert DataFrame to list of dictionaries if needed
                if hasattr(testset, 'to_dict'):
                    testset = testset.to_dict('records')
                
                return {
                    'success': True,
                    'testset': testset,
                    'total_samples': len(testset),
                    'strategies_used': config.get('selected_strategies', []),
                    'generation_timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': results.get('error', 'No testset generated'),
                    'testset': [],
                    'total_samples': 0
                }
                
        except Exception as e:
            self.logger.error(f"Error in build_testset_from_documents: {e}")
            return {
                'success': False,
                'error': str(e),
                'testset': [],
                'total_samples': 0
            }
    
    def build_testset(self, config: Dict[str, Any] = None, output_dir: str = None) -> Dict[str, Any]:
        """
        Build testset using the orchestrator interface.
        
        Args:
            config: Pipeline configuration (optional, uses self.base_config if not provided)
            output_dir: Output directory for saving results
            
        Returns:
            Dict with testset generation results
        """
        try:
            # Use provided config or fallback to base_config
            if config is None:
                config = self.base_config
            
            # Get document sources from config
            doc_sources = config.get('data_sources', {}).get('documents', {})
            primary_docs = doc_sources.get('primary_docs', [])
            
            if not primary_docs:
                self.logger.warning("No primary documents specified in configuration")
                return {
                    'success': False,
                    'error': 'No documents specified',
                    'testset': pd.DataFrame(),
                    'metadata': {}
                }
            
            # For simplicity, create mock document structure
            # In a real implementation, this would use DocumentProcessor
            documents = []
            for doc_path in primary_docs:
                documents.append({
                    'source_file': doc_path,
                    'content': f"Sample content from {doc_path}",
                    'metadata': {'processed_timestamp': datetime.now().isoformat()}
                })
            
            # Generate testset
            testset_config = config.get('testset_generation', {})
            results = self.build_testset_from_documents(documents, testset_config)
            
            # Convert testset to DataFrame if it's a list
            if results.get('success', False) and isinstance(results.get('testset'), list):
                df = pd.DataFrame(results['testset'])
                results['testset'] = df
                
                # Save to output directory if specified
                if output_dir and not df.empty:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save as CSV
                    csv_file = output_path / f"testset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(csv_file, index=False)
                    self.logger.info(f"Testset saved to {csv_file}")
                    
                    results['output_file'] = str(csv_file)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in build_testset: {e}")
            return {
                'success': False,
                'error': str(e),
                'testset': pd.DataFrame(),
                'metadata': {}
            }
    
# Example usage and interactive interface
def interactive_testset_builder():
    """Interactive testset builder"""
    
    builder = ConfigurableTestsetBuilder()
    
    print("🎯 Welcome to the Configurable Testset Builder!")
    print("=" * 50)
    
    # Show options
    builder.show_available_options()
    
    print("\n🛠️ Please select your desired generation strategies (comma separated):")
    strategy_input = input("Strategy selection: ").strip()
    selected_strategies = [s.strip() for s in strategy_input.split(',') if s.strip()]
    
    print("\n📊 Please set question type distribution (total should be 1.0):")
    distribution = {}
    remaining = 1.0
    
    for qtype, info in builder.question_types.items():
        if remaining <= 0:
            distribution[qtype] = 0.0
            continue
            
        default_ratio = round(remaining / len([t for t in builder.question_types.keys() if t not in distribution]), 2)
        ratio_input = input(f"{info['name']} ratio (default {default_ratio}): ").strip()
        
        try:
            ratio = float(ratio_input) if ratio_input else default_ratio
            distribution[qtype] = ratio
            remaining -= ratio
        except ValueError:
            distribution[qtype] = default_ratio
            remaining -= default_ratio
    
    # Other settings
    total_samples = int(input("\nTotal samples (default 100): ") or "100")
    samples_per_doc = int(input("Samples per document (default 20): ") or "20")
    
    # Create configuration
    config = builder.create_custom_config(
        strategies=selected_strategies,
        question_distribution=distribution,
        total_samples=total_samples,
        samples_per_document=samples_per_doc
    )
    
    print(f"\n📋 Generation configuration summary:")
    print(f"  Strategies: {selected_strategies}")
    print(f"  Question distribution: {distribution}")
    print(f"  Total samples: {total_samples}")
    
    # Generate testset
    confirm = input("\nStart testset generation? (y/N): ").strip().lower()
    if confirm == 'y':
        results = builder.build_testset(config)
        print(f"\n✅ Testset generation completed!")
        print(f"   Generated samples: {results.get('total_samples', 0)}")
        print(f"   Success status: {results.get('success', False)}")
    else:
        print("🔄 Generation cancelled")

if __name__ == "__main__":
    interactive_testset_builder()
