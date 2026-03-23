# DEPRECATED: This root-level script has been migrated to
# eval-pipeline/tests/test_legacy_final_batch_migration.py.
# It is retained for reference only and should not be executed directly.
# See LIMITATIONS_PROGRESS_20260317.md for migration history.

#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility smoke entry point for the batch evaluation pipeline."""

from pathlib import Path
import sys


def main() -> int:
    pipeline_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(pipeline_dir))

    from run_pipeline import DEFAULT_STAGE_CONFIGS, validate_ragas_setup

    print("🚀 RAGAS integration smoke check")
    validate_ragas_setup()

    for stage, config_path in DEFAULT_STAGE_CONFIGS.items():
        print(f"✅ default config for {stage}: {config_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())#!/usr/bin/env python3
"""
Comprehensive RAGAS Integration Test
Tests the complete pipeline from CSV input to RAGAS testset generation and evaluation
"""

import json
import sys
import yaml
import logging
from pathlib import Path
import pandas as pd
import requests
from typing import Dict, List, Any

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

class RagasIntegrationTester:
    """Test suite for RAGAS integration with custom LLM API"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = config_path
        self.config = None
        self.logger = self._setup_logging()
        self.test_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for test output"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_config(self) -> bool:
        """Load pipeline configuration"""
        try:
            self.logger.info(f"📄 Loading configuration from: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self.logger.info("✅ Configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            return False
    
    def test_api_connectivity(self) -> bool:
        """Test 1: API connectivity with your working endpoint"""
        self.logger.info("\n🧪 TEST 1: API Connectivity")
        self.logger.info("-" * 40)
        
        try:
            # Get API configuration
            testset_config = self.config.get('testset_generation', {})
            ragas_config = testset_config.get('ragas_config', {})
            custom_llm = ragas_config.get('custom_llm', {})
            
            endpoint = custom_llm.get('endpoint')
            api_key = custom_llm.get('api_key')
            model = custom_llm.get('model')
            headers = custom_llm.get('headers', {})
            
            self.logger.info(f"   Endpoint: {endpoint}")
            self.logger.info(f"   Model: {model}")
            
            # Prepare test request (matching your working format)
            test_headers = headers.copy()
            test_headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "messages": [
                    {"content": "", "role": "system"},
                    {"content": "Hello, this is a test message for RAGAS integration.", "role": "user"}
                ],
                "model": model,
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 100
            }
            
            # Make request
            response = requests.post(endpoint, headers=test_headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                self.logger.info(f"✅ API Test Success!")
                self.logger.info(f"   Response: {content[:100]}...")
                self.test_results['api_connectivity'] = True
                return True
            else:
                self.logger.error(f"❌ API Test Failed: {response.status_code} - {response.text}")
                self.test_results['api_connectivity'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ API connectivity test failed: {e}")
            self.test_results['api_connectivity'] = False
            return False
    
    def test_csv_data_loading(self) -> bool:
        """Test 2: CSV data loading and preprocessing"""
        self.logger.info("\n🧪 TEST 2: CSV Data Loading")
        self.logger.info("-" * 40)
        
        try:
            # Get CSV configuration
            data_sources = self.config.get('data_sources', {})
            csv_config = data_sources.get('csv', {})
            csv_files = csv_config.get('csv_files', [])
            
            if not csv_files:
                self.logger.error("❌ No CSV files configured")
                self.test_results['csv_loading'] = False
                return False
            
            total_rows = 0
            valid_files = 0
            
            for csv_file in csv_files:
                self.logger.info(f"   Testing CSV file: {csv_file}")
                
                if Path(csv_file).exists():
                    df = pd.read_csv(csv_file)
                    rows = len(df)
                    total_rows += rows
                    valid_files += 1
                    
                    self.logger.info(f"   ✅ {csv_file}: {rows} rows")
                    self.logger.info(f"   📋 Columns: {list(df.columns)}")
                    
                    # Test JSON content extraction
                    if 'content' in df.columns:
                        sample_content = df['content'].iloc[0] if len(df) > 0 else ""
                        try:
                            content_json = json.loads(sample_content)
                            text_content = content_json.get('text', '')
                            self.logger.info(f"   📄 Sample content length: {len(text_content)} chars")
                        except (json.JSONDecodeError, AttributeError):
                            self.logger.info(f"   📄 Sample content (non-JSON): {len(str(sample_content))} chars")
                    
                else:
                    self.logger.warning(f"   ⚠️ File not found: {csv_file}")
            
            if valid_files > 0:
                self.logger.info(f"✅ CSV loading test passed: {valid_files} files, {total_rows} total rows")
                self.test_results['csv_loading'] = True
                return True
            else:
                self.logger.error("❌ No valid CSV files found")
                self.test_results['csv_loading'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ CSV loading test failed: {e}")
            self.test_results['csv_loading'] = False
            return False
    
    def test_custom_llm_wrapper(self) -> bool:
        """Test 3: Custom LLM wrapper creation"""
        self.logger.info("\n🧪 TEST 3: Custom LLM Wrapper")
        self.logger.info("-" * 40)
        
        try:
            # Import the hybrid generator
            from data.hybrid_testset_generator import HybridTestsetGenerator
            
            # Get testset generation config
            testset_config = self.config.get('testset_generation', {})
            
            # Create generator
            generator = HybridTestsetGenerator(testset_config)
            
            # Test custom LLM creation
            ragas_config = testset_config.get('ragas_config', {})
            custom_llm_config = ragas_config.get('custom_llm', {})
            
            if custom_llm_config:
                custom_llm = generator._create_custom_llm(custom_llm_config, temperature=0.3)
                
                if custom_llm:
                    self.logger.info("✅ Custom LLM created successfully")
                    self.logger.info(f"   LLM type: {type(custom_llm)}")
                    
                    # Check if it's properly wrapped for RAGAS
                    from ragas.llms import LangchainLLMWrapper
                    if isinstance(custom_llm, LangchainLLMWrapper):
                        self.logger.info("✅ Properly wrapped with LangchainLLMWrapper for RAGAS")
                        self.test_results['custom_llm'] = True
                        return True
                    else:
                        self.logger.error("❌ Not wrapped with LangchainLLMWrapper")
                        self.test_results['custom_llm'] = False
                        return False
                else:
                    self.logger.error("❌ Custom LLM creation failed")
                    self.test_results['custom_llm'] = False
                    return False
            else:
                self.logger.error("❌ No custom LLM config found")
                self.test_results['custom_llm'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Custom LLM wrapper test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['custom_llm'] = False
            return False
    
    def test_csv_to_ragas_converter(self) -> bool:
        """Test 4: CSV-to-RAGAS converter"""
        self.logger.info("\n🧪 TEST 4: CSV-to-RAGAS Converter")
        self.logger.info("-" * 40)
        
        try:
            # Import the converter
            from data.csv_ragas_converter import CSVToRagasConverter
            
            # Get testset generation config
            testset_config = self.config.get('testset_generation', {})
            
            # Create converter
            converter = CSVToRagasConverter(testset_config)
            
            # Test CSV loading
            df = converter.load_csv_data()
            self.logger.info(f"✅ CSV data loaded: {len(df)} rows")
            
            # Test document conversion
            documents = converter.csv_to_documents(df)
            self.logger.info(f"✅ Documents created: {len(documents)} documents")
            
            if len(documents) > 0:
                # Show sample document
                sample_doc = documents[0]
                content_length = len(sample_doc.page_content)
                metadata_keys = list(sample_doc.metadata.keys())
                
                self.logger.info(f"   Sample document content: {content_length} chars")
                self.logger.info(f"   Sample metadata keys: {metadata_keys}")
                self.logger.info(f"   Content preview: {sample_doc.page_content[:200]}...")
                
                self.test_results['csv_converter'] = True
                return True
            else:
                self.logger.error("❌ No documents created from CSV")
                self.test_results['csv_converter'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ CSV-to-RAGAS converter test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['csv_converter'] = False
            return False
    
    def test_ragas_testset_generation(self) -> bool:
        """Test 5: End-to-end RAGAS testset generation"""
        self.logger.info("\n🧪 TEST 5: RAGAS Testset Generation")
        self.logger.info("-" * 40)
        
        try:
            # Import required components
            from data.hybrid_testset_generator import HybridTestsetGenerator
            
            # Get testset generation config
            testset_config = self.config.get('testset_generation', {})
            
            # Set CSV-to-RAGAS mode for testing
            testset_config['csv_to_ragas_mode'] = True
            testset_config['max_total_samples'] = 5  # Small test
            
            # Create generator
            generator = HybridTestsetGenerator(testset_config)
            
            # Test the CSV-to-RAGAS integration
            output_dir = Path("outputs/test_ragas")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = generator._generate_with_csv_ragas_integration([], output_dir)
            
            if result.get('success', False):
                testset_df = result.get('testset', pd.DataFrame())
                samples_generated = result.get('samples_generated', 0)
                
                self.logger.info(f"✅ RAGAS testset generation successful!")
                self.logger.info(f"   Samples generated: {samples_generated}")
                self.logger.info(f"   DataFrame shape: {testset_df.shape}")
                
                if len(testset_df) > 0:
                    self.logger.info(f"   Columns: {list(testset_df.columns)}")
                    
                    # Show sample question
                    if 'question' in testset_df.columns:
                        sample_question = testset_df['question'].iloc[0]
                        self.logger.info(f"   Sample question: {sample_question[:200]}...")
                    
                    self.test_results['ragas_generation'] = True
                    return True
                else:
                    self.logger.error("❌ No samples generated")
                    self.test_results['ragas_generation'] = False
                    return False
            else:
                error = result.get('error', 'Unknown error')
                self.logger.error(f"❌ RAGAS generation failed: {error}")
                self.test_results['ragas_generation'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ RAGAS testset generation test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['ragas_generation'] = False
            return False
    
    def test_ragas_evaluation_setup(self) -> bool:
        """Test 6: RAGAS evaluation metrics setup"""
        self.logger.info("\n🧪 TEST 6: RAGAS Evaluation Setup")
        self.logger.info("-" * 40)
        
        try:
            # Import RAGAS metrics
            from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # Get evaluation config
            eval_config = self.config.get('evaluation', {})
            ragas_config = eval_config.get('ragas_metrics', {})
            llm_config = ragas_config.get('llm', {})
            
            # Create custom LLM for evaluation
            from data.hybrid_testset_generator import HybridTestsetGenerator
            generator = HybridTestsetGenerator({})
            
            custom_llm_config = {
                'endpoint': llm_config.get('endpoint'),
                'api_key': llm_config.get('api_key'),
                'model': llm_config.get('model_name', 'gpt-4o'),
                'temperature': llm_config.get('temperature', 0.1),
                'max_tokens': llm_config.get('max_length', 512),
                'headers': llm_config.get('headers', {})
            }
            
            custom_llm = generator._create_custom_llm(custom_llm_config, temperature=0.1)
            
            if not custom_llm:
                self.logger.error("❌ Failed to create custom LLM for evaluation")
                self.test_results['ragas_evaluation'] = False
                return False
            
            # Create embeddings
            embeddings_config = ragas_config.get('embeddings', {})
            embeddings_model = embeddings_config.get('model_name', 'all-MiniLM-L6-v2')
            embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=embeddings_model)
            )
            
            # Test setting LLM and embeddings on metrics
            test_metrics = {
                'faithfulness': faithfulness,
                'context_precision': context_precision,
                'context_recall': context_recall,
                'answer_relevancy': answer_relevancy
            }
            
            configured_metrics = 0
            for metric_name, metric in test_metrics.items():
                try:
                    # Set custom LLM
                    if hasattr(metric, 'llm'):
                        metric.llm = custom_llm
                        self.logger.info(f"   ✅ Set custom LLM for {metric_name}")
                        configured_metrics += 1
                    
                    # Set embeddings
                    if hasattr(metric, 'embeddings'):
                        metric.embeddings = embeddings
                        self.logger.info(f"   ✅ Set embeddings for {metric_name}")
                    
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Failed to configure {metric_name}: {e}")
            
            if configured_metrics > 0:
                self.logger.info(f"✅ RAGAS evaluation setup successful: {configured_metrics} metrics configured")
                self.test_results['ragas_evaluation'] = True
                return True
            else:
                self.logger.error("❌ No metrics configured successfully")
                self.test_results['ragas_evaluation'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ RAGAS evaluation setup test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['ragas_evaluation'] = False
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests"""
        self.logger.info("🚀 Starting RAGAS Integration Test Suite")
        self.logger.info("=" * 60)
        
        # Load configuration first
        if not self.load_config():
            return {"config_loading": False}
        
        # Run all tests
        tests = [
            ("API Connectivity", self.test_api_connectivity),
            ("CSV Data Loading", self.test_csv_data_loading),
            ("Custom LLM Wrapper", self.test_custom_llm_wrapper),
            ("CSV-to-RAGAS Converter", self.test_csv_to_ragas_converter),
            ("RAGAS Testset Generation", self.test_ragas_testset_generation),
            ("RAGAS Evaluation Setup", self.test_ragas_evaluation_setup)
        ]
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                if not success:
                    self.logger.warning(f"⚠️ Test '{test_name}' failed - check logs above")
            except Exception as e:
                self.logger.error(f"❌ Test '{test_name}' crashed: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = False
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def print_test_summary(self):
        """Print test results summary"""
        self.logger.info("\n📊 Test Results Summary")
        self.logger.info("=" * 50)
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, success in self.test_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            self.logger.info(f"{status} {test_name.replace('_', ' ').title()}")
            if success:
                passed += 1
        
        self.logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 All tests passed! Your RAGAS integration is working correctly.")
            self.logger.info("\n🚀 Ready to run:")
            self.logger.info("   python run_pipeline.py --stage testset-generation")
        else:
            self.logger.warning("⚠️ Some tests failed. Please check the configuration and API settings.")


def main():
    """Run the RAGAS integration test suite"""
    tester = RagasIntegrationTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
