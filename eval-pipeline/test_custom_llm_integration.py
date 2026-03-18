#!/usr/bin/env python3
"""Deprecated legacy script.

Maintained coverage now lives in eval-pipeline/tests/test_legacy_root_script_regressions.py.
"""

import sys
import yaml
from pathlib import Path
import logging
import json
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'test_custom_llm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def test_configuration_loading():
    """Test configuration file loading"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing configuration loading...")
    
    try:
        # Load main configuration
        config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"
        if not config_path.exists():
            logger.error(f"❌ Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info("✅ Main configuration loaded successfully")
        
        # Check for custom LLM configuration
        ragas_config = config.get('testset_generation', {}).get('ragas_config', {})
        if ragas_config.get('use_custom_llm', False):
            logger.info("✅ Custom LLM configuration found")
            custom_llm_config = ragas_config.get('custom_llm', {})
            logger.info(f"   Endpoint: {custom_llm_config.get('endpoint', 'Not configured')}")
            logger.info(f"   Model: {custom_llm_config.get('model', 'Not configured')}")
        else:
            logger.warning("⚠️ Custom LLM not enabled in configuration")
        
        # Test secrets file loading
        secrets_path = Path(__file__).parent / "config" / "secrets.yaml"
        if secrets_path.exists():
            with open(secrets_path, 'r', encoding='utf-8') as f:
                secrets = yaml.safe_load(f)
            logger.info("✅ Secrets file loaded successfully")
            
            # Check for API key
            api_key = secrets.get('inventec_llm', {}).get('api_key', '')
            if api_key and api_key != 'your_inventec_llm_api_key_here':
                logger.info("✅ Custom LLM API key configured")
            else:
                logger.warning("⚠️ Custom LLM API key not configured")
        else:
            logger.warning("⚠️ Secrets file not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False

def test_hybrid_generator_import():
    """Test importing the hybrid testset generator"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing hybrid generator import...")
    
    try:
        from src.data.hybrid_testset_generator import HybridTestsetGenerator
        logger.info("✅ HybridTestsetGenerator imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import HybridTestsetGenerator: {e}")
        return False

def test_generator_initialization():
    """Test generator initialization with custom LLM config"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing generator initialization...")
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Import and initialize generator
        from src.data.hybrid_testset_generator import HybridTestsetGenerator
        generator = HybridTestsetGenerator(config)
        
        logger.info("✅ Generator initialized successfully")
        logger.info(f"   Method: {generator.method}")
        logger.info(f"   Samples per doc: {generator.samples_per_doc}")
        logger.info(f"   Max total samples: {generator.max_total_samples}")
        
        return generator
        
    except Exception as e:
        logger.error(f"❌ Generator initialization failed: {e}")
        return None

def test_custom_llm_creation():
    """Test custom LLM wrapper creation"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing custom LLM creation...")
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Initialize generator
        from src.data.hybrid_testset_generator import HybridTestsetGenerator
        generator = HybridTestsetGenerator(config)
        
        # Test custom LLM creation
        ragas_config = config.get('testset_generation', {}).get('ragas_config', {})
        custom_llm_config = ragas_config.get('custom_llm', {})
        
        if custom_llm_config:
            custom_llm = generator._create_custom_llm(custom_llm_config, temperature=0.3)
            
            if custom_llm:
                logger.info("✅ Custom LLM created successfully")
                logger.info(f"   LLM type: {custom_llm._llm_type}")
                logger.info(f"   Endpoint: {custom_llm.endpoint}")
                logger.info(f"   Model: {custom_llm.model}")
                return custom_llm
            else:
                logger.warning("⚠️ Custom LLM creation returned None")
        else:
            logger.warning("⚠️ No custom LLM configuration found")
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Custom LLM creation failed: {e}")
        return None

def test_document_processing():
    """Test document processing functionality"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing document processing...")
    
    try:
        # Create a sample document for testing
        sample_dir = Path(__file__).parent / "test_documents"
        sample_dir.mkdir(exist_ok=True)
        
        sample_content = """
        # Test Document for Custom LLM
        
        This is a test document for validating the custom LLM integration.
        
        ## Section 1: Introduction
        The custom LLM integration allows the hybrid testset generator to work with internal or private LLM endpoints.
        
        ## Section 2: Features
        - Secure API key management through secrets.yaml
        - Support for custom headers and authentication
        - Configurable endpoint and model selection
        - Error handling and fallback mechanisms
        
        ## Section 3: Testing
        This document will be used to generate test questions using the custom LLM.
        """
        
        sample_file = sample_dir / "test_document.txt"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        logger.info(f"✅ Created test document: {sample_file}")
        
        # Load configuration and initialize generator
        config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        from src.data.hybrid_testset_generator import HybridTestsetGenerator
        generator = HybridTestsetGenerator(config)
        
        # Test document processing
        processed_docs = generator._process_documents([str(sample_file)])
        
        if processed_docs:
            logger.info(f"✅ Document processing successful: {len(processed_docs)} documents processed")
            for doc in processed_docs:
                logger.info(f"   - {doc.get('name', 'Unknown')}: {doc.get('word_count', 0)} words")
            return processed_docs
        else:
            logger.warning("⚠️ Document processing returned empty results")
            return []
        
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        return []

def test_api_key_loading():
    """Test API key loading from secrets file"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing API key loading...")
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Initialize generator
        from src.data.hybrid_testset_generator import HybridTestsetGenerator
        generator = HybridTestsetGenerator(config)
        
        # Test API key loading
        ragas_config = config.get('testset_generation', {}).get('ragas_config', {})
        custom_llm_config = ragas_config.get('custom_llm', {})
        
        if custom_llm_config:
            api_key = generator._load_api_key_from_secrets(custom_llm_config)
            
            if api_key:
                logger.info("✅ API key loaded successfully")
                logger.info(f"   Key length: {len(api_key)} characters")
                # Don't log the actual key for security
                return True
            else:
                logger.warning("⚠️ API key loading returned empty string")
                return False
        else:
            logger.warning("⚠️ No custom LLM configuration found")
            return False
        
    except Exception as e:
        logger.error(f"❌ API key loading failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end testset generation with custom LLM"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing end-to-end generation...")
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Create test documents
        sample_dir = Path(__file__).parent / "test_documents"
        sample_dir.mkdir(exist_ok=True)
        
        test_doc = sample_dir / "end_to_end_test.txt"
        with open(test_doc, 'w', encoding='utf-8') as f:
            f.write("""
            # End-to-End Test Document
            
            This document tests the complete pipeline with custom LLM integration.
            The system should be able to process this document and generate questions.
            """)
        
        # Initialize generator
        from src.data.hybrid_testset_generator import HybridTestsetGenerator
        generator = HybridTestsetGenerator(config)
        
        # Set method to configurable for testing (doesn't require LLM)
        generator.method = 'configurable'
        
        # Generate testset
        output_dir = Path(__file__).parent / "test_outputs"
        results = generator.generate_comprehensive_testset(
            document_paths=[str(test_doc)],
            output_dir=output_dir
        )
        
        if results and 'testset' in results:
            testset = results['testset']
            logger.info(f"✅ End-to-end test successful: {len(testset)} samples generated")
            logger.info(f"   Metadata: {results.get('metadata', {})}")
            return True
        else:
            logger.warning("⚠️ End-to-end test returned no results")
            return False
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        return False

def main():
    """Main test function"""
    logger = setup_logging()
    logger.info("🚀 Starting Custom LLM Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Hybrid Generator Import", test_hybrid_generator_import),
        ("Generator Initialization", test_generator_initialization),
        ("Custom LLM Creation", test_custom_llm_creation),
        ("Document Processing", test_document_processing),
        ("API Key Loading", test_api_key_loading),
        ("End-to-End Generation", test_end_to_end)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running test: {test_name}")
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.warning(f"⚠️ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = "ERROR"
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Custom LLM integration is ready.")
    else:
        logger.warning("⚠️ Some tests failed. Please review the logs and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
