#!/usr/bin/env python3
"""
Test script for Custom Document Loading
Demonstrates how to use the enhanced dataset generator with your own documents
"""

import os
import sys
from pathlib import Path

def create_sample_documents():
    """Create sample documents for testing"""
    
    # Create documents directory
    docs_dir = Path("sample_documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Create sample text files
    sample_texts = [
        """
        Artificial Intelligence and Machine Learning
        
        Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. Machine learning is a subset of AI that focuses on the ability of machines to receive data and learn for themselves without being explicitly programmed.
        
        Key components of machine learning include:
        - Supervised learning: Learning with labeled training data
        - Unsupervised learning: Finding patterns in data without labels  
        - Reinforcement learning: Learning through rewards and punishments
        - Neural networks: Computing systems inspired by biological neural networks
        
        Applications of AI and ML include natural language processing, computer vision, robotics, and autonomous vehicles.
        """,
        
        """
        Cloud Computing Architecture
        
        Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the Internet ("the cloud") to offer faster innovation, flexible resources, and economies of scale.
        
        Main service models:
        - Infrastructure as a Service (IaaS): Provides virtualized computing resources
        - Platform as a Service (PaaS): Provides a platform for developing applications
        - Software as a Service (SaaS): Provides access to software applications
        
        Deployment models include public cloud, private cloud, hybrid cloud, and multi-cloud environments. Key benefits include cost reduction, scalability, reliability, and global accessibility.
        """,
        
        """
        Cybersecurity Best Practices
        
        Cybersecurity involves protecting systems, networks, and programs from digital attacks. These cyberattacks are usually aimed at accessing, changing, or destroying sensitive information; extorting money from users; or interrupting normal business processes.
        
        Essential security measures:
        - Multi-factor authentication (MFA)
        - Regular software updates and patches
        - Strong password policies
        - Network segmentation
        - Employee security training
        - Backup and recovery procedures
        - Incident response planning
        
        Common threats include malware, phishing, ransomware, social engineering, and advanced persistent threats (APTs).
        """
    ]
    
    filenames = [
        "ai_machine_learning.txt",
        "cloud_computing.txt", 
        "cybersecurity.txt"
    ]
    
    for text, filename in zip(sample_texts, filenames):
        with open(docs_dir / filename, 'w', encoding='utf-8') as f:
            f.write(text.strip())
    
    print(f"✅ Created {len(sample_texts)} sample documents in {docs_dir}/")
    return docs_dir

def create_test_config(docs_dir):
    """Create a test configuration file"""
    
    config_content = f"""# Test Configuration for Custom Document Processing
mode: 'local'

custom_data:
  enabled: true
  
  data_sources:
    text_files:
      - '{docs_dir}/ai_machine_learning.txt'
      - '{docs_dir}/cloud_computing.txt' 
      - '{docs_dir}/cybersecurity.txt'
    
    directories:
      - '{docs_dir}/'
  
  processing:
    chunk_size: 800
    chunk_overlap: 100
    min_chunk_size: 50
    remove_extra_whitespace: true
  
  topic_extraction:
    enabled: true
    method: 'keybert'
    max_topics_per_document: 4
  
  question_templates:
    - "What does the document explain about {{}}?"
    - "According to the text, what is {{}}?"
    - "How is {{}} described in the document?"
    - "What are the key aspects of {{}} mentioned?"

dataset:
  num_samples: 15
  output_file: 'test_custom_dataset.xlsx'

logging:
  level: 'INFO'
  show_progress: true
"""
    
    config_file = "config_test.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Created test configuration: {config_file}")
    return config_file

def test_document_loading():
    """Test the document loading functionality"""
    print("🧪 Testing Custom Document Loading")
    print("=" * 50)
    
    # Create sample documents
    docs_dir = create_sample_documents()
    
    # Create test configuration
    config_file = create_test_config(docs_dir)
    
    # Test document loader
    try:
        from document_loader import DocumentLoader
        import yaml
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        loader = DocumentLoader(config)
        documents, metadata = loader.load_all_documents()
        
        print(f"\n📊 Loading Results:")
        print(f"Loaded {len(documents)} document chunks")
        
        if documents:
            print(f"\nSample document chunk:")
            print("-" * 40)
            print(documents[0][:200] + "..." if len(documents[0]) > 200 else documents[0])
            
            topics = loader.get_topics_from_metadata()
            if topics:
                print(f"\nExtracted topics: {', '.join(topics[:5])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document loading test failed: {e}")
        return False

def test_dataset_generation():
    """Test the complete dataset generation"""
    print("\n🔬 Testing Dataset Generation with Custom Documents")
    print("=" * 60)
    
    try:
        from generate_dataset_configurable import ConfigurableDatasetGenerator
        
        # Generate dataset using test config
        generator = ConfigurableDatasetGenerator("config_test.yaml")
        dataset = generator.generate_dataset(10)
        
        # Save dataset
        output_file = generator.save_dataset(dataset, "test_output.xlsx")
        
        # Print summary
        generator.print_summary(dataset)
        
        print(f"\n📄 Sample generated questions:")
        for i, row in dataset.head(3).iterrows():
            print(f"  Q{i+1}: {row['question']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Custom Document Integration Test Suite")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "document_loader.py",
        "generate_dataset_configurable.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return
    
    # Run tests
    tests = [
        ("Document Loading", test_document_loading),
        ("Dataset Generation", test_dataset_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name} test {'passed' if result else 'failed'}")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📋 Test Summary:")
    print("=" * 30)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your custom document integration is working.")
        print("\nNext steps:")
        print("1. Replace sample documents with your actual documents")
        print("2. Update config_test.yaml with your document paths")
        print("3. Run: python generate_dataset_configurable.py")
        print("4. Use the generated dataset with your evaluation pipeline")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("Make sure all dependencies are installed:")
        print("pip install PyPDF2 pdfplumber python-docx langdetect keybert")

if __name__ == "__main__":
    main()
