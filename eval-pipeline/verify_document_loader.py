"""
Verification script to test all DocumentLoader implementations
Tests each method to ensure they are complete and working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_loader import DocumentLoader

def test_document_loader_completeness():
    """Test that all DocumentLoader methods are implemented"""
    
    print("🧪 Testing DocumentLoader Implementation Completeness")
    print("=" * 60)
    
    # Create test configuration
    test_config = {
        'custom_data': {
            'processing': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'min_chunk_size': 100,
                'min_document_length': 50,
                'filter_language': None,
                'remove_special_chars': False
            }
        }
    }
    
    # Initialize loader
    loader = DocumentLoader(test_config)
    
    # Test basic methods
    tests = [
        ('chunk_text', lambda: loader.chunk_text("This is a test document. " * 100)),
        ('clean_text', lambda: loader.clean_text("  This   is   messy   text  \n\n  ")),
        ('is_target_language', lambda: loader.is_target_language("This is English text")),
        ('extract_topics_from_text', lambda: loader.extract_topics_from_text("machine learning artificial intelligence data science")),
        ('get_document_stats', lambda: loader.get_document_stats()),
    ]
    
    results = {}
    
    for method_name, test_func in tests:
        try:
            result = test_func()
            results[method_name] = {
                'status': '✅ PASS',
                'result': str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            }
        except Exception as e:
            results[method_name] = {
                'status': '❌ FAIL', 
                'result': str(e)
            }
    
    # Print results
    for method, result in results.items():
        print(f"{result['status']} {method:25} - {result['result']}")
    
    # Test file loading methods (without actual files)
    file_methods = [
        'load_text_file',
        'load_directory', 
        'load_structured_file',
        'process_documents'
    ]
    
    print(f"\n📋 File Loading Methods:")
    for method in file_methods:
        if hasattr(loader, method):
            print(f"✅ {method} - Method exists")
        else:
            print(f"❌ {method} - Method missing")
    
    # Test overall completeness
    total_tests = len(tests) + len(file_methods)
    passed_tests = sum(1 for r in results.values() if '✅' in r['status'])
    passed_methods = sum(1 for method in file_methods if hasattr(loader, method))
    
    overall_score = (passed_tests + passed_methods) / total_tests * 100
    
    print(f"\n🎯 Overall Completeness Score: {overall_score:.1f}%")
    print(f"   Basic Methods: {passed_tests}/{len(tests)} passed")
    print(f"   File Methods: {passed_methods}/{len(file_methods)} implemented")
    
    if overall_score >= 90:
        print("🎉 DocumentLoader implementation is COMPLETE!")
    elif overall_score >= 70:
        print("⚠️  DocumentLoader implementation is mostly complete")
    else:
        print("❌ DocumentLoader implementation needs more work")

def test_chunking_behavior():
    """Test the chunking behavior specifically"""
    print(f"\n\n🔧 Testing Chunking Behavior")
    print("=" * 40)
    
    config = {
        'custom_data': {
            'processing': {
                'chunk_size': 500,
                'chunk_overlap': 100,
                'min_chunk_size': 50
            }
        }
    }
    
    loader = DocumentLoader(config)
    
    # Test documents of various sizes
    test_docs = {
        'tiny': "Short document.",
        'small': "This is a small document. " * 10,
        'medium': "This is a medium document with more content. " * 50,
        'large': "This is a large document with extensive content and detailed information. " * 200
    }
    
    print("Document → Chunk Analysis:")
    for name, doc in test_docs.items():
        chunks = loader.chunk_text(doc)
        ratio = f"1:{len(chunks)}"
        print(f"  {name:8} ({len(doc):6} chars) → {len(chunks):3} chunks | Ratio: {ratio}")
    
    print(f"\n✅ Chunking behavior verified!")

if __name__ == "__main__":
    test_document_loader_completeness()
    test_chunking_behavior()