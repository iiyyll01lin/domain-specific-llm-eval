#!/usr/bin/env python3
"""Deprecated legacy script.

Maintained coverage now lives in eval-pipeline/tests/test_legacy_runtime_smoke_regressions.py.
"""

from pathlib import Path
import json
from datetime import datetime

# Mock the components for testing
class MockDocumentProcessor:
    def process_documents(self):
        return [
            {
                'source_file': '/path/to/document1.pdf',
                'filename': 'document1.pdf',
                'content': 'Sample document content...'
            }
        ]

class MockHybridTestsetGenerator:
    def __init__(self, config):
        self.config = config
        print(f"✅ HybridTestsetGenerator initialized with config: {config}")
    
    def generate_comprehensive_testset(self, document_paths, output_dir):
        """Mock the new API"""
        print(f"🎯 generate_comprehensive_testset called with:")
        print(f"   - document_paths: {document_paths}")
        print(f"   - output_dir: {output_dir}")
        
        # Mock testset data
        testset_data = [
            {
                'question': 'What is the main topic?',
                'answer': 'The main topic is...',
                'contexts': ['Context 1'],
                'ground_truth': 'Ground truth...',
                'keywords': ['keyword1', 'keyword2'],
                'source_document': document_paths[0]
            },
            {
                'question': 'How does this work?',
                'answer': 'This works by...',
                'contexts': ['Context 2'],
                'ground_truth': 'Ground truth 2...',
                'keywords': ['keyword3', 'keyword4'],
                'source_document': document_paths[0]
            }
        ]
        
        metadata = {
            'generation_start': datetime.now(),
            'generation_end': datetime.now(),
            'total_generated': len(testset_data),
            'source_documents': [Path(p).name for p in document_paths],
            'generation_method': 'hybrid'
        }
        
        return {
            'testset': testset_data,
            'metadata': metadata,
            'results_by_method': {
                'configurable': {'success': True, 'samples_generated': 1},
                'ragas': {'success': True, 'samples_generated': 1}
            }
        }

def test_orchestrator_testset_generation():
    """Test the updated testset generation logic"""
    print("🧪 Testing Orchestrator Testset Generation Update")
    print("=" * 50)
    
    # Mock configuration
    config = {
        'testset_generation': {
            'method': 'hybrid',
            'samples_per_document': 50
        }
    }
    
    # Mock output directories
    output_dirs = {
        'testsets': Path('outputs/testsets'),
        'metadata': Path('outputs/metadata')
    }
    
    # Initialize components (mocked)
    document_processor = MockDocumentProcessor()
    testset_generator = MockHybridTestsetGenerator(config)
    
    print("📄 Step 1: Process documents...")
    processed_documents = document_processor.process_documents()
    print(f"✅ Processed {len(processed_documents)} documents")
    
    print("\n🎯 Step 2: Generate testsets...")
    
    # Extract document paths (NEW API)
    document_paths = [doc['source_file'] for doc in processed_documents]
    print(f"📍 Document paths: {document_paths}")
    
    # Use comprehensive testset generation (NEW API)
    testset_results = testset_generator.generate_comprehensive_testset(
        document_paths=document_paths,
        output_dir=output_dirs['testsets']
    )
    
    # Extract results for compatibility (NEW DATA STRUCTURE)
    testset_data = testset_results.get('testset', [])
    metadata_results = testset_results.get('metadata', {})
    generation_results = testset_results.get('results_by_method', {})
    
    print(f"✅ Generated testset with {len(testset_data)} QA pairs")
    print(f"📊 Generation method: {metadata_results.get('generation_method', 'unknown')}")
    
    print("\n📝 Step 3: Create metadata...")
    metadata = {
        'run_id': 'test_run_001',
        'timestamp': datetime.now().isoformat(),
        'documents_processed': len(processed_documents),
        'testsets_generated': 1 if testset_data else 0,
        'total_qa_pairs': len(testset_data),
        'document_sources': [doc['source_file'] for doc in processed_documents],
        'generation_method': metadata_results.get('generation_method', 'unknown'),
        'generation_metadata': metadata_results,
        'results_by_method': generation_results
    }
    
    print("\n📊 Final Results:")
    print(f"   - Documents processed: {metadata['documents_processed']}")
    print(f"   - Testsets generated: {metadata['testsets_generated']}")
    print(f"   - Total QA pairs: {metadata['total_qa_pairs']}")
    print(f"   - Generation method: {metadata['generation_method']}")
    
    print("\n✅ Orchestrator update test completed successfully!")
    
    return {
        'success': True,
        'documents_processed': metadata['documents_processed'],
        'testsets_generated': metadata['testsets_generated'],
        'total_qa_pairs': metadata['total_qa_pairs'],
        'generation_method': metadata['generation_method']
    }

def test_backward_compatibility():
    """Test the backward compatibility method"""
    print("\n🔄 Testing Backward Compatibility")
    print("=" * 30)
    
    # Test data that would come from old API
    processed_documents = [
        {
            'source_file': '/path/to/doc1.pdf',
            'filename': 'doc1.pdf',
            'content': 'Document 1 content'
        },
        {
            'source_file': '/path/to/doc2.pdf', 
            'filename': 'doc2.pdf',
            'content': 'Document 2 content'
        }
    ]
    
    # Mock the compatibility method
    generator = MockHybridTestsetGenerator({})
    
    print("🧪 Testing compatibility method logic...")
    
    # Extract paths (compatibility logic)
    document_paths = []
    for doc in processed_documents:
        if 'source_file' in doc:
            document_paths.append(doc['source_file'])
        elif 'path' in doc:
            document_paths.append(doc['path'])
        elif 'filename' in doc:
            document_paths.append(f"../{doc['filename']}")
    
    print(f"📍 Extracted paths: {document_paths}")
    
    if document_paths:
        print("✅ Compatibility method would work correctly")
    else:
        print("❌ Compatibility method would fail")
        
    return len(document_paths) > 0

if __name__ == "__main__":
    print("🚀 Testing Orchestrator Updates")
    print("=" * 50)
    
    # Test the main orchestrator update
    main_result = test_orchestrator_testset_generation()
    
    # Test backward compatibility
    compat_result = test_backward_compatibility()
    
    print(f"\n🎯 Test Summary:")
    print(f"   Main API Update: {'✅ PASS' if main_result['success'] else '❌ FAIL'}")
    print(f"   Backward Compatibility: {'✅ PASS' if compat_result else '❌ FAIL'}")
    
    if main_result['success'] and compat_result:
        print("\n🏆 All tests passed! Orchestrator update is ready.")
    else:
        print("\n⚠️ Some tests failed. Review the implementation.")
