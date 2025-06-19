"""
Test script to demonstrate document-to-chunk ratios
Shows exactly how many knowledge base chunks will be generated from your raw documents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from document_loader import DocumentLoader
except ImportError:
    print("⚠️ document_loader.py not found. Creating simple demo...")
    DocumentLoader = None

def simple_chunk_text(text, chunk_size=1000, chunk_overlap=200, min_chunk_size=100):
    """Simple chunking function for demonstration"""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Find a good break point (sentence end)
        if end < len(text):
            # Look for sentence breaks near the end
            for i in range(min(50, chunk_size // 4)):
                if text[end - i] in '.!?':
                    end = end - i + 1
                    break
        
        chunk = text[start:end].strip()
        
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        
        start = end - chunk_overlap
        
        if start >= len(text):
            break
    
    return chunks

def test_chunking_ratios():
    """Test document chunking with different configurations"""
    
    # Test configurations
    test_configs = {
        'small_chunks': {
            'chunk_size': 500,
            'chunk_overlap': 100,
            'min_chunk_size': 50
        },
        'medium_chunks': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'min_chunk_size': 100
        },
        'large_chunks': {
            'chunk_size': 2000,
            'chunk_overlap': 400,
            'min_chunk_size': 200
        }
    }
    
    # Sample documents of different sizes
    sample_documents = {
        'short_doc': "This is a short document with minimal content. " * 20,  # ~1,000 chars
        'medium_doc': "This is a medium-length document with substantial content. " * 100,  # ~5,900 chars
        'long_doc': "This is a very long document with extensive content and detailed information. " * 300,  # ~23,700 chars
    }
    
    print("📊 Document Chunking Analysis")
    print("=" * 60)
    
    for config_name, chunk_config in test_configs.items():
        print(f"\n🔧 Configuration: {config_name.upper()}")
        print(f"   Chunk Size: {chunk_config['chunk_size']} chars")
        print(f"   Overlap: {chunk_config['chunk_overlap']} chars")
        print(f"   Min Size: {chunk_config['min_chunk_size']} chars")
        print("-" * 40)
        
        # Create mock loader or use simple function
        if DocumentLoader:
            config = {
                'custom_data': {
                    'processing': chunk_config
                }
            }
            loader = DocumentLoader(config)
            chunks_func = loader.chunk_text
        else:
            chunks_func = lambda text: simple_chunk_text(
                text, 
                chunk_config['chunk_size'], 
                chunk_config['chunk_overlap'], 
                chunk_config['min_chunk_size']
            )
        
        for doc_name, doc_content in sample_documents.items():
            chunks = chunks_func(doc_content)
            ratio = f"1:{len(chunks)}"
            
            print(f"   📄 {doc_name:12} ({len(doc_content):6} chars) → {len(chunks):3} chunks | Ratio: {ratio}")
    
    print(f"\n🎯 Key Insights:")
    print(f"   • 1 Raw Document → Multiple Knowledge Base Chunks")
    print(f"   • Smaller chunk_size = MORE chunks per document")
    print(f"   • Larger chunk_size = FEWER chunks per document")
    print(f"   • chunk_overlap helps maintain context between chunks")

def test_real_batch_scenario():
    """Simulate a real batch processing scenario"""
    print(f"\n\n🏢 REAL BATCH SCENARIO SIMULATION")
    print("=" * 60)
    
    # Simulate a realistic batch of documents
    batch_documents = [
        {'name': 'Technical Manual PDF', 'size': 50000, 'pages': 25},
        {'name': 'Research Paper PDF', 'size': 30000, 'pages': 15},
        {'name': 'Specification DOCX', 'size': 15000, 'pages': 8},
        {'name': 'Knowledge Article TXT', 'size': 8000, 'pages': 4},
        {'name': 'User Guide PDF', 'size': 40000, 'pages': 20},
        {'name': 'Process Document DOCX', 'size': 12000, 'pages': 6},
        {'name': 'Training Material PDF', 'size': 60000, 'pages': 30},        {'name': 'Policy Document TXT', 'size': 5000, 'pages': 3},
    ]
    
    # Standard configuration
    chunk_size = 1000
    chunk_overlap = 200
    min_chunk_size = 100
    
    total_input_docs = len(batch_documents)
    total_output_chunks = 0
    
    print(f"📂 Processing {total_input_docs} raw documents:")
    print("-" * 60)
    
    for doc in batch_documents:
        # Estimate chunks based on size (more accurate estimation)
        estimated_chunks = max(1, (doc['size'] - chunk_overlap) // (chunk_size - chunk_overlap))
        total_output_chunks += estimated_chunks
        
        print(f"📄 {doc['name']:25} ({doc['size']:6} chars) → ~{estimated_chunks:3} chunks")
    
    print("-" * 60)
    print(f"📊 BATCH SUMMARY:")
    print(f"   Input:  {total_input_docs} raw documents")
    print(f"   Output: ~{total_output_chunks} knowledge base chunks")
    print(f"   Ratio:  1 raw doc → {total_output_chunks//total_input_docs:.1f} chunks (average)")
    print(f"   Expansion Factor: ~{total_output_chunks/total_input_docs:.1f}x")

def demonstrate_configuration_control():
    """Show how to control the output quantity"""
    print(f"\n\n⚙️ CONTROLLING OUTPUT QUANTITY")
    print("=" * 60)
    
    sample_doc = "This is a sample document. " * 200  # ~5,400 chars
    
    configurations = [
        {'name': 'Maximum Chunks', 'chunk_size': 300, 'chunk_overlap': 50},
        {'name': 'Balanced Chunks', 'chunk_size': 1000, 'chunk_overlap': 200},
        {'name': 'Minimum Chunks', 'chunk_size': 3000, 'chunk_overlap': 500},
        {'name': 'Single Chunk', 'chunk_size': 10000, 'chunk_overlap': 0},
    ]
    
    print(f"Sample document size: {len(sample_doc)} characters")
    print("-" * 40)
    
    for cfg in configurations:
        chunks = simple_chunk_text(
            sample_doc,
            cfg['chunk_size'],
            cfg['chunk_overlap'],
            50  # min_chunk_size
        )
        
        print(f"🔧 {cfg['name']:15} (size:{cfg['chunk_size']:4}) → {len(chunks):2} chunks")
    
    print(f"\n💡 Configuration Strategy:")
    print(f"   • For MORE knowledge base files: Use smaller chunk_size (300-500)")
    print(f"   • For FEWER knowledge base files: Use larger chunk_size (2000-3000)")
    print(f"   • For balanced processing: Use default (1000)")

if __name__ == "__main__":
    test_chunking_ratios()
    test_real_batch_scenario()
    demonstrate_configuration_control()
