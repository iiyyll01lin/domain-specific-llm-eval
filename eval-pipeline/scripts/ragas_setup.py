#!/usr/bin/env python3
"""
RAGAS Setup Module for Domain-Specific RAG Evaluation Pipeline

This module ensures RAGAS is properly available to the evaluation pipeline
by setting up the correct import paths and dependencies.
"""
import sys
import os
from pathlib import Path
import logging

def setup_ragas_environment():
    """
    Set up RAGAS environment for the evaluation pipeline.
    
    This function:
    1. Adds RAGAS source directory to Python path
    2. Verifies RAGAS dependencies are available
    3. Tests RAGAS imports to ensure everything works
    
    Returns:
        bool: True if RAGAS is successfully set up, False otherwise
    """
    
    # Get the pipeline root directory
    pipeline_root = Path(__file__).parent.parent
    workspace_root = pipeline_root.parent
    
    # Path to RAGAS source code in the submodule
    ragas_src_path = workspace_root / "ragas" / "ragas" / "src"
    
    print(f"🔍 Setting up RAGAS environment...")
    print(f"   Pipeline root: {pipeline_root}")
    print(f"   Workspace root: {workspace_root}")
    print(f"   RAGAS source path: {ragas_src_path}")
    
    # Check if RAGAS source exists
    if not ragas_src_path.exists():
        print(f"❌ RAGAS source not found at {ragas_src_path}")
        print("   Make sure the RAGAS submodule is properly initialized")
        return False
    
    # Add RAGAS to Python path
    ragas_src_str = str(ragas_src_path)
    if ragas_src_str not in sys.path:
        sys.path.insert(0, ragas_src_str)
        print(f"✅ Added RAGAS to Python path: {ragas_src_str}")
    else:
        print(f"✅ RAGAS already in Python path")
    
    # Test basic RAGAS import
    try:
        import ragas
        print(f"✅ RAGAS imported successfully")
        print(f"   Version: {ragas.__version__}")
        print(f"   Location: {ragas.__file__}")
        
        # Test critical RAGAS components
        test_imports = [
            ("ragas.testset", "TestsetGenerator"),
            ("ragas", "evaluate"),
            ("ragas.metrics", "context_precision"),
            ("ragas.metrics", "context_recall"), 
            ("ragas.metrics", "faithfulness"),
            ("ragas.metrics", "answer_relevancy")
        ]
        
        successful_imports = 0
        for module_name, component_name in test_imports:
            try:
                module = __import__(module_name, fromlist=[component_name])
                getattr(module, component_name)
                successful_imports += 1
                print(f"   ✅ {module_name}.{component_name}")
            except (ImportError, AttributeError) as e:
                print(f"   ⚠️  {module_name}.{component_name}: {e}")
        
        if successful_imports == len(test_imports):
            print("🎉 All RAGAS components available!")
            return True
        else:
            print(f"⚠️  {successful_imports}/{len(test_imports)} RAGAS components available")
            return successful_imports > 0  # Partial success is still usable
            
    except ImportError as e:
        print(f"❌ Failed to import RAGAS: {e}")
        
        # Check for missing dependencies
        missing_deps = []
        critical_deps = [
            "appdirs", "datasets", "langchain", "langchain_core",
            "langchain_community", "nest_asyncio", "pydantic", "diskcache"
        ]
        
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ Missing RAGAS dependencies: {', '.join(missing_deps)}")
            print("   Install with: pip install " + " ".join(missing_deps))
        
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error setting up RAGAS: {e}")
        return False

def check_ragas_availability():
    """
    Quick check if RAGAS is available for import.
    
    Returns:
        dict: Status information about RAGAS availability
    """
    status = {
        'available': False,
        'version': None,
        'location': None,
        'components': {},
        'errors': []
    }
    
    try:
        # Check if RAGAS path is set up
        ragas_src_path = Path(__file__).parent.parent.parent / "ragas" / "ragas" / "src"
        if ragas_src_path.exists() and str(ragas_src_path) not in sys.path:
            sys.path.insert(0, str(ragas_src_path))
        
        # Try importing RAGAS
        import ragas
        status['available'] = True
        status['version'] = getattr(ragas, '__version__', 'unknown')
        status['location'] = getattr(ragas, '__file__', 'unknown')
        
        # Test key components
        components_to_test = {
            'evaluate': 'ragas.evaluate',
            'TestsetGenerator': 'ragas.testset.TestsetGenerator',
            'context_precision': 'ragas.metrics.context_precision',
            'context_recall': 'ragas.metrics.context_recall',
            'faithfulness': 'ragas.metrics.faithfulness',
            'answer_relevancy': 'ragas.metrics.answer_relevancy'
        }
        
        for component, import_path in components_to_test.items():
            try:
                parts = import_path.split('.')
                module_path = '.'.join(parts[:-1])
                component_name = parts[-1]
                
                module = __import__(module_path, fromlist=[component_name])
                getattr(module, component_name)
                status['components'][component] = True
            except Exception as e:
                status['components'][component] = False
                status['errors'].append(f"{component}: {e}")
        
    except Exception as e:
        status['errors'].append(f"Main import failed: {e}")
    
    return status

if __name__ == "__main__":
    # Test the setup when run directly
    print("🧪 Testing RAGAS setup...")
    success = setup_ragas_environment()
    
    if success:
        print("\n🎉 RAGAS setup successful!")
        
        # Show detailed status
        status = check_ragas_availability()
        print(f"Version: {status['version']}")
        print(f"Available components: {sum(status['components'].values())}/{len(status['components'])}")
        
        if status['errors']:
            print("⚠️  Issues found:")
            for error in status['errors']:
                print(f"   - {error}")
    else:
        print("\n❌ RAGAS setup failed!")
        print("Please check the error messages above and resolve dependencies.")
        sys.exit(1)
