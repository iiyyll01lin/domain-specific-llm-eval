#!/usr/bin/env python3
"""
Quick test script to verify the fixes work
"""
import sys
import os
from pathlib import Path

# Add source paths
sys.path.append(str(Path(__file__).parent / "src"))

def test_package_imports():
    """Test that all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    packages_to_test = [
        ("PyYAML", "yaml"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("openpyxl", "openpyxl"),
    ]
    
    success_count = 0
    for package_name, import_name in packages_to_test:
        try:
            __import__(import_name)
            print(f"✅ {package_name} (import {import_name}) - OK")
            success_count += 1
        except ImportError as e:
            print(f"❌ {package_name} (import {import_name}) - FAILED: {e}")
    
    print(f"📊 Package imports: {success_count}/{len(packages_to_test)} successful")
    return success_count == len(packages_to_test)

def test_environment_validation():
    """Test that environment validation works"""
    print("🧪 Testing environment validation...")
    
    try:
        from pipeline.utils import validate_environment
        
        # Test configuration
        config = {
            'evaluation': {
                'contextual_keywords': {'enabled': True},
                'ragas_metrics': {'enabled': False}  # Disable RAGAS for basic test
            },
            'output': {
                'base_dir': './outputs'
            }
        }
        
        result = validate_environment(config)
        
        if result.get('errors'):
            print(f"❌ Environment validation failed:")
            for error in result['errors']:
                print(f"   - {error}")
            return False
        else:
            print("✅ Environment validation passed")
            if result.get('warnings'):
                print("⚠️  Warnings:")
                for warning in result['warnings']:
                    print(f"   - {warning}")
            return True
            
    except Exception as e:
        print(f"❌ Environment validation error: {e}")
        return False

def test_tiktoken_fallback():
    """Test that tiktoken fallback works"""
    print("🧪 Testing tiktoken fallback...")
    
    try:
        # Import our fallback
        sys.path.append(str(Path(__file__).parent / "scripts"))
        from tiktoken_fallback import patch_tiktoken_with_fallback
        
        # Apply patch
        patch_tiktoken_with_fallback()
        
        # Test basic functionality
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode("Hello world")
        
        print(f"✅ Tiktoken fallback working - 'Hello world' -> {len(tokens)} tokens")
        
        # Test core module access (for RAGAS)
        import tiktoken.core
        print("✅ Tiktoken core module accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Tiktoken fallback error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running fix verification tests...")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_package_imports),
        ("Environment Validation", test_environment_validation),
        ("Tiktoken Fallback", test_tiktoken_fallback),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"💥 {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Fixes are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
