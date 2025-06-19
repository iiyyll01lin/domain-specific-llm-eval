#!/usr/bin/env python3
"""
Simple Startup Script for RAG Evaluation Pipeline (No Docker)
This script handles setup and runs the pipeline with minimal configuration
"""

import sys
import os
import subprocess
from pathlib import Path

def print_header():
    print("🚀 RAG Evaluation Pipeline - Local Setup")
    print("=" * 50)

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    if version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    return True

def install_essential_packages():
    """Install essential packages with error handling"""
    print("\n📦 Installing essential packages...")
    
    # Essential packages in order of importance
    packages = [
        "pandas",
        "numpy", 
        "pyyaml",
        "openpyxl",
        "requests",
        "tqdm"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")
            
    # Advanced packages (optional)
    advanced_packages = [
        "scikit-learn",
        "sentence-transformers", 
        "keybert",
        "yake",
        "spacy"
    ]
    
    print("\n📦 Installing advanced packages (optional)...")
    for package in advanced_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True, timeout=60)
            print(f"✅ {package} installed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"⚠️ Skipped {package}: {e}")

def download_spacy_model():
    """Download spacy English model"""
    try:
        print("\n🔤 Downloading spaCy English model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                     check=True, capture_output=True)
        print("✅ spaCy model downloaded")
    except subprocess.CalledProcessError:
        print("⚠️ Could not download spaCy model (optional)")

def create_config():
    """Create a simple configuration file"""
    print("\n⚙️ Creating configuration...")
    
    config = {
        'pipeline': {
            'name': 'Simple RAG Evaluation',
            'mode': 'local'
        },
        'data_sources': {
            'documents': {
                'primary_docs': ['../../documents/DSP0266_1.22.0.pdf'],
                'file_types': ['pdf', 'txt', 'docx']
            }
        },
        'testset_generation': {
            'method': 'configurable',  # Local mode - no API keys needed
            'samples_per_document': 10,
            'max_total_samples': 20
        },
        'evaluation': {
            'enable_contextual': True,
            'enable_ragas': False,  # Disable RAGAS initially
            'thresholds': {
                'contextual_threshold': 0.6
            }
        },
        'output': {
            'base_dir': 'outputs',
            'formats': ['excel', 'json']
        }
    }
    
    import yaml
    config_path = Path('config/simple_config.yaml')
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"✅ Configuration created: {config_path}")
    return config_path

def test_imports():
    """Test that essential imports work"""
    print("\n🧪 Testing imports...")
    
    try:
        import pandas
        print("✅ pandas")
    except ImportError:
        print("❌ pandas - install with: pip install pandas")
        return False
        
    try:
        import numpy
        print("✅ numpy")
    except ImportError:
        print("❌ numpy - install with: pip install numpy")
        return False
        
    try:
        import yaml
        print("✅ yaml")
    except ImportError:
        print("❌ yaml - install with: pip install pyyaml")
        return False
        
    return True

def run_simple_testset_generator():
    """Run a simple testset generator"""
    print("\n🎯 Running simple testset generation...")
    
    try:
        # Try the configurable generator
        import generate_dataset_configurable
        print("✅ Running configurable dataset generator...")
        
        # This should work with minimal dependencies
        os.system(f"{sys.executable} generate_dataset_configurable.py")
        
    except Exception as e:
        print(f"⚠️ Configurable generator failed: {e}")
        
        # Try the simple generator
        try:
            print("🔄 Trying simple generator...")
            os.system(f"{sys.executable} generate_synthetic_dataset-simple.py")
        except Exception as e2:
            print(f"❌ Simple generator also failed: {e2}")

def main():
    print_header()
    
    # Check Python version
    if not check_python():
        return
    
    # Change to pipeline directory
    pipeline_dir = Path(__file__).parent
    os.chdir(pipeline_dir)
    print(f"📁 Working directory: {pipeline_dir}")
    
    # Install packages
    install_essential_packages()
    download_spacy_model()
    
    # Test imports
    if not test_imports():
        print("\n❌ Essential packages missing. Install manually:")
        print("pip install pandas numpy pyyaml openpyxl requests tqdm")
        return
    
    # Create configuration
    config_path = create_config()
    
    # Try to run pipeline
    print("\n🚀 Starting pipeline...")
    
    try:
        # Try the full pipeline first
        cmd = f"{sys.executable} run_pipeline.py --config {config_path}"
        print(f"Running: {cmd}")
        result = os.system(cmd)
        
        if result == 0:
            print("\n✅ Pipeline completed successfully!")
            print("📊 Check the 'outputs' directory for results")
        else:
            print("\n⚠️ Full pipeline had issues, trying simple generator...")
            run_simple_testset_generator()
            
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        print("🔄 Trying simple testset generation...")
        run_simple_testset_generator()
    
    print("\n✅ Setup complete!")
    print("\n📖 Next steps:")
    print("1. Check 'outputs' directory for generated files")
    print("2. Edit 'config/simple_config.yaml' for your documents")
    print("3. Run: python run_pipeline.py --config config/simple_config.yaml")

if __name__ == "__main__":
    main()
