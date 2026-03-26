#!/usr/bin/env python3
"""
Setup Script for Hybrid RAG Evaluation Pipeline
Installs dependencies and configures the environment
"""

import sys
import subprocess
import shutil
from pathlib import Path
import yaml
import os

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"🔄 {description or cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python dependencies...")
    
    # Check if pip is available
    if not shutil.which("pip"):
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print(f"❌ Requirements file not found: {req_file}")
        return False
    
    return run_command(
        f"pip install -r {req_file}",
        "Installing Python packages from requirements.txt"
    )

def install_spacy_models():
    """Install spaCy language models"""
    print("\n🔤 Installing spaCy language models...")
    
    models = ["en_core_web_sm", "en_core_web_md"]
    
    for model in models:
        success = run_command(
            f"python -m spacy download {model}",
            f"Installing spaCy model: {model}"
        )
        if not success:
            print(f"⚠️ Failed to install {model}, continuing...")

def setup_configuration():
    """Set up configuration files"""
    print("\n⚙️ Setting up configuration...")
    
    config_dir = Path(__file__).parent / "config"
    
    # Check if secrets.yaml exists, if not copy from template
    secrets_file = config_dir / "secrets.yaml"
    secrets_template = config_dir / "secrets.yaml.template"
    
    if not secrets_file.exists() and secrets_template.exists():
        print("📋 Creating secrets.yaml from template...")
        try:
            with open(secrets_template, 'r', encoding='utf-8') as f:
                content = f.read()
            
            with open(secrets_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Created {secrets_file}")
            print("⚠️  Please edit config/secrets.yaml and add your API keys")
        except Exception as e:
            print(f"❌ Failed to create secrets.yaml: {e}")
            return False
    elif secrets_file.exists():
        print("✅ secrets.yaml already exists")
    else:
        print("⚠️  secrets.yaml.template not found")
    
    return True

def test_installation():
    """Test if the installation is working"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import sentence_transformers
        import spacy
        print("✅ Core packages imported successfully")
        
        # Test spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp("This is a test sentence.")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("⚠️  spaCy model 'en_core_web_sm' not found")
        
        # Test sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer model loaded successfully")
        except Exception as e:
            print(f"⚠️  Sentence transformer test failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def create_sample_documents():
    """Create sample documents for testing"""
    print("\n📄 Creating sample documents...")
    
    sample_dir = Path(__file__).parent / "sample_documents"
    sample_dir.mkdir(exist_ok=True)
    
    # Create a sample text document
    sample_text = """
# Sample Document for RAG Evaluation

This is a sample document for testing the hybrid RAG evaluation pipeline.

## Introduction

The Redfish API provides a standards-based approach for accessing server management functions. It offers a RESTful interface that enables secure management of servers.

## Key Features

- RESTful API design
- JSON-based data representation  
- Secure authentication mechanisms
- Standardized resource modeling

## Authentication

Redfish supports multiple authentication methods:
1. Basic authentication
2. Session-based authentication
3. Token-based authentication

## Resource Collections

Common Redfish resource collections include:
- Systems: Physical and virtual computer systems
- Chassis: Physical enclosures and containers
- Managers: Management controllers and services
"""
    
    sample_file = sample_dir / "sample_redfish_doc.txt"
    try:
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print(f"✅ Created sample document: {sample_file}")
    except Exception as e:
        print(f"❌ Failed to create sample document: {e}")

def main():
    """Main setup function"""
    print("🚀 Hybrid RAG Evaluation Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check the errors above.")
        return False
    
    # Install spaCy models
    install_spacy_models()
    
    # Set up configuration
    if not setup_configuration():
        print("❌ Configuration setup failed")
        return False
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return False
    
    # Create sample documents
    create_sample_documents()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit config/secrets.yaml with your API keys")
    print("2. Update config/pipeline_config.yaml as needed")
    print("3. Run: python -m src.pipeline.orchestrator")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
