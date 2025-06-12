#!/usr/bin/env python3
"""
Installation helper script for Configurable Dataset Generator
Installs dependencies based on the mode you want to use
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package):
    """Check if a package is already installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def install_local_mode_dependencies():
    """Install dependencies for local mode"""
    print("📦 Installing LOCAL mode dependencies...")
    packages = [
        "sentence-transformers",
        "scikit-learn", 
        "keybert",
        "yake",
        "pandas",
        "numpy",
        "openpyxl",
        "matplotlib",       # For plotting in dynamic_ragas_gate_with_human_feedback.py
        "spacy",
        # Document processing dependencies
        "PyPDF2",           # PDF processing
        "pdfplumber",       # Better PDF text extraction
        "python-docx",      # Word document processing
        "langdetect"        # Language detection
    ]
    
    success = []
    failed = []
    
    for package in packages:
        print(f"  Installing {package}...")
        if install_package(package):
            success.append(package) 
            print(f"  ✅ {package}")
        else:
            failed.append(package)
            print(f"  ❌ {package}")    # Install spacy model
    print("  Installing spaCy English model...")
    try:
        # First try with spacy download and break-system-packages
        env = os.environ.copy()
        env['PIP_BREAK_SYSTEM_PACKAGES'] = '1'
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], env=env)
        print("  ✅ en_core_web_sm")
    except subprocess.CalledProcessError:
        # Fallback to direct pip install with wheel URL
        print("  ❌ Spacy download failed, trying direct pip install...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--break-system-packages",
                "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
            ])
            print("  ✅ en_core_web_sm (installed via pip)")
        except subprocess.CalledProcessError:
            print("  ❌ en_core_web_sm (install manually: python -m spacy download en_core_web_sm)")
            failed.append("en_core_web_sm")
    
    return success, failed

def install_ragas_mode_dependencies():
    """Install dependencies for RAGAS mode"""
    print("📦 Installing RAGAS mode dependencies...")
    
    packages = [
        "ragas",
        "datasets", 
        "transformers",
        "torch",
        "langchain-community",
        "keybert",
        "yake",
        "pandas",
        "numpy",
        "openpyxl",
        # Document processing dependencies
        "PyPDF2",           # PDF processing
        "pdfplumber",       # Better PDF text extraction
        "python-docx",      # Word document processing
        "langdetect"        # Language detection
    ]
    
    success = []
    failed = []
    
    for package in packages:
        print(f"  Installing {package}...")
        if install_package(package):
            success.append(package)
            print(f"  ✅ {package}")
        else:
            failed.append(package)
            print(f"  ❌ {package}")
    
    return success, failed

def install_basic_dependencies():
    """Install basic dependencies needed for all modes"""
    print("📦 Installing basic dependencies...")
    
    packages = [
        "pandas",
        "numpy", 
        "pyyaml",
        "openpyxl"
    ]
    
    success = []
    failed = []
    
    for package in packages:
        print(f"  Installing {package}...")
        if install_package(package):
            success.append(package)
            print(f"  ✅ {package}")
        else:
            failed.append(package)
            print(f"  ❌ {package}")

    return success, failed

def install_document_processing_dependencies():
    """Install dependencies for document processing (PDF, DOCX, etc.)"""
    print("📦 Installing document processing dependencies...")
    
    packages = [
        "PyPDF2",           # PDF processing
        "pdfplumber",       # Better PDF text extraction
        "python-docx",      # Word document processing
        "langdetect",       # Language detection
        "pandas",
        "numpy",
        "openpyxl"
    ]
    
    success = []
    failed = []
    
    for package in packages:
        print(f"  Installing {package}...")
        if install_package(package):
            success.append(package)
            print(f"  ✅ {package}")
        else:
            failed.append(package)
            print(f"  ❌ {package}")
    
    # Add missing dependencies for complete document processing
    missing_document_deps = [
        'langdetect',  # Language detection
        'keybert',     # Keyword/topic extraction  
        'yake'         # Alternative keyword extraction
    ]

    print("📦 Installing additional document processing dependencies...")
    for dep in missing_document_deps:
        install_package(dep)
    
    return success, failed

def main():
    """Main installation function"""
    print("🔧 Configurable Dataset Generator - Dependency Installer")
    print("=" * 60)
    print("Choose installation option:")
    print("1. LOCAL mode (recommended - no API keys needed)")
    print("2. RAGAS mode (requires more dependencies)")
    print("3. Basic dependencies only")
    print("4. All dependencies (kitchen sink)")
    print("5. Document processing only (PDF, DOCX, etc.)")
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 Installation cancelled")
        return
    
    if choice == "1":
        print("\n🏠 Installing LOCAL mode dependencies...")
        basic_success, basic_failed = install_basic_dependencies()
        local_success, local_failed = install_local_mode_dependencies()
        
        all_success = basic_success + local_success
        all_failed = basic_failed + local_failed
        
    elif choice == "2":
        print("\n🔬 Installing RAGAS mode dependencies...")
        basic_success, basic_failed = install_basic_dependencies()
        ragas_success, ragas_failed = install_ragas_mode_dependencies()
        
        all_success = basic_success + ragas_success
        all_failed = basic_failed + ragas_failed
        
    elif choice == "3":
        print("\n📦 Installing basic dependencies only...")
        all_success, all_failed = install_basic_dependencies()
        
    elif choice == "4":
        print("\n🚀 Installing ALL dependencies...")
        basic_success, basic_failed = install_basic_dependencies()
        local_success, local_failed = install_local_mode_dependencies()
        ragas_success, ragas_failed = install_ragas_mode_dependencies()
        all_success = basic_success + local_success + ragas_success
        all_failed = basic_failed + local_failed + ragas_failed
        
    elif choice == "5":
        print("\n📄 Installing document processing dependencies...")
        doc_success, doc_failed = install_document_processing_dependencies()
        
        all_success = doc_success
        all_failed = doc_failed
        
    else:
        print("❌ Invalid choice. Please run the script again.")
        return
    
    # Print summary
    print(f"\n📊 Installation Summary:")
    print("=" * 40)
    print(f"✅ Successfully installed: {len(all_success)} packages")
    print(f"❌ Failed to install: {len(all_failed)} packages")
    
    if all_failed:
        print(f"\n⚠️  Failed packages:")
        for package in all_failed:
            print(f"  - {package}")
        print(f"\nTry installing failed packages manually:")
        print(f"pip install {' '.join(all_failed)}")
    
    print(f"\n🎯 Next steps:")
    print("1. Edit config.yaml to set your preferred mode")
    print("2. Run: python generate_dataset_configurable.py")
    print("3. Run your evaluation pipeline")
    
    # Check config file
    if not os.path.exists("config.yaml"):
        print("\n⚠️  config.yaml not found. A default will be created when you run the generator.")

if __name__ == "__main__":
    main()