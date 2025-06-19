#!/usr/bin/env python3
"""
Backup installation script for essential packages
Use this when the main requirements.txt installation fails
"""

import subprocess
import sys
import time

def install_package(package_name, retries=3):
    """Install a package with retries"""
    for attempt in range(retries):
        try:
            print(f"📦 Installing {package_name} (attempt {attempt + 1}/{retries})")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir", "--timeout", "60", 
                package_name
            ])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            if attempt < retries - 1:
                print(f"⏳ Retrying in 5 seconds...")
                time.sleep(5)
    return False

def main():
    print("🚨 Emergency Package Installation")
    print("=" * 40)
    
    # Essential packages in order of importance
    essential_packages = [
        "PyYAML",
        "pandas",
        "numpy",
        "openpyxl",
        "sentence-transformers",
        "spacy",
        "requests",
        "tqdm",
        "pathlib",
        "typing-extensions"
    ]
    
    success_count = 0
    failed_packages = []
    
    for package in essential_packages:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    print(f"\n📊 Installation Results:")
    print(f"✅ Successfully installed: {success_count}/{len(essential_packages)} packages")
    
    if failed_packages:
        print(f"❌ Failed to install: {', '.join(failed_packages)}")
        return 1
    else:
        print("🎉 All essential packages installed successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
