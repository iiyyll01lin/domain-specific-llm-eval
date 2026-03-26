#!/usr/bin/env python3
"""
Script to fix missing packages in the running container
"""
import subprocess
import sys
import importlib

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}: {e}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported"""
    if import_name is None:
        import_name = package_name
        
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is available (imports as '{import_name}')")
        return True
    except ImportError:
        print(f"❌ {package_name} is not available (should import as '{import_name}')")
        return False

def main():
    print("🔍 Checking and fixing missing packages...")
    
    # Critical packages needed by the pipeline
    packages_to_check = [
        ("PyYAML", "yaml"),  # Package name, import name
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("openpyxl", "openpyxl"),
        ("sentence-transformers", "sentence_transformers"),
        ("spacy", "spacy"),
    ]
    
    missing_packages = []
    
    # Check each package
    for package_name, import_name in packages_to_check:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package_name in missing_packages:
            print(f"Installing {package_name}...")
            if install_package(package_name):
                print(f"✅ {package_name} installed successfully")
            else:
                print(f"❌ Failed to install {package_name}")
    else:
        print("\n✅ All required packages are available!")
    
    # Final verification
    print("\n🔍 Final verification...")
    all_good = True
    for package_name, import_name in packages_to_check:
        if not check_package(package_name, import_name):
            all_good = False
    
    if all_good:
        print("\n🎉 All packages are now available!")
        return 0
    else:
        print("\n❌ Some packages are still missing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
