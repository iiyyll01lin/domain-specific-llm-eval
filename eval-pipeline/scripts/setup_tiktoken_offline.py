#!/usr/bin/env python3
"""
Complete Offline Tiktoken Setup for On-Premises Deployment
This script downloads tiktoken encoding files and sets up offline operation
"""
import os
import sys
import logging
import requests
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Known tiktoken encoding files with their download URLs
TIKTOKEN_ENCODINGS = {
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", 
    "p50k_base": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    "r50k_base": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
}

def get_tiktoken_cache_dir() -> Path:
    """Get the tiktoken cache directory"""
    # Check environment override first
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = Path(os.environ["TIKTOKEN_CACHE_DIR"])
    else:
        # Use default locations based on platform
        if os.name == 'nt':  # Windows
            cache_dir = Path.home() / "AppData" / "Local" / "tiktoken"
        else:  # Linux/macOS
            cache_dir = Path.home() / ".cache" / "tiktoken"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using tiktoken cache directory: {cache_dir}")
    return cache_dir

def download_encoding_file(name: str, url: str, cache_dir: Path) -> bool:
    """Download a tiktoken encoding file with retry logic"""
    cache_file = cache_dir / f"{name}.tiktoken"
    
    # Skip if already exists
    if cache_file.exists():
        logger.info(f"✅ {name} already exists ({cache_file.stat().st_size} bytes)")
        return True
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"📥 Downloading {name} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            logger.info(f"✅ Downloaded {name} ({downloaded} bytes)")
            return True
            
        except Exception as e:
            logger.warning(f"❌ Attempt {attempt + 1} failed for {name}: {e}")
            if cache_file.exists():
                cache_file.unlink()  # Remove partial file
            
            if attempt < max_retries - 1:
                logger.info(f"   Retrying in 2 seconds...")
                import time
                time.sleep(2)
    
    logger.error(f"❌ Failed to download {name} after {max_retries} attempts")
    return False

def create_offline_cache() -> bool:
    """Create a complete offline tiktoken cache"""
    cache_dir = get_tiktoken_cache_dir()
    logger.info(f"🚀 Creating tiktoken offline cache in: {cache_dir}")
    
    success_count = 0
    total_count = len(TIKTOKEN_ENCODINGS)
    
    for name, url in TIKTOKEN_ENCODINGS.items():
        if download_encoding_file(name, url, cache_dir):
            success_count += 1
    
    logger.info(f"📊 Downloaded {success_count}/{total_count} encodings successfully")
    return success_count > 0

def setup_offline_environment():
    """Setup environment variables for offline operation"""
    cache_dir = get_tiktoken_cache_dir()
    
    # Environment variables for offline operation
    env_vars = {
        "TIKTOKEN_CACHE_DIR": str(cache_dir),
        "TIKTOKEN_CACHE_ONLY": "1",
        "TIKTOKEN_DISABLE_DOWNLOAD": "1"
    }
    
    logger.info("🔧 Setting up offline environment variables:")
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"   {key}={value}")

def validate_offline_setup() -> bool:
    """Validate that tiktoken works offline"""
    logger.info("🧪 Testing offline tiktoken functionality...")
    
    try:
        import tiktoken
        
        # Test each encoding
        success_count = 0
        for encoding_name in TIKTOKEN_ENCODINGS.keys():
            try:
                encoding = tiktoken.get_encoding(encoding_name)
                test_text = "Hello, world! This is a test for offline tokenization."
                tokens = encoding.encode(test_text)
                
                logger.info(f"   ✅ {encoding_name}: {len(tokens)} tokens")
                success_count += 1
                
            except Exception as e:
                logger.warning(f"   ⚠️ {encoding_name}: Failed - {e}")
        
        if success_count > 0:
            logger.info("✅ Offline tiktoken validation successful!")
            return True
        else:
            logger.error("❌ No encodings working offline")
            return False
        
    except ImportError:
        logger.error("❌ tiktoken not available for validation")
        return False
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

def create_portable_cache(output_dir: Optional[str] = None) -> bool:
    """Create a portable tiktoken cache that can be copied to other systems"""
    if output_dir is None:
        output_dir = "./tiktoken_cache_portable"
    
    output_path = Path(output_dir)
    cache_dir = get_tiktoken_cache_dir()
    
    logger.info(f"📦 Creating portable cache in: {output_path}")
    
    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all tiktoken files
        copied_count = 0
        for encoding_name in TIKTOKEN_ENCODINGS.keys():
            src_file = cache_dir / f"{encoding_name}.tiktoken"
            dst_file = output_path / f"{encoding_name}.tiktoken"
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                logger.info(f"   📄 Copied {encoding_name}")
                copied_count += 1
            else:
                logger.warning(f"   ⚠️ Missing {encoding_name}")
        
        # Create setup script for target systems
        setup_script = output_path / "install_cache.sh"
        setup_script.write_text(f"""#!/bin/bash
# Install tiktoken cache for offline operation

echo "🚀 Installing tiktoken offline cache..."

# Determine cache directory
if [ -n "$TIKTOKEN_CACHE_DIR" ]; then
    CACHE_DIR="$TIKTOKEN_CACHE_DIR"
elif [ -n "$HOME" ]; then
    CACHE_DIR="$HOME/.cache/tiktoken"
else
    CACHE_DIR="/tmp/tiktoken_cache"
fi

echo "📁 Cache directory: $CACHE_DIR"

# Create cache directory
mkdir -p "$CACHE_DIR"

# Copy encoding files
for file in *.tiktoken; do
    if [ -f "$file" ]; then
        cp "$file" "$CACHE_DIR/"
        echo "   📄 Installed $file"
    fi
done

# Set environment variables
echo "🔧 Setting environment variables..."
echo "export TIKTOKEN_CACHE_DIR=$CACHE_DIR" >> ~/.bashrc
echo "export TIKTOKEN_CACHE_ONLY=1" >> ~/.bashrc
echo "export TIKTOKEN_DISABLE_DOWNLOAD=1" >> ~/.bashrc

echo "✅ Tiktoken offline cache installed successfully!"
echo "Please run: source ~/.bashrc"
""")
        setup_script.chmod(0o755)
        
        # Create Windows batch script
        batch_script = output_path / "install_cache.bat"
        batch_script.write_text(f"""@echo off
REM Install tiktoken cache for offline operation on Windows

echo Installing tiktoken offline cache...

REM Determine cache directory
if defined TIKTOKEN_CACHE_DIR (
    set CACHE_DIR=%TIKTOKEN_CACHE_DIR%
) else (
    set CACHE_DIR=%USERPROFILE%\\AppData\\Local\\tiktoken
)

echo Cache directory: %CACHE_DIR%

REM Create cache directory
mkdir "%CACHE_DIR%" 2>nul

REM Copy encoding files
for %%f in (*.tiktoken) do (
    copy "%%f" "%CACHE_DIR%\\" >nul
    echo    Installed %%f
)

REM Set environment variables (user-level)
setx TIKTOKEN_CACHE_DIR "%CACHE_DIR%"
setx TIKTOKEN_CACHE_ONLY "1"
setx TIKTOKEN_DISABLE_DOWNLOAD "1"

echo Tiktoken offline cache installed successfully!
pause
""")
        
        # Create README
        import datetime
        readme = output_path / "README.md"
        readme_content = f"""# Tiktoken Offline Cache

This directory contains pre-downloaded tiktoken encoding files for offline operation.

## Files Included
{chr(10).join(f"- {name}.tiktoken" for name in TIKTOKEN_ENCODINGS.keys())}

## Installation

### Linux/macOS:
```bash
chmod +x install_cache.sh
./install_cache.sh
source ~/.bashrc
```

### Windows:
```cmd
install_cache.bat
```

### Manual Installation:
1. Copy all *.tiktoken files to your tiktoken cache directory:
   - Linux/macOS: `~/.cache/tiktoken/`
   - Windows: `%USERPROFILE%\\AppData\\Local\\tiktoken\\`

2. Set environment variables:
   ```bash
   export TIKTOKEN_CACHE_ONLY=1
   export TIKTOKEN_DISABLE_DOWNLOAD=1
   ```

## Usage
After installation, tiktoken will work offline without requiring internet access.

Generated on: {datetime.datetime.now().isoformat()}
Total files: {copied_count}/{len(TIKTOKEN_ENCODINGS)}
"""
        readme.write_text(readme_content)
        
        logger.info(f"✅ Portable cache created with {copied_count} files")
        logger.info(f"📦 Package ready: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create portable cache: {e}")
        return False

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup tiktoken for offline operation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_tiktoken_offline.py                    # Basic setup
  python setup_tiktoken_offline.py --portable         # Create portable cache
  python setup_tiktoken_offline.py --output /my/dir   # Custom output directory
        """
    )
    parser.add_argument("--portable", action="store_true", 
                       help="Create portable cache for other systems")
    parser.add_argument("--output", type=str, 
                       help="Output directory for portable cache")
    parser.add_argument("--cache-dir", type=str,
                       help="Custom tiktoken cache directory")
    
    args = parser.parse_args()
    
    # Set custom cache directory if provided
    if args.cache_dir:
        os.environ["TIKTOKEN_CACHE_DIR"] = args.cache_dir
    
    logger.info("🚀 Starting tiktoken offline setup...")
    
    try:
        # Step 1: Download encoding files
        cache_success = create_offline_cache()
        
        # Step 2: Setup environment
        setup_offline_environment()
        
        # Step 3: Validate setup
        validation_success = validate_offline_setup()
        
        # Step 4: Create portable cache if requested
        if args.portable:
            portable_success = create_portable_cache(args.output)
            if not portable_success:
                logger.warning("⚠️ Portable cache creation failed")
        
        # Final status
        if cache_success and validation_success:
            logger.info("✅ Tiktoken offline setup completed successfully!")
            logger.info("🎯 Your system is now ready for offline tiktoken operation!")
            return 0
        else:
            logger.warning("⚠️ Setup completed with issues")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏹️ Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
