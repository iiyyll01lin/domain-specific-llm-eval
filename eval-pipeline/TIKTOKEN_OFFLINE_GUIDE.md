# Tiktoken Offline Setup for On-Premises Deployment

This guide provides multiple approaches to set up tiktoken for offline operation in your RAG evaluation pipeline.

## 🎯 Option 1: Standalone Setup (Recommended for Testing)

Use this approach to set up tiktoken offline on any system:

```bash
# Run the complete offline setup
python3 scripts/setup_tiktoken_offline.py

# Create a portable cache for other systems
python3 scripts/setup_tiktoken_offline.py --portable

# Use custom cache directory
python3 scripts/setup_tiktoken_offline.py --cache-dir /custom/path
```

## 🐳 Option 2: Docker Build Integration

### Method A: Update your existing Dockerfile

Add this to your `Dockerfile` after the dependencies stage:

```dockerfile
# ============================================================================
# TIKTOKEN OFFLINE CACHE SETUP
# ============================================================================

# Copy the download script
COPY scripts/download_tiktoken_cache.sh /tmp/download_tiktoken_cache.sh
RUN chmod +x /tmp/download_tiktoken_cache.sh

# Pre-download tiktoken encodings for offline operation
RUN /tmp/download_tiktoken_cache.sh && rm /tmp/download_tiktoken_cache.sh

# Set environment variables for offline operation
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    TIKTOKEN_CACHE_ONLY=1 \
    TIKTOKEN_DISABLE_DOWNLOAD=1

# Ensure proper ownership
RUN chown -R pipeline:pipeline /app/.cache/ 2>/dev/null || true
```

### Method B: Replace the existing tiktoken download line

Find this line in your Dockerfile:
```dockerfile
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')..." || echo "⚠️ Tiktoken pre-download failed..."
```

Replace it with:
```dockerfile
# Enhanced tiktoken offline setup
COPY scripts/download_tiktoken_cache.sh /tmp/download_tiktoken_cache.sh
RUN chmod +x /tmp/download_tiktoken_cache.sh && \
    /tmp/download_tiktoken_cache.sh && \
    rm /tmp/download_tiktoken_cache.sh

# Set offline environment variables
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    TIKTOKEN_CACHE_ONLY=1 \
    TIKTOKEN_DISABLE_DOWNLOAD=1
```

## 🚀 Option 3: Quick Fix for Running Container

For your currently running container:

```bash
# Run the quick fix script
./quick_tiktoken_fix.sh

# Or manually set environment variables
docker exec -it rag-eval-pipeline bash -c "
export TIKTOKEN_CACHE_ONLY=1
export TIKTOKEN_DISABLE_DOWNLOAD=1
echo 'Configured for offline operation'
"
```

## 📦 Option 4: Portable Cache Distribution

If you need to distribute tiktoken cache to multiple systems:

1. **Create portable cache** (on a system with internet):
   ```bash
   python3 scripts/setup_tiktoken_offline.py --portable --output ./tiktoken_distribution
   ```

2. **Distribute the cache**:
   ```bash
   # Copy the tiktoken_distribution directory to target systems
   scp -r tiktoken_distribution user@target-system:/tmp/
   ```

3. **Install on target systems**:
   ```bash
   # Linux/macOS
   cd /tmp/tiktoken_distribution
   chmod +x install_cache.sh
   ./install_cache.sh
   
   # Windows
   cd \tmp\tiktoken_distribution
   install_cache.bat
   ```

## 🔧 Verification

Test that tiktoken works offline:

```python
import os
os.environ["TIKTOKEN_CACHE_ONLY"] = "1"
os.environ["TIKTOKEN_DISABLE_DOWNLOAD"] = "1"

import tiktoken
encoding = tiktoken.get_encoding("o200k_base")
tokens = encoding.encode("Hello, world!")
print(f"✅ Offline tiktoken working: {len(tokens)} tokens")
```

## 📁 File Locations

- **Linux/macOS Cache**: `~/.cache/tiktoken/`
- **Windows Cache**: `%USERPROFILE%\AppData\Local\tiktoken\`
- **Docker Cache**: `/app/.cache/tiktoken/`

## 🌐 Environment Variables

Set these for offline operation:

```bash
export TIKTOKEN_CACHE_DIR=/path/to/cache
export TIKTOKEN_CACHE_ONLY=1
export TIKTOKEN_DISABLE_DOWNLOAD=1
```

## 🔍 Troubleshooting

### Issue: "Connection reset by peer"
**Solution**: The system is trying to download tiktoken data. Ensure:
1. Cache directory exists and contains `.tiktoken` files
2. Environment variables are set correctly
3. Cache files have proper permissions

### Issue: "tiktoken not available"
**Solution**: Install tiktoken or use the offline tokenizer fallback:
```python
from scripts.offline_tokenizer import patch_tiktoken_for_offline
patch_tiktoken_for_offline()
```

### Issue: Permission denied on cache files
**Solution**: Fix file permissions:
```bash
chmod 644 ~/.cache/tiktoken/*.tiktoken
chown $USER:$USER ~/.cache/tiktoken/*.tiktoken
```

## 🏗️ Building with Offline Cache

To rebuild your Docker image with offline tiktoken support:

```bash
# Copy the Dockerfile patch content into your Dockerfile
# Then rebuild
docker build -t rag-eval-pipeline:offline .

# Or use the deploy script with the updated Dockerfile
./deploy.sh --full --dev-advanced
```

## ✅ Success Indicators

You'll know it's working when you see:
- ✅ "Downloaded X/4 encodings successfully" during setup
- ✅ "Offline tiktoken validation successful!" in logs  
- ✅ Container starts without "Connection reset by peer" errors
- ✅ No network requests for tiktoken during runtime

## 📋 File Checklist

Ensure these files exist in your tiktoken cache:
- [ ] `o200k_base.tiktoken` (GPT-4o)
- [ ] `cl100k_base.tiktoken` (GPT-4, GPT-3.5-turbo)  
- [ ] `p50k_base.tiktoken` (GPT-3)
- [ ] `r50k_base.tiktoken` (GPT-3 davinci)

## 🎉 Next Steps

1. Choose your preferred option above
2. Test with your evaluation pipeline
3. Verify offline operation works
4. Deploy to your on-premises environment

Your RAG evaluation pipeline will now work completely offline! 🎯
