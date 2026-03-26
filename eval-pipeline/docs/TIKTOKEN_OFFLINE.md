# Tiktoken Offline Integration Guide

This guide explains how the tiktoken offline solution is integrated with your `./deploy.sh` deployment process.

## 🎯 Overview

The tiktoken offline solution is now fully integrated into your Docker build process. When you run `./deploy.sh --full --dev-advanced`, the container will automatically:

1. **Download tiktoken encodings** during build (requires internet)
2. **Cache them locally** in `/app/.cache/tiktoken`
3. **Run completely offline** at runtime

## 🚀 How to Use

### Deploy with Tiktoken Offline Support

```bash
cd /mnt/d/workspace/domain-specific-llm-eval/eval-pipeline
./deploy.sh --full --dev-advanced
```

That's it! Your container now includes:
- ✅ Pre-downloaded tiktoken encodings
- ✅ Offline environment configuration
- ✅ Automatic fallback handling

### Verify Tiktoken Offline Operation

After deployment, test that tiktoken works offline:

```bash
# Test tiktoken in the container
docker exec -it rag-eval-pipeline python3 scripts/validate_tiktoken_offline.py
```

Expected output:
```
✅ Cache directory exists: /app/.cache/tiktoken
🎯 Tiktoken cache files: ['o200k_base.tiktoken', 'cl100k_base.tiktoken', ...]
✅ o200k_base: 13 tokens, decode successful
✅ Tiktoken offline validation: PASSED
```

## 🔧 What Changed

### 1. Dockerfile Updates
- Added **tiktoken-setup stage** for downloading encodings
- Set **environment variables** for offline operation
- Integrated **cache setup script**

### 2. Environment Variables
The container now automatically sets:
```bash
TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
TIKTOKEN_CACHE_ONLY=1
TIKTOKEN_DISABLE_DOWNLOAD=1
```

### 3. Build Process
During `docker build`:
- Downloads all common tiktoken encodings
- Stores them in `/app/.cache/tiktoken`
- Validates the cache works

### 4. Runtime Behavior
At runtime:
- Uses cached tiktoken files only
- No internet requests for tiktoken
- Graceful fallback if cache issues

## 📊 Supported Encodings

The solution pre-downloads these tiktoken encodings:
- **o200k_base** - GPT-4o, GPT-4-turbo
- **cl100k_base** - GPT-4, GPT-3.5-turbo
- **p50k_base** - GPT-3 models
- **r50k_base** - GPT-3 davinci

## 🛠️ Troubleshooting

### Issue: Build fails during tiktoken download
**Solution**: Ensure internet access during build
```bash
# Check if you can reach tiktoken files
curl -I https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
```

### Issue: Runtime tiktoken errors
**Solution**: Validate the cache
```bash
docker exec -it rag-eval-pipeline ls -la /app/.cache/tiktoken/
docker exec -it rag-eval-pipeline python3 scripts/validate_tiktoken_offline.py
```

### Issue: Cache directory missing
**Solution**: Rebuild the container
```bash
./deploy.sh --full --dev-advanced
```

## 🎯 Benefits

✅ **Complete offline operation** - No runtime internet dependency
✅ **Automatic setup** - Works with existing deploy script
✅ **Robust caching** - Handles network failures gracefully
✅ **Production ready** - Optimized for enterprise environments
✅ **Easy validation** - Built-in testing tools

## 🔄 Migration from Quick Fix

If you were using the quick fix script:

1. **Remove temporary fixes**:
   ```bash
   docker exec -it rag-eval-pipeline bash -c "
   unset TIKTOKEN_CACHE_ONLY
   unset TIKTOKEN_DISABLE_DOWNLOAD
   "
   ```

2. **Rebuild with integrated solution**:
   ```bash
   ./deploy.sh --full --dev-advanced
   ```

3. **Verify operation**:
   ```bash
   docker exec -it rag-eval-pipeline python3 scripts/validate_tiktoken_offline.py
   ```

The integrated solution is more robust and permanent than the quick fix!
