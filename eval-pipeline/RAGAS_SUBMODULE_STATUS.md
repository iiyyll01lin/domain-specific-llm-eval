# RAGAS Submodule Status

## Current Situation

The RAGAS submodule setup encountered a network connectivity issue during initialization. This document explains the current state and how to proceed.

## What Has Been Done

✅ **Completed:**
- Created `.gitmodules` file with correct RAGAS submodule configuration
- Updated `manage-ragas.sh` and `manage-ragas.bat` scripts with better error handling
- Created placeholder RAGAS directory to prevent build errors
- Updated Dockerfile to support local RAGAS builds
- Updated Docker Compose files to mount local RAGAS when available
- Updated `requirements.txt` to use local RAGAS when available

⚠️ **Pending:**
- RAGAS submodule initialization (network connectivity required)

## How to Initialize RAGAS Submodule

When you have network connectivity, run one of these commands:

### Option 1: Using the manage-ragas script (Recommended)
```bash
cd eval-pipeline
./manage-ragas.sh --init
```

On Windows:
```cmd
cd eval-pipeline
manage-ragas.bat --init
```

### Option 2: Manual Git commands
```bash
# From project root
git submodule update --init --recursive ragas
```

### Option 3: If submodule commands fail
```bash
# Remove placeholder directory
rm -rf ragas

# Clone RAGAS manually
git clone https://github.com/explodinggradients/ragas.git ragas

# Add to git (the .gitmodules file already exists)
git add .gitmodules ragas
git commit -m "Add RAGAS submodule"
```

## Current Docker Build Behavior

### With Placeholder RAGAS Directory
- Docker builds will succeed
- RAGAS will be installed from PyPI (standard version)
- All functionality works, but uses the standard RAGAS library

### After RAGAS Submodule Initialization
- Docker builds will automatically detect the local RAGAS source
- Local RAGAS will be built and installed instead of PyPI version
- You can modify RAGAS locally for development
- Build scripts will automatically manage the submodule

## Verification

After initializing the RAGAS submodule, verify it's working:

### Check submodule status
```bash
git submodule status
# Should show: [commit-hash] ragas (branch-or-tag)
```

### Test Docker build with local RAGAS
```bash
cd eval-pipeline
./manage-ragas.sh --status
docker build -t rag-eval-pipeline .
```

### Test in development mode
```bash
cd eval-pipeline
docker-compose -f docker-compose.dev.yml up --build
```

## Benefits of Local RAGAS Build

Once the submodule is initialized:

1. **Development**: Modify RAGAS locally and test changes immediately
2. **Version Control**: Pin to specific RAGAS commits/branches
3. **Customization**: Apply local patches or customizations
4. **Consistency**: Same RAGAS version across all environments
5. **Offline Development**: Work without needing PyPI access

## Build Modes

The pipeline supports three build modes:

### Development Mode (default)
- Uses local RAGAS if available, falls back to PyPI
- Mounts local volumes for development
- Includes debugging tools
```bash
docker-compose -f docker-compose.dev.yml up
```

### Advanced Development Mode
- Same as dev mode but with additional debugging features
- Includes VS Code server for remote development
```bash
docker-compose -f docker-compose.dev.yml up -d --profile advanced-dev
```

### Production Mode
- Optimized for production deployment
- Always uses local RAGAS (fails if not available)
- Minimal image size
```bash
docker-compose -f docker-compose.prod.yml up
```

## Next Steps

1. **When network is available**: Run `./manage-ragas.sh --init` to initialize the submodule
2. **Test the build**: Verify the Docker build works with local RAGAS
3. **Develop**: Make any needed modifications to RAGAS locally
4. **Deploy**: Use the appropriate build mode for your environment

## Troubleshooting

### "No url found for submodule path 'ragas' in .gitmodules"
- This error occurred before the fix. The `.gitmodules` file is now properly created.
- Run the initialization commands above.

### Network connectivity issues
- Try using a VPN or different network
- Clone manually using Option 3 above
- Contact your network administrator about GitHub access

### Docker build fails
- If RAGAS submodule is not initialized, the build will use PyPI RAGAS
- Check that the submodule is properly initialized: `git submodule status`
- Ensure Docker can access the RAGAS directory: check `.dockerignore`
