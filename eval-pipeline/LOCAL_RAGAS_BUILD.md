# Local RAGAS Build Documentation

This document provides comprehensive guidance for building and using the RAG Evaluation Pipeline with a local RAGAS submodule for development and customization.

## 🎯 Overview

The local RAGAS build system allows you to:
- Use a local RAGAS repository as a git submodule
- Develop and customize RAGAS functionality
- Build Docker containers with your local RAGAS changes
- Support both development and production deployments
- Work behind corporate proxies

## 📁 Directory Structure

```
domain-specific-llm-eval/
├── ragas/                      # RAGAS git submodule
│   ├── ragas/                  # Nested RAGAS structure
│   │   ├── src/ragas/         # RAGAS source code
│   │   └── pyproject.toml     # RAGAS package config
│   └── ...
├── eval-pipeline/              # Main pipeline
│   ├── Dockerfile             # Multi-stage build with RAGAS
│   ├── manage-ragas.sh        # Submodule management (Linux/Mac)
│   ├── manage-ragas.bat       # Submodule management (Windows)
│   ├── local-ragas-dev.sh     # Development helper script
│   ├── build-with-proxy.sh    # Proxy-aware build (Linux/Mac)
│   ├── build-with-proxy.bat   # Proxy-aware build (Windows)
│   └── requirements.txt       # Pipeline dependencies (RAGAS commented out)
└── ...
```

## 🚀 Quick Start

### 1. Initialize RAGAS Submodule

**Linux/Mac:**
```bash
cd eval-pipeline
./manage-ragas.sh --init
```

**Windows:**
```cmd
cd eval-pipeline
manage-ragas.bat --init
```

### 2. Build Docker Image

**Linux/Mac:**
```bash
./build-with-proxy.sh
```

**Windows:**
```cmd
build-with-proxy.bat
```

### 3. Test Installation

**Linux/Mac:**
```bash
./local-ragas-dev.sh --test
```

## 🛠️ Detailed Setup

### Prerequisites

- Git
- Docker
- Python 3.9+ (for local development)
- Internet connection (for initial setup)

### RAGAS Submodule Management

The `manage-ragas` scripts provide comprehensive submodule management:

#### Available Commands

| Command | Description |
|---------|-------------|
| `--init` | Initialize RAGAS submodule (first time) |
| `--update` | Update to latest RAGAS version |
| `--status` | Check current submodule status |
| `--reset` | Clean reinstall of submodule |
| `--help` | Show usage information |

#### Examples

```bash
# First time setup
./manage-ragas.sh --init

# Update to latest RAGAS
./manage-ragas.sh --update

# Check current status
./manage-ragas.sh --status

# Clean reinstall
./manage-ragas.sh --reset
```

### Docker Build Process

The Docker build uses a multi-stage approach:

1. **Base Stage**: Python environment and system dependencies
2. **Dependencies Stage**: Install pipeline dependencies
3. **RAGAS-Local Stage**: Install RAGAS from local submodule
4. **Tiktoken Setup**: Offline token handling
5. **Application/Production Stage**: Final optimized image

#### Build Features

- **Automatic Detection**: Detects pyproject.toml vs setup.py
- **Fallback Support**: Falls back to PyPI if local build fails
- **Proxy Support**: Full proxy configuration
- **Robust Error Handling**: Continues build even if some components fail
- **Verification**: Tests RAGAS installation during build

### Local Development Environment

The `local-ragas-dev.sh` script provides a complete development workflow:

```bash
# Full setup and test
./local-ragas-dev.sh --setup --build --test

# Start development container
./local-ragas-dev.sh --run-dev

# Show workflow guide
./local-ragas-dev.sh --workflow
```

#### Development Features

- Creates isolated Python virtual environment
- Installs RAGAS in editable mode
- Mounts local RAGAS code into Docker containers
- Provides interactive development container
- Automated testing and verification

## 🔧 Configuration

### Docker Compose Integration

The local RAGAS build integrates with all Docker Compose modes:

#### Development Mode (default)
```yaml
# docker-compose.yml
volumes:
  - ../ragas:/app/ragas  # Mount local RAGAS for live development
```

#### Advanced Development Mode
```yaml
# docker-compose.dev.yml
volumes:
  - ../ragas:/app/ragas:ro  # Read-only mount for stability
```

#### Production Mode
```yaml
# docker-compose.prod.yml
# No RAGAS volume mount - uses built-in version
```

### Requirements Configuration

The `requirements.txt` file has RAGAS commented out:

```pip
# RAGAS dependencies
# ragas>=0.1.0              # RAG evaluation metrics - BUILT FROM LOCAL SUBMODULE
# NOTE: RAGAS is installed from local submodule in Dockerfile, not from PyPI
```

This prevents conflicts between PyPI and local installations.

## 🐛 Development Workflow

### 1. Local Python Development

```bash
# Setup local environment
./local-ragas-dev.sh --setup

# Activate virtual environment
source venv-ragas/bin/activate

# Make changes to RAGAS code
cd ../ragas/ragas/src/ragas/
# Edit Python files...

# Test changes locally
python -c "import ragas; print(ragas.__version__)"
```

### 2. Docker Development

```bash
# Build with local changes
./local-ragas-dev.sh --build

# Run development container
./local-ragas-dev.sh --run-dev

# Inside container, RAGAS changes are live-mounted
```

### 3. Testing and Validation

```bash
# Test RAGAS installation
./local-ragas-dev.sh --test

# Run full pipeline test
docker run --rm rag-eval-pipeline:latest python setup.py --quick-test
```

## 🌐 Proxy Support

### Automatic Proxy Detection

The build scripts automatically detect and use proxy settings:

```bash
# Environment variables
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Build automatically uses proxy
./build-with-proxy.sh
```

### Manual Proxy Configuration

Edit the build scripts to set specific proxy settings:

```bash
# In build-with-proxy.sh
PROXY_HOST="your-proxy-server"
PROXY_PORT="8080"
```

## 🐳 Docker Deployment Modes

### Development Mode
```bash
# Use docker-compose.yml (default)
docker-compose up -d
```
- Live code reloading
- Local RAGAS mounted
- Debug logging enabled
- Interactive shell access

### Advanced Development Mode
```bash
# Use docker-compose.dev.yml
docker-compose -f docker-compose.dev.yml up -d
```
- Higher resource limits
- Additional development tools
- Read-only RAGAS mount for stability

### Production Mode
```bash
# Use docker-compose.prod.yml
docker-compose -f docker-compose.prod.yml up -d
```
- Optimized performance
- Built-in RAGAS version
- Minimal resource usage
- Security-hardened

## 🔍 Troubleshooting

### Common Issues

#### 1. RAGAS Submodule Not Found
```bash
# Check submodule status
./manage-ragas.sh --status

# Reinitialize if needed
./manage-ragas.sh --reset
```

#### 2. Docker Build Fails
```bash
# Check proxy settings
echo $HTTP_PROXY

# Try direct build without proxy
docker build -t rag-eval-pipeline:latest .

# Check RAGAS directory structure
ls -la ../ragas/
```

#### 3. Import Errors in Container
```bash
# Test RAGAS installation
docker run --rm rag-eval-pipeline:latest python -c "import ragas; print(ragas.__file__)"

# Check Python path
docker run --rm rag-eval-pipeline:latest python -c "import sys; print(sys.path)"
```

#### 4. Proxy Authentication Issues
```bash
# Set proxy with authentication
export HTTP_PROXY=http://username:password@proxy.company.com:8080

# Or use .netrc file for credentials
```

### Debug Commands

```bash
# Check RAGAS version and location
docker run --rm rag-eval-pipeline:latest python -c "
import ragas
print(f'Version: {ragas.__version__}')
print(f'Location: {ragas.__file__}')
"

# List installed packages
docker run --rm rag-eval-pipeline:latest pip list | grep ragas

# Check submodule status
git submodule status

# Verify Docker build stages
docker build --target ragas-local -t test-ragas .
docker run --rm test-ragas python -c "import ragas; print('Success')"
```

## 📈 Performance Considerations

### Build Optimization

1. **Multi-stage builds** minimize final image size
2. **Dependency caching** speeds up rebuilds
3. **Proxy support** reduces network timeouts
4. **Fallback mechanisms** ensure builds succeed

### Development Efficiency

1. **Live mounting** enables real-time code changes
2. **Virtual environments** isolate dependencies
3. **Automated scripts** reduce manual steps
4. **Comprehensive testing** catches issues early

## 🔄 Update Workflow

### Updating RAGAS

```bash
# Update to latest RAGAS
./manage-ragas.sh --update

# Rebuild Docker image
./build-with-proxy.sh

# Test new version
./local-ragas-dev.sh --test
```

### Updating Pipeline

```bash
# Pull latest pipeline changes
git pull origin main

# Update submodules
git submodule update --remote

# Rebuild everything
./local-ragas-dev.sh --update --build --test
```

## 🎛️ Advanced Configuration

### Custom RAGAS Branch

```bash
cd ../ragas
git checkout feature-branch
cd ../eval-pipeline
./build-with-proxy.sh
```

### Multiple RAGAS Versions

```bash
# Create separate branches
git checkout -b ragas-v1
./manage-ragas.sh --init

git checkout -b ragas-v2
# Manually update ragas submodule to different version
```

### Environment-Specific Builds

```bash
# Development build
docker build --target application -t rag-eval-dev .

# Production build
docker build --target production -t rag-eval-prod .
```

## 📚 Additional Resources

- [RAGAS Documentation](https://github.com/explodinggradients/ragas)
- [Docker Multi-stage Builds](https://docs.docker.com/develop/dev-best-practices/dockerfile_best-practices/#use-multi-stage-builds)
- [Git Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

## 🤝 Contributing

When contributing to the local RAGAS build system:

1. Test all changes in both local and Docker environments
2. Update documentation for any new features
3. Ensure Windows and Linux/Mac compatibility
4. Test proxy and non-proxy scenarios
5. Verify all deployment modes work correctly

---

**Note**: This build system is designed to be robust and handle various corporate environments, proxy configurations, and development workflows. If you encounter issues not covered here, please check the troubleshooting section or create an issue with detailed error information.
