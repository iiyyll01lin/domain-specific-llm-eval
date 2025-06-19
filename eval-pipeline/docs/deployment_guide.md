# � **Docker Deployment Guide**
# RAG Evaluation Pipeline Containerization

## 🎯 **Overview**

This guide provides comprehensive instructions for deploying the RAG Evaluation Pipeline using Docker containers. The containerized deployment offers:

- **Consistent Environment**: Same runtime across development, testing, and production
- **Easy Deployment**: Single command deployment with Docker Compose
- **Resource Management**: Configurable CPU and memory limits
- **Volume Persistence**: Persistent storage for data, outputs, and configurations
- **Scalability**: Ready for orchestration with Kubernetes or Docker Swarm

## 🚀 **Quick Start**

### **Automated Deployment**

**Linux/macOS:**
```bash
# Make deployment script executable
chmod +x deploy.sh

# Full deployment (build + deploy)
./deploy.sh --full
```

**Windows:**
```cmd
# Run deployment script
deploy.bat --full
```

### **Manual Deployment**
```bash
# Build Docker image
docker build -t rag-eval-pipeline:latest .

# Deploy with Docker Compose
docker compose up -d
```

## 📋 **Prerequisites**

### **System Requirements**
- **Operating System**: Linux, macOS, or Windows 10/11
- **Memory**: Minimum 4GB RAM, Recommended 8GB+
- **Storage**: 10GB free space for images and data
- **Network**: Internet access for downloading dependencies

### **Software Requirements**
- **Docker**: Version 20.0+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ ([Install Docker Compose](https://docs.docker.com/compose/install/))

### **Verify Installation**
```bash
# Check Docker installation
docker --version
docker compose version

# Test Docker functionality
docker run hello-world
```

## 🏗️ **Docker Architecture**

### **Multi-Stage Build Process**

The Dockerfile uses a multi-stage build approach:

1. **Base Stage**: System dependencies and Python environment
2. **Dependencies Stage**: Python packages and NLP models
3. **Application Stage**: Application code and configuration
4. **Production Stage**: Optimized final image

### **Container Structure**

```
Container: rag-eval-pipeline
├── /app/                           # Application root
│   ├── src/                        # Source code
│   ├── config/                     # Configuration files
│   ├── data/ → Volume Mount        # Input documents
│   ├── outputs/ → Volume Mount     # Evaluation results
│   ├── cache/                      # Model cache
│   ├── logs/ → Volume Mount        # Application logs
│   └── temp/                       # Temporary files
├── User: pipeline (non-root)       # Security best practice
└── Working Directory: /app
```

## ⚙️ **Configuration**

### **Environment Variables**

Configure the container using environment variables:

```yaml
# docker-compose.yml
environment:
  - PYTHONUNBUFFERED=1              # Unbuffered Python output
  - PIPELINE_ENV=docker             # Runtime environment
  - LOG_LEVEL=INFO                  # Logging level
  - CACHE_DIR=/app/cache            # Model cache directory
  - TEMP_DIR=/app/temp              # Temporary files directory
```

### **Volume Mounts**

The deployment uses volume mounts for data persistence:

```yaml
volumes:
  # Input documents (read-only)
  - ./data:/app/data:ro
  
  # Output results (read-write)
  - ./outputs:/app/outputs:rw
  
  # Configuration files (read-only)  
  - ./config:/app/config:ro
  
  # Named volumes for cache and logs
  - pipeline-cache:/app/cache:rw
  - pipeline-logs:/app/logs:rw
```

## 📁 **Directory Setup**

### **Host Directory Structure**

Before deployment, ensure the following directory structure:

```
eval-pipeline/
├── data/
│   └── documents/                  # Place your documents here
│       ├── document1.pdf
│       └── document2.docx
├── config/
│   ├── pipeline_config.yaml       # Main configuration
│   └── secrets.yaml              # API keys (optional)
├── outputs/                       # Results will be generated here
└── docker-compose.yml
```

### **Prepare Directories**

```bash
# Create required directories
mkdir -p data/documents
mkdir -p outputs
mkdir -p config
mkdir -p logs

# Set appropriate permissions
chmod 755 outputs logs
```

## 🔄 **Deployment Commands**

### **Using Deployment Scripts**

**Full Deployment:**
```bash
./deploy.sh --full        # Linux/macOS
deploy.bat --full         # Windows
```

**Build Only:**
```bash
./deploy.sh --build       # Linux/macOS
deploy.bat --build        # Windows
```

**Deploy Only:**
```bash
./deploy.sh --deploy      # Linux/macOS
deploy.bat --deploy       # Windows
```

### **Using Docker Compose**

**Start Services:**
```bash
docker compose up -d                    # Start in background
docker compose up                       # Start with logs
```

**Manage Services:**
```bash
docker compose ps                       # Show running services
docker compose logs -f rag-eval-pipeline  # Follow logs
docker compose restart                  # Restart services
docker compose stop                     # Stop services
docker compose down                     # Stop and remove containers
docker compose down -v                  # Stop and remove volumes
```

## 📊 **Monitoring and Logs**

### **View Container Status**
```bash
# Container status
docker ps -f name=rag-eval-pipeline

# Container resource usage
docker stats rag-eval-pipeline

# Container details
docker inspect rag-eval-pipeline
```

### **Access Logs**
```bash
# Real-time logs
docker logs -f rag-eval-pipeline

# Last 100 lines
docker logs --tail 100 rag-eval-pipeline

# Export logs to file
docker logs rag-eval-pipeline > pipeline.log
```

### **Container Shell Access**
```bash
# Interactive shell
docker exec -it rag-eval-pipeline /bin/bash

# Run specific commands
docker exec rag-eval-pipeline python --version
docker exec rag-eval-pipeline ls -la /app/outputs
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Build Failures**

**Issue**: Docker build fails with package installation errors
```bash
# Solution: Clear Docker build cache
docker builder prune -f
docker build --no-cache -t rag-eval-pipeline:latest .
```

#### **Runtime Issues**

**Issue**: Container exits immediately
```bash
# Check container logs
docker logs rag-eval-pipeline

# Check exit code
docker ps -a -f name=rag-eval-pipeline
```

**Issue**: Permission denied errors
```bash
# Fix host directory permissions
sudo chown -R $USER:$USER outputs logs
chmod -R 755 outputs logs
```

#### **Network Issues**

**Issue**: Cannot connect to RAG system
```bash
# Use host.docker.internal (Docker Desktop)
# In config: api_endpoint: "http://host.docker.internal:8000"
```

### **Debug Mode**

Run container in debug mode:
```bash
# Interactive mode with shell
docker run -it --entrypoint /bin/bash rag-eval-pipeline

# Debug with verbose logging
docker run -e LOG_LEVEL=DEBUG rag-eval-pipeline
```

## 🔒 **Security Considerations**

### **Container Security**

- **Non-root user**: Container runs as `pipeline` user
- **Read-only mounts**: Input data mounted read-only
- **Resource limits**: CPU and memory constraints
- **Network isolation**: Custom network for service communication

### **Secrets Management**

```yaml
# config/secrets.yaml
openai:
  api_key: "your_openai_api_key_here"

custom_llm:
  api_key: "your_custom_llm_api_key"
  endpoint: "https://your-llm-endpoint.com"
```

## 📈 **Performance Optimization**

### **Resource Optimization**

```yaml
deploy:
  resources:
    limits:
      memory: 4G                    # Maximum memory usage
      cpus: '2.0'                   # Maximum CPU cores
    reservations:
      memory: 2G                    # Reserved memory
      cpus: '1.0'                   # Reserved CPU cores
```

### **Cache Optimization**

Use named volumes for persistent caching:
```yaml
volumes:
  pipeline-cache:/app/cache:rw    # Persistent model cache
```

## 🚀 **Production Deployment**

### **Kubernetes Deployment**

Convert to Kubernetes manifests:
```bash
# Convert docker-compose to Kubernetes
kompose convert -f docker-compose.yml
```

### **CI/CD Integration**

Example GitHub Actions workflow:
```yaml
name: Deploy RAG Evaluation Pipeline
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2      - name: Build and deploy
        run: |
          docker build -t rag-eval-pipeline:latest .
          docker compose up -d
```

## 📋 **Maintenance**

### **Regular Updates**

```bash
# Update base images
docker pull python:3.10-slim-bullseye

# Rebuild with latest base
docker build --pull -t rag-eval-pipeline:latest .
```

### **Cleanup**

```bash
# Remove unused containers and images
docker system prune -f

# Remove unused volumes
docker volume prune -f
```

### **Backup and Restore**

```bash
# Backup volumes
docker run --rm -v pipeline-cache:/data -v $(pwd):/backup alpine tar czf /backup/cache-backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v pipeline-cache:/data -v $(pwd):/backup alpine tar xzf /backup/cache-backup.tar.gz -C /data
```

Ready to deploy your RAG evaluation pipeline in a containerized environment! 🐳🚀

---

## 📋 **Pre-Deployment Checklist** (Traditional Deployment)

### **✅ Environment Requirements**
- [ ] Python 3.8+ installed
- [ ] Minimum 8GB RAM (16GB recommended for large documents)
- [ ] 10GB free disk space (for outputs and caching)
- [ ] Network access to your RAG system API
- [ ] PDF processing libraries installed (poppler, etc.)

### **✅ Dependencies Verification**
```bash
# Run setup and verification
python setup.py --quick-test

# Manual verification
python -c "import pandas, numpy, sentence_transformers, spacy; print('All dependencies OK')"
```

### **✅ Configuration Validation**
```bash
# Validate configuration
python run_pipeline.py --mode validate

# Test dry run
python run_pipeline.py --mode dry-run
```

## 🏗️ **Traditional Production Deployment Options**

### **Option 1: Local Server Deployment**
```bash
# Clone and setup
git clone <your-repo>
cd eval-pipeline
python setup.py

# Production configuration
cp config/pipeline_config.yaml config/production_config.yaml
# Edit production_config.yaml with your settings
```
