# 🐳 Docker Development Mode - Quick Reference

The Docker deployment now **defaults to development mode** for easier development workflow!

## 🚀 **Quick Start - Development Mode (Default)**

**Linux/macOS:**
```bash
# Default: Development mode with live code reloading
./deploy.sh --full

# Same as above - dev is default
make deploy
```

**Windows:**
```cmd
# Default: Development mode
deploy.bat --full
```

## 🎯 **Available Modes**

### 1. **Development Mode (Default)** ✨
- **Features**: Live code reloading, debug logging, interactive shell
- **Target**: `application` stage in Dockerfile
- **Resources**: 6GB RAM, 3 CPUs
- **Ports**: 8080, 8888 exposed
- **Volumes**: Source code mounted as writable

```bash
# These are all equivalent (dev is default):
docker compose up -d
./deploy.sh --full
make deploy
```

### 2. **Production Mode** 🚀
- **Features**: Optimized, read-only, minimal resources
- **Target**: `production` stage in Dockerfile
- **Resources**: 4GB RAM, 2 CPUs
- **Security**: No exposed ports, read-only config

```bash
# Linux/macOS
./deploy.sh --full --production
make deploy-prod

# Windows
deploy.bat --full --production

# Manual
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 3. **Advanced Development Mode** 🛠️
- **Features**: All dev features + extra tools, higher resources
- **Resources**: 8GB RAM, 4 CPUs
- **Extras**: Auto-reload, extended volumes

```bash
# Linux/macOS
./deploy.sh --full --dev-advanced
make deploy-dev

# Windows
deploy.bat --full --dev-advanced

# Manual
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

## 📋 **Development Workflow**

### **1. Start Development (Default)**
```bash
# Just run - dev mode is default!
./deploy.sh --full

# Access the container
docker exec -it rag-eval-pipeline /bin/bash

# View logs
docker logs -f rag-eval-pipeline
```

### **2. Code Changes**
- Edit files on your host machine
- Changes are automatically reflected in container
- No rebuild needed for code changes!

### **3. Configuration Changes**
```bash
# Edit config files directly
nano config/pipeline_config.yaml

# Restart to apply config changes
docker compose restart
```

### **4. Quick Operations**
```bash
# Restart container
make restart

# View logs
make logs

# Access shell
make shell

# Stop everything
make stop
```

## 🔧 **File Structure Changes**

### **Main Compose File (`docker-compose.yml`)**
- **Now defaults to development mode**
- Includes live code mounting
- Debug logging enabled
- Interactive features enabled

### **Production Override (`docker-compose.prod.yml`)**
- Optimizes for production
- Removes development features
- Read-only volumes
- Minimal resource usage

### **Advanced Dev Override (`docker-compose.dev.yml`)**
- Enhanced development features
- Higher resource limits
- Additional development tools

## 🎯 **When to Use Each Mode**

| Mode | Use Case | Command |
|------|----------|---------|
| **Development** | Daily coding, testing, debugging | `./deploy.sh --full` |
| **Advanced Dev** | Heavy development, ML training | `./deploy.sh --full --dev-advanced` |
| **Production** | Deployment, CI/CD, performance testing | `./deploy.sh --full --production` |

## 🔥 **Benefits of Default Dev Mode**

1. **Instant Feedback**: Code changes reflect immediately
2. **Debug Ready**: Full logging and interactive access
3. **Port Access**: Web interfaces available at localhost:8080
4. **Jupyter Ready**: Port 8888 available for notebooks
5. **Config Flexibility**: Writable configuration files
6. **No Rebuilds**: No need to rebuild image for code changes

## 💡 **Pro Tips**

**Start with dev mode (default):**
```bash
# One command to get started
./deploy.sh --full
```

**Switch to production for testing:**
```bash
# Test production configuration
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**Use Makefile shortcuts:**
```bash
make deploy      # Development (default)
make deploy-prod # Production
make deploy-dev  # Advanced development
```

**Monitor everything:**
```bash
# Real-time logs and stats
make logs &
watch 'docker stats rag-eval-pipeline'
```

The development mode is now the default, making it much easier to get started with development! 🎉
