# 🐳 Docker Deployment - Quick Reference

This directory contains all the necessary files to deploy the RAG Evaluation Pipeline using Docker containers.

## 📦 **What's Included**

```
📁 Docker Deployment Files:
├── 🐳 Dockerfile                    # Multi-stage Docker build
├── 🔧 docker-compose.yml           # Production deployment
├── 🛠️ docker-compose.dev.yml       # Development deployment
├── 🚀 deploy.sh                    # Linux/macOS deployment script
├── 🚀 deploy.bat                   # Windows deployment script
├── ❤️ healthcheck.sh               # Container health monitoring
├── 🔧 Makefile                     # Docker operation shortcuts
├── 🚫 .dockerignore                # Files excluded from build
└── 📖 docs/deployment_guide.md     # Complete deployment guide
```

## ⚡ **Quick Start**

### **Automated Deployment (Recommended)**

**Linux/macOS:**
```bash
chmod +x deploy.sh
./deploy.sh --full
```

**Windows:**
```cmd
deploy.bat --full
```

### **Manual Deployment**

```bash
# Build and deploy
docker build -t rag-eval-pipeline:latest .
docker compose up -d

# View status
docker ps
docker logs -f rag-eval-pipeline
```

### **Using Makefile (Linux/macOS)**

```bash
# Build and deploy
make deploy

# View logs
make logs

# Access container shell
make shell

# Stop deployment
make down
```

## 📁 **Setup Your Data**

1. **Place documents** in `data/documents/`:
   ```
   data/documents/
   ├── document1.pdf
   ├── document2.docx
   └── ...
   ```

2. **Configure** in `config/pipeline_config.yaml`:
   ```yaml
   data_sources:
     documents:
       primary_docs:
         - "/app/data/documents/document1.pdf"
   
   rag_system:
     api_endpoint: "http://host.docker.internal:8000/query"
   ```

3. **Run evaluation**:
   ```bash
   # Full pipeline
   docker exec rag-eval-pipeline python run_pipeline.py
   
   # Custom config
   docker exec rag-eval-pipeline python run_pipeline.py --config config/my_config.yaml
   ```

## 📊 **View Results**

Results are automatically saved to the `outputs/` directory:

```
outputs/
├── 📊 reports/executive_summary.html
├── 🔧 reports/technical_analysis.html
├── 📈 evaluations/detailed_results.xlsx
└── 📉 visualizations/*.png
```

## 🔄 **Common Commands**

```bash
# View logs
docker logs -f rag-eval-pipeline

# Access container shell
docker exec -it rag-eval-pipeline /bin/bash

# Stop and restart
docker compose restart

# Update and redeploy
docker compose pull
docker compose up -d

# Clean up
docker compose down -v
```

## 🐛 **Troubleshooting**

**Container won't start?**
```bash
# Check logs
docker logs rag-eval-pipeline

# Check configuration
docker exec rag-eval-pipeline python run_pipeline.py --mode validate
```

**Permission issues?**
```bash
# Fix directory permissions
chmod -R 755 outputs logs data
```

**Out of memory?**
```bash
# Increase Docker memory limit in Docker Desktop
# Or modify docker-compose.yml resource limits
```

## 🔗 **Next Steps**

1. ✅ **Deploy**: Use deployment scripts above
2. 📁 **Add Documents**: Place files in `data/documents/`
3. ⚙️ **Configure**: Edit `config/pipeline_config.yaml`
4. 🚀 **Run**: Execute pipeline in container
5. 📊 **Review**: Check results in `outputs/`

For detailed instructions, see: [`docs/deployment_guide.md`](docs/deployment_guide.md)

---

**Need help?** 
- Check container logs: `docker logs rag-eval-pipeline`
- Access shell: `docker exec -it rag-eval-pipeline /bin/bash`
- Run health check: `docker exec rag-eval-pipeline /app/healthcheck.sh`

Ready to containerize your RAG evaluation! 🚀🐳
