# Makefile for RAG Evaluation Pipeline Docker Operations
# Simplifies common Docker commands and operations

.PHONY: help build deploy start stop restart logs shell clean health status

# Configuration
IMAGE_NAME := rag-eval-pipeline
CONTAINER_NAME := rag-eval-pipeline
COMPOSE_FILE := docker-compose.yml
COMPOSE_DEV_FILE := docker-compose.dev.yml
COMPOSE_PROD_FILE := docker-compose.prod.yml

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "RAG Evaluation Pipeline - Docker Operations"
	@echo "==========================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make build     # Build Docker image"
	@echo "  make deploy    # Deploy with docker compose"
	@echo "  make logs      # Follow container logs"
	@echo "  make shell     # Access container shell"

build: ## Build Docker image
	@echo "🏗️ Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME):latest .
	@echo "✅ Build completed"

build-no-cache: ## Build Docker image without cache
	@echo "🏗️ Building Docker image without cache: $(IMAGE_NAME)"
	docker build --no-cache -t $(IMAGE_NAME):latest .
	@echo "✅ Build completed"

deploy: ## Deploy services in development mode (default)
	@echo "🚀 Deploying services in development mode..."
	docker compose -f $(COMPOSE_FILE) up -d
	@echo "✅ Development deployment completed"
	@make status

deploy-prod: ## Deploy in production mode
	@echo "🚀 Deploying in production mode..."
	docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PROD_FILE) up -d
	@echo "✅ Production deployment completed"
	@make status

deploy-dev: ## Deploy in advanced development mode
	@echo "🚀 Deploying in advanced development mode..."
	docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_DEV_FILE) up -d
	@echo "✅ Advanced development deployment completed"
	@make status

start: ## Start existing containers
	@echo "▶️ Starting containers..."
	docker compose -f $(COMPOSE_FILE) start
	@make status

stop: ## Stop running containers
	@echo "⏹️ Stopping containers..."
	docker compose -f $(COMPOSE_FILE) stop
	@echo "✅ Containers stopped"

restart: ## Restart containers
	@echo "🔄 Restarting containers..."
	docker compose -f $(COMPOSE_FILE) restart
	@make status

down: ## Stop and remove containers
	@echo "⬇️ Stopping and removing containers..."
	docker compose -f $(COMPOSE_FILE) down
	@echo "✅ Containers removed"

down-volumes: ## Stop and remove containers with volumes
	@echo "⬇️ Stopping and removing containers with volumes..."
	docker compose -f $(COMPOSE_FILE) down -v
	@echo "✅ Containers and volumes removed"

logs: ## Follow container logs
	@echo "📋 Following logs for $(CONTAINER_NAME)..."
	docker logs -f $(CONTAINER_NAME)

logs-all: ## Show logs for all services
	@echo "📋 Following logs for all services..."
	docker compose -f $(COMPOSE_FILE) logs -f

shell: ## Access container shell
	@echo "🐚 Accessing container shell..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

status: ## Show container status
	@echo "📊 Container Status:"
	@echo "==================="
	@docker ps -f name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"
	@echo ""
	@echo "📈 Resource Usage:"
	@echo "=================="
	@docker stats $(CONTAINER_NAME) --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

health: ## Run health check
	@echo "🔍 Running health check..."
	docker exec $(CONTAINER_NAME) /app/healthcheck.sh

inspect: ## Inspect container details
	@echo "🔍 Container Details:"
	@echo "===================="
	@docker inspect $(CONTAINER_NAME) --format='{{json .State}}' | jq '.'
	@echo ""
	@echo "📊 Health Status:"
	@echo "================"
	@docker inspect --format='{{.State.Health.Status}}' $(CONTAINER_NAME) 2>/dev/null || echo "No health check configured"

clean: ## Remove stopped containers and unused images
	@echo "🧹 Cleaning up Docker resources..."
	docker container prune -f
	docker image prune -f
	@echo "✅ Cleanup completed"

clean-all: ## Remove all containers, images, and volumes
	@echo "🧹 Cleaning up all Docker resources..."
	docker compose -f $(COMPOSE_FILE) down -v
	docker container prune -f
	docker image prune -a -f
	docker volume prune -f
	@echo "✅ Full cleanup completed"

rebuild: ## Stop, rebuild, and deploy
	@echo "🔄 Rebuilding and redeploying..."
	@make stop
	@make build
	@make deploy
	@echo "✅ Rebuild and redeploy completed"

test: ## Run tests in container
	@echo "🧪 Running tests in container..."
	docker exec $(CONTAINER_NAME) python -m pytest tests/ -v

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	docker exec $(CONTAINER_NAME) python test_pipeline.py

validate: ## Validate pipeline configuration
	@echo "✅ Validating pipeline configuration..."
	docker exec $(CONTAINER_NAME) python run_pipeline.py --mode validate

dry-run: ## Run pipeline in dry-run mode
	@echo "🔍 Running pipeline dry-run..."
	docker exec $(CONTAINER_NAME) python run_pipeline.py --mode dry-run

backup: ## Backup container data
	@echo "💾 Backing up container data..."
	@mkdir -p backups
	docker run --rm -v rag-eval-pipeline_pipeline-cache:/data -v $(PWD)/backups:/backup alpine tar czf /backup/cache-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	docker run --rm -v rag-eval-pipeline_pipeline-logs:/data -v $(PWD)/backups:/backup alpine tar czf /backup/logs-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "✅ Backup completed in ./backups/"

restore: ## Restore container data from backup
	@echo "📥 Available backups:"
	@ls -la backups/ | grep -E "(cache|logs)-backup" || echo "No backups found"
	@echo ""
	@echo "To restore, use:"
	@echo "  docker run --rm -v rag-eval-pipeline_pipeline-cache:/data -v \$$(PWD)/backups:/backup alpine tar xzf /backup/your-backup-file.tar.gz -C /data"

update: ## Update base images and rebuild
	@echo "📡 Updating base images..."
	docker pull python:3.10-slim-bullseye
	@make build
	@make deploy
	@echo "✅ Update completed"

monitor: ## Start monitoring (logs + stats)
	@echo "📊 Starting monitoring..."
	@echo "Press Ctrl+C to stop monitoring"
	@(make logs &) && (while true; do clear; make status; sleep 30; done)

# Development helpers
dev-shell: ## Access development container shell
	@echo "🐚 Accessing development container shell..."
	docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_DEV_FILE) exec rag-eval-pipeline /bin/bash

dev-logs: ## Follow development container logs
	@echo "📋 Following development logs..."
	docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_DEV_FILE) logs -f

# System information
docker-info: ## Show Docker system information
	@echo "🐳 Docker System Information:"
	@echo "============================"
	@docker system df
	@echo ""
	@echo "🖼️ Images:"
	@echo "=========="
	@docker images | head -10
	@echo ""
	@echo "📦 Containers:"
	@echo "=============="
	@docker ps -a | head -10

# Help for individual commands
help-deploy: ## Detailed help for deployment
	@echo "Deployment Help:"
	@echo "==============="
	@echo ""
	@echo "Basic deployment:"
	@echo "  make deploy          # Deploy in production mode"
	@echo "  make deploy-dev      # Deploy in development mode"
	@echo ""
	@echo "Full workflow:"
	@echo "  make build           # Build the image"
	@echo "  make deploy          # Deploy containers"
	@echo "  make status          # Check status"
	@echo "  make logs            # View logs"
	@echo ""
	@echo "Maintenance:"
	@echo "  make restart         # Restart containers"
	@echo "  make down            # Stop and remove containers"
	@echo "  make clean           # Clean up unused resources"
