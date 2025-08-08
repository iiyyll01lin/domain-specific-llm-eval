#!/bin/bash

# =============================================================================
# RAG Evaluation Pipeline - Docker Deployment Script
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="rag-eval-pipeline"
IMAGE_NAME="rag-eval-pipeline"
CONTAINER_NAME="rag-eval-pipeline"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  RAG Evaluation Pipeline - Docker Deploy  ${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Deployment Functions
# =============================================================================

check_docker() {
    print_step "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please ensure Docker Compose V2 is installed."
        exit 1
    fi
    
    print_info "Docker and Docker Compose are installed."
}

prepare_directories() {
    print_step "Preparing required directories..."
    
    # Create directories if they don't exist
    mkdir -p "$SCRIPT_DIR/data/documents"
    mkdir -p "$SCRIPT_DIR/outputs"
    mkdir -p "$SCRIPT_DIR/config"
    mkdir -p "$SCRIPT_DIR/logs"
    
    # Set permissions
    chmod 755 "$SCRIPT_DIR/outputs"
    chmod 755 "$SCRIPT_DIR/logs"
    
    print_info "Directories prepared."
}

build_image() {
    print_step "Building Docker image..."
    
    # Change to project root for RAGAS access
    cd "$SCRIPT_DIR/.." || exit 1
    print_info "Building from project root: $(pwd)"
    
    # Verify RAGAS directory exists
    if [ -d "ragas" ]; then
        print_info "✅ RAGAS directory found"
    else
        print_error "❌ RAGAS directory not found in project root"
        exit 1
    fi
    
    # Update RAGAS submodule before building
    print_info "Updating RAGAS submodule..."
    if [ -f "eval-pipeline/manage-ragas.sh" ]; then
        chmod +x eval-pipeline/manage-ragas.sh
        eval-pipeline/manage-ragas.sh --update || print_warning "Could not update RAGAS submodule"
    fi
    
    # Check if proxy build script exists and use it
    if [ -f "eval-pipeline/build-with-proxy.sh" ]; then
        print_info "Using proxy-aware build script..."
        chmod +x eval-pipeline/build-with-proxy.sh
        eval-pipeline/build-with-proxy.sh --auto
    else
        # Fallback to direct docker build from project root
        print_warning "Proxy build script not found, using direct build..."
        docker build -t "$IMAGE_NAME:latest" -f eval-pipeline/Dockerfile .
    fi
    
    if [ $? -eq 0 ]; then
        print_info "Image built successfully: $IMAGE_NAME:latest"
    else
        print_error "Image build failed!"
        exit 1
    fi
}

deploy_container() {
    print_step "Deploying container..."
    cd "$SCRIPT_DIR"
    
    # Determine deployment mode
    local compose_files="-f docker-compose.yml"
    local mode_description="development (default)"
    
    # Check for mode flags
    if [[ "$*" == *"--production"* ]] || [[ "$*" == *"--prod"* ]]; then
        compose_files="-f docker-compose.yml -f docker-compose.prod.yml"
        mode_description="production"
    elif [[ "$*" == *"--dev-advanced"* ]] || [[ "$*" == *"--dev-extra"* ]]; then
        compose_files="-f docker-compose.yml -f docker-compose.dev.yml"
        mode_description="advanced development"
    fi
    
    print_info "Deploying in $mode_description mode..."
    
    # Stop existing container if running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        print_info "Stopping existing container..."
        docker stop "$CONTAINER_NAME"
    fi
    
    # Remove existing container if exists
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        print_info "Removing existing container..."
        docker rm "$CONTAINER_NAME"
    fi
    
    # Deploy using docker compose with appropriate files
    eval "docker compose $compose_files up -d"
    
    if [ $? -eq 0 ]; then
        print_info "Container deployed successfully in $mode_description mode!"
        print_info "Container name: $CONTAINER_NAME"
    else
        print_error "Container deployment failed!"
        exit 1
    fi
}

verify_deployment() {
    print_step "Verifying deployment..."
    
    # Wait for container to start
    sleep 10
    
    # Check if container is running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        print_info "Container is running successfully!"
        
        # Show container status
        docker ps -f name=$CONTAINER_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        
        # Show logs
        echo -e "\n${BLUE}Recent logs:${NC}"
        docker logs --tail 20 "$CONTAINER_NAME"
        
    else
        print_error "Container is not running!"
        echo -e "\n${RED}Container logs:${NC}"
        docker logs "$CONTAINER_NAME"
        exit 1
    fi
}

show_usage() {
    print_step "Deployment completed successfully!"
    echo -e "\n${GREEN}Usage Commands:${NC}"
    echo "  View logs:     docker logs -f $CONTAINER_NAME"
    echo "  Stop:          docker compose down"
    echo "  Restart:       docker compose restart"
    echo "  Shell access:  docker exec -it $CONTAINER_NAME /bin/bash"
    echo "  Remove:        docker compose down -v"
    echo ""
    echo -e "${GREEN}Directories:${NC}"
    echo "  Documents:     $SCRIPT_DIR/data/documents/"
    echo "  Outputs:       $SCRIPT_DIR/outputs/"
    echo "  Config:        $SCRIPT_DIR/config/"
    echo "  Logs:          $SCRIPT_DIR/logs/"
}

# =============================================================================
# Command Line Interface
# =============================================================================

show_help() {
    echo "RAG Evaluation Pipeline - Docker Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deployment Options:"
    echo "  --full                    Full deployment (build + deploy) - DEFAULT: Development mode"
    echo "  --build                   Build Docker image only"
    echo "  --deploy                  Deploy containers only"
    echo "  --stop                    Stop deployment"
    echo "  --clean                   Clean up containers and images"
    echo "  --logs                    Show container logs"
    echo ""
    echo "Mode Options:"
    echo "  (default)                 Development mode - Live code reloading, debug logging"
    echo "  --production, --prod      Production mode - Optimized, read-only, minimal resources"
    echo "  --dev-advanced, --dev-extra  Advanced development - Extra features, higher resources"
    echo ""
    echo "Examples:"
    echo "  $0 --full                 # Development mode (default)"
    echo "  $0 --full --production    # Production mode"
    echo "  $0 --full --dev-advanced  # Advanced development mode"
    echo "  $0 --build                # Build only"
    echo "  $0 --deploy --prod        # Deploy in production mode"
    echo ""
    echo "Environment Variables:"
    echo "  HTTP_PROXY               HTTP proxy URL"
    echo "  HTTPS_PROXY              HTTPS proxy URL"
    echo "  NO_PROXY                 No proxy list"
    echo "Options:"
    echo "  -h, --help       Show this help message"
    echo "  -b, --build      Build Docker image only"
    echo "  -d, --deploy     Deploy container only (assumes image exists)"
    echo "  -f, --full       Full deployment (build + deploy)"
    echo "  -s, --stop       Stop and remove containers"
    echo "  -l, --logs       Show container logs"
    echo "  -c, --clean      Clean up images and containers"
    echo ""
    echo "Examples:"
    echo "  $0 --full        # Complete deployment"
    echo "  $0 --build       # Build image only"
    echo "  $0 --logs        # View logs"
    echo "  $0 --stop        # Stop deployment"
}

stop_deployment() {
    print_step "Stopping deployment..."
    cd "$SCRIPT_DIR"
    docker compose down
    print_info "Deployment stopped."
}

show_logs() {
    print_step "Showing container logs..."
    docker logs -f "$CONTAINER_NAME"
}

clean_deployment() {
    print_step "Cleaning up deployment..."
    cd "$SCRIPT_DIR"
    
    # Stop and remove containers
    docker compose down -v
    
    # Remove images
    docker rmi "$IMAGE_NAME:latest" 2>/dev/null || true
    
    # Remove unused volumes
    docker volume prune -f
    
    print_info "Cleanup completed."
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    print_header
    
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -b|--build)
            check_docker
            prepare_directories
            build_image
            ;;
        -d|--deploy)
            check_docker
            prepare_directories
            deploy_container
            verify_deployment
            show_usage
            ;;
        -f|--full)
            check_docker
            prepare_directories
            build_image
            deploy_container
            verify_deployment
            show_usage
            ;;
        -s|--stop)
            stop_deployment
            ;;
        -l|--logs)
            show_logs
            ;;
        -c|--clean)
            clean_deployment
            ;;
        *)
            print_info "No option specified. Running full deployment..."
            check_docker
            prepare_directories
            build_image
            deploy_container
            verify_deployment
            show_usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"
