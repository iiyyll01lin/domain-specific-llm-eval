#!/bin/bash

# =============================================================================
# RAGAS Submodule Management Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Function to initialize RAGAS submodule
init_ragas_submodule() {
    print_info "Initializing RAGAS submodule..."
    
    cd "$PROJECT_ROOT"
    
    # Check if .gitmodules exists
    if [ ! -f ".gitmodules" ]; then
        print_error ".gitmodules file not found. Creating it..."
        cat > .gitmodules << EOF
[submodule "ragas"]
	path = ragas
	url = https://github.com/explodinggradients/ragas.git
	branch = main
EOF
        git add .gitmodules
    fi
    
    # Check if ragas directory exists and if it's a placeholder
    if [ -d "ragas" ] && [ -f "ragas/README_PLACEHOLDER.md" ]; then
        print_warning "Found placeholder RAGAS directory, removing it..."
        rm -rf ragas
    fi
    
    # Try to initialize submodule
    if ! git submodule update --init --recursive ragas; then
        print_error "Failed to initialize submodule. Trying alternative method..."
        
        # Alternative: clone directly
        if git clone https://github.com/explodinggradients/ragas.git ragas; then
            print_success "Successfully cloned RAGAS repository"
            # Add it to git
            git add ragas .gitmodules
            print_info "RAGAS submodule initialized. Consider committing with: git commit -m 'Add RAGAS submodule'"
        else
            print_error "Failed to clone RAGAS repository. Check your network connection."
            return 1
        fi
    else
        print_success "RAGAS submodule initialized successfully"
    fi
    
    print_success "RAGAS submodule initialized"
}

# Function to update RAGAS submodule
update_ragas_submodule() {
    print_info "Updating RAGAS submodule..."
    
    cd "$PROJECT_ROOT"
    
    if [ ! -d "ragas" ]; then
        print_error "RAGAS submodule not found. Run with --init first."
        exit 1
    fi
    
    cd ragas
    git fetch origin
    git checkout main
    git pull origin main
    
    cd "$PROJECT_ROOT"
    git add ragas
    
    print_success "RAGAS submodule updated to latest version"
}

# Function to check RAGAS submodule status
check_ragas_status() {
    print_info "Checking RAGAS submodule status..."
    
    cd "$PROJECT_ROOT"
    
    if [ ! -d "ragas" ]; then
        print_warning "RAGAS submodule not found"
        return 1
    fi
    
    # Check if ragas is a git repository
    if [ ! -d "ragas/.git" ]; then
        print_warning "RAGAS directory exists but is not a git submodule"
        return 1
    fi
    
    cd ragas
    local current_commit=$(git rev-parse HEAD)
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    
    print_info "Current RAGAS commit: $current_commit"
    print_info "Current RAGAS branch: $current_branch"
    
    # Check if there are updates available
    git fetch origin
    local latest_commit=$(git rev-parse origin/main)
    
    if [ "$current_commit" != "$latest_commit" ]; then
        print_warning "RAGAS submodule is behind latest version"
        print_info "Current: $current_commit"
        print_info "Latest:  $latest_commit"
        print_info "Run with --update to update"
    else
        print_success "RAGAS submodule is up to date"
    fi
    
    cd "$PROJECT_ROOT"
}

# Function to clean and reset RAGAS submodule
reset_ragas_submodule() {
    print_info "Resetting RAGAS submodule..."
    
    cd "$PROJECT_ROOT"
    
    if [ -d "ragas" ]; then
        print_info "Removing existing RAGAS directory..."
        rm -rf ragas
    fi
    
    # Remove from git if it exists
    git rm -f ragas 2>/dev/null || true
    git rm -f .gitmodules 2>/dev/null || true
    
    # Re-add the submodule
    git submodule add https://github.com/explodinggradients/ragas.git ragas
    git submodule init ragas
    git submodule update ragas
    
    print_success "RAGAS submodule reset successfully"
}

# Function to show usage
show_usage() {
    echo "RAGAS Submodule Management Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --init      Initialize RAGAS submodule"
    echo "  --update    Update RAGAS submodule to latest version"
    echo "  --status    Check RAGAS submodule status"
    echo "  --reset     Reset RAGAS submodule (clean reinstall)"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --init      # First time setup"
    echo "  $0 --update    # Update to latest RAGAS"
    echo "  $0 --status    # Check current status"
}

# Main function
main() {
    case "${1:---status}" in
        --init)
            init_ragas_submodule
            ;;
        --update)
            update_ragas_submodule
            ;;
        --status)
            check_ragas_status
            ;;
        --reset)
            reset_ragas_submodule
            ;;
        --help)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
