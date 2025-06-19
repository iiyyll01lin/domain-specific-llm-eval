#!/bin/bash

# =============================================================================
# Test Local RAGAS Build Implementation
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

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Test function
test_file_exists() {
    local file="$1"
    local description="$2"
    
    if [ -f "$file" ]; then
        print_success "$description exists: $file"
        return 0
    else
        print_error "$description missing: $file"
        return 1
    fi
}

test_directory_exists() {
    local dir="$1"
    local description="$2"
    
    if [ -d "$dir" ]; then
        print_success "$description exists: $dir"
        return 0
    else
        print_error "$description missing: $dir"
        return 1
    fi
}

test_script_executable() {
    local script="$1"
    local description="$2"
    
    if [ -x "$script" ]; then
        print_success "$description is executable: $script"
        return 0
    else
        print_warning "$description is not executable: $script"
        chmod +x "$script" 2>/dev/null || print_error "Could not make $script executable"
        return 1
    fi
}

# Main test function
main() {
    print_header "Testing Local RAGAS Build Implementation"
    
    local tests_passed=0
    local tests_total=0
    
    # Test 1: Check if RAGAS submodule directory exists
    ((tests_total++))
    if test_directory_exists "$PROJECT_ROOT/ragas" "RAGAS submodule directory"; then
        ((tests_passed++))
    fi
    
    # Test 2: Check if manage-ragas.sh exists
    ((tests_total++))
    if test_file_exists "$SCRIPT_DIR/manage-ragas.sh" "RAGAS management script (Unix)"; then
        ((tests_passed++))
    fi
    
    # Test 3: Check if manage-ragas.bat exists
    ((tests_total++))
    if test_file_exists "$SCRIPT_DIR/manage-ragas.bat" "RAGAS management script (Windows)"; then
        ((tests_passed++))
    fi
    
    # Test 4: Check if local-ragas-dev.sh exists
    ((tests_total++))
    if test_file_exists "$SCRIPT_DIR/local-ragas-dev.sh" "Local RAGAS development script"; then
        ((tests_passed++))
    fi
    
    # Test 5: Check if build-with-proxy.sh exists
    ((tests_total++))
    if test_file_exists "$SCRIPT_DIR/build-with-proxy.sh" "Proxy-aware build script (Unix)"; then
        ((tests_passed++))
    fi
    
    # Test 6: Check if build-with-proxy.bat exists
    ((tests_total++))
    if test_file_exists "$SCRIPT_DIR/build-with-proxy.bat" "Proxy-aware build script (Windows)"; then
        ((tests_passed++))
    fi
    
    # Test 7: Check if Dockerfile has ragas-local stage
    ((tests_total++))
    if [ -f "$SCRIPT_DIR/Dockerfile" ]; then
        if grep -q "FROM dependencies as ragas-local" "$SCRIPT_DIR/Dockerfile"; then
            print_success "Dockerfile contains ragas-local stage"
            ((tests_passed++))
        else
            print_error "Dockerfile missing ragas-local stage"
        fi
    else
        print_error "Dockerfile not found"
    fi
    
    # Test 8: Check if requirements.txt has RAGAS commented out
    ((tests_total++))
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        if grep -q "# ragas>=0.1.0" "$SCRIPT_DIR/requirements.txt"; then
            print_success "requirements.txt has RAGAS properly commented out"
            ((tests_passed++))
        else
            print_warning "requirements.txt may not have RAGAS properly commented out"
        fi
    else
        print_error "requirements.txt not found"
    fi
    
    # Test 9: Check if LOCAL_RAGAS_BUILD.md exists
    ((tests_total++))
    if test_file_exists "$SCRIPT_DIR/LOCAL_RAGAS_BUILD.md" "Local RAGAS build documentation"; then
        ((tests_passed++))
    fi
    
    # Test 10: Check if RAGAS has pyproject.toml
    ((tests_total++))
    if [ -d "$PROJECT_ROOT/ragas" ]; then
        if [ -f "$PROJECT_ROOT/ragas/ragas/pyproject.toml" ]; then
            print_success "RAGAS pyproject.toml found"
            ((tests_passed++))
        else
            print_warning "RAGAS pyproject.toml not found (may need submodule update)"
        fi
    else
        print_error "RAGAS submodule directory not found"
    fi
    
    # Test script executability (on Unix-like systems)
    if [ "$(uname -s)" != "MINGW"* ] && [ "$(uname -s)" != "CYGWIN"* ]; then
        ((tests_total++))
        if test_script_executable "$SCRIPT_DIR/manage-ragas.sh" "RAGAS management script"; then
            ((tests_passed++))
        fi
        
        ((tests_total++))
        if test_script_executable "$SCRIPT_DIR/local-ragas-dev.sh" "Local RAGAS development script"; then
            ((tests_passed++))
        fi
        
        ((tests_total++))
        if test_script_executable "$SCRIPT_DIR/build-with-proxy.sh" "Proxy-aware build script"; then
            ((tests_passed++))
        fi
    fi
    
    # Summary
    print_header "Test Results"
    
    if [ $tests_passed -eq $tests_total ]; then
        print_success "All tests passed! ($tests_passed/$tests_total)"
        print_info "Local RAGAS build implementation is complete and ready to use."
        echo ""
        print_info "Next steps:"
        echo "  1. Initialize RAGAS submodule: ./manage-ragas.sh --init"
        echo "  2. Build Docker image: ./build-with-proxy.sh"
        echo "  3. Test installation: ./local-ragas-dev.sh --test"
        echo "  4. See LOCAL_RAGAS_BUILD.md for detailed documentation"
    else
        print_warning "Some tests failed: $tests_passed/$tests_total passed"
        print_info "Please check the failed tests above and ensure all files are in place."
    fi
    
    # Additional information
    print_header "Additional Information"
    print_info "Project structure:"
    echo "  📁 $PROJECT_ROOT/"
    echo "  ├── 📁 ragas/                    # RAGAS submodule"
    echo "  └── 📁 eval-pipeline/"
    echo "      ├── 🐳 Dockerfile            # Multi-stage build with RAGAS"
    echo "      ├── 🔧 manage-ragas.sh       # Submodule management"
    echo "      ├── 🔧 manage-ragas.bat      # Submodule management (Windows)"
    echo "      ├── 🚀 local-ragas-dev.sh    # Development helper"
    echo "      ├── 🌐 build-with-proxy.sh   # Proxy-aware build"
    echo "      ├── 🌐 build-with-proxy.bat  # Proxy-aware build (Windows)"
    echo "      └── 📚 LOCAL_RAGAS_BUILD.md  # Documentation"
    
    return $(( tests_total - tests_passed ))
}

# Run main function
main "$@"
