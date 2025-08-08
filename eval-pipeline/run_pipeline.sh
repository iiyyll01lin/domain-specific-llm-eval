#!/bin/bash
# Pipeline Execution Script with Centralized Configuration
# This script provides easy access to the pipeline with different options

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/pipeline_config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Function to check if config file exists
check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    print_status "Configuration file found: $CONFIG_FILE"
}

# Function to show help
show_help() {
    echo "🎯 Pipeline Execution Script"
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  help                    Show this help message"
    echo "  verify                  Verify configuration and dependencies"
    echo "  guide                   Show quick start guide"
    echo "  test                    Run interactive test script"
    echo "  validate                Validate configuration (dry run)"
    echo "  generate                Generate testsets only"
    echo "  extract                 Extract keywords only"
    echo "  evaluate                Run evaluation only"
    echo "  report                  Generate reports only"
    echo "  full                    Run full pipeline"
    echo ""
    echo "Options:"
    echo "  --debug                 Enable debug logging"
    echo "  --force                 Force overwrite existing outputs"
    echo "  --output-dir DIR        Custom output directory"
    echo ""
    echo "Examples:"
    echo "  $0 verify               # Check configuration"
    echo "  $0 validate             # Test configuration without execution"
    echo "  $0 generate --debug     # Generate testsets with debug logging"
    echo "  $0 full --force         # Run full pipeline, overwrite existing"
}

# Function to run verification
run_verify() {
    print_info "Running configuration verification..."
    python3 "$SCRIPT_DIR/verify_config.py"
}

# Function to run guide
run_guide() {
    print_info "Showing quick start guide..."
    python3 "$SCRIPT_DIR/quick_start_guide.py"
}

# Function to run test
run_test() {
    print_info "Running interactive test..."
    python3 "$SCRIPT_DIR/test_centralized_config.py"
}

# Function to run pipeline
run_pipeline() {
    local stage="$1"
    local mode="$2"
    shift 2
    local extra_args="$@"
    
    local cmd="python3 $SCRIPT_DIR/run_pipeline.py --config $CONFIG_FILE"
    
    if [ ! -z "$stage" ]; then
        cmd="$cmd --stage $stage"
    fi
    
    if [ ! -z "$mode" ]; then
        cmd="$cmd --mode $mode"
    fi
    
    # Add extra arguments
    cmd="$cmd $extra_args"
    
    print_info "Running command: $cmd"
    echo "----------------------------------------"
    
    eval $cmd
    local exit_code=$?
    
    echo "----------------------------------------"
    if [ $exit_code -eq 0 ]; then
        print_status "Pipeline completed successfully!"
    else
        print_error "Pipeline failed with exit code: $exit_code"
    fi
    
    return $exit_code
}

# Parse arguments
parse_args() {
    local extra_args=""
    
    # Process options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --debug)
                extra_args="$extra_args --log-level DEBUG"
                shift
                ;;
            --force)
                extra_args="$extra_args --force"
                shift
                ;;
            --output-dir)
                extra_args="$extra_args --output-dir $2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    echo "$extra_args"
}

# Main execution
main() {
    echo "🚀 Pipeline Execution Script"
    echo "================================="
    
    # Check if no arguments provided
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # Check configuration file
    check_config
    
    # Parse command
    local command="$1"
    shift
    
    # Parse extra arguments
    local extra_args=$(parse_args "$@")
    
    case $command in
        help|--help|-h)
            show_help
            ;;
        verify)
            run_verify
            ;;
        guide)
            run_guide
            ;;
        test)
            run_test
            ;;
        validate)
            run_pipeline "all" "dry-run" $extra_args
            ;;
        generate)
            run_pipeline "testset-generation" "" $extra_args
            ;;
        extract)
            run_pipeline "keyword-extraction" "" $extra_args
            ;;
        evaluate)
            run_pipeline "evaluation" "" $extra_args
            ;;
        report)
            run_pipeline "reporting" "" $extra_args
            ;;
        full)
            run_pipeline "all" "" $extra_args
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"