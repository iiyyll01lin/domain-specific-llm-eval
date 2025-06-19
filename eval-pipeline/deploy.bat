@echo off
REM =============================================================================
REM RAG Evaluation Pipeline - Docker Deployment Script (Windows)
REM =============================================================================

setlocal enabledelayedexpansion

set "PROJECT_NAME=rag-eval-pipeline"
set "IMAGE_NAME=rag-eval-pipeline"
set "CONTAINER_NAME=rag-eval-pipeline"
set "SCRIPT_DIR=%~dp0"

REM =============================================================================
REM Helper Functions
REM =============================================================================

:print_header
echo ============================================
echo   RAG Evaluation Pipeline - Docker Deploy  
echo ============================================
goto :eof

:print_step
echo [STEP] %~1
goto :eof

:print_info
echo [INFO] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

REM =============================================================================
REM Main Functions
REM =============================================================================

:check_docker
call :print_step "Checking Docker installation..."

docker --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not installed. Please install Docker Desktop first."
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        call :print_error "Docker Compose is not available. Please install Docker Desktop with Compose V2."
        exit /b 1
    ) else (
        call :print_info "Using Docker Compose V2"
    )
) else (
    call :print_info "Docker Compose V1 detected - will use 'docker compose' commands"
)

call :print_info "Docker and Docker Compose are installed."
goto :eof

:prepare_directories
call :print_step "Preparing required directories..."

if not exist "%SCRIPT_DIR%data\documents" mkdir "%SCRIPT_DIR%data\documents"
if not exist "%SCRIPT_DIR%outputs" mkdir "%SCRIPT_DIR%outputs"
if not exist "%SCRIPT_DIR%config" mkdir "%SCRIPT_DIR%config"
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

call :print_info "Directories prepared."
goto :eof

:build_image
call :print_step "Building Docker image..."

cd /d "%SCRIPT_DIR%"
docker build -t "%IMAGE_NAME%:latest" .

if errorlevel 1 (
    call :print_error "Image build failed!"
    exit /b 1
)

call :print_info "Image built successfully: %IMAGE_NAME%:latest"
goto :eof

:deploy_container
call :print_step "Deploying container..."

cd /d "%SCRIPT_DIR%"

REM Determine deployment mode
set "compose_files=-f docker-compose.yml"
set "mode_description=development (default)"

REM Check for mode flags
echo %* | findstr /i "production prod" >nul 2>&1
if not errorlevel 1 (
    set "compose_files=-f docker-compose.yml -f docker-compose.prod.yml"
    set "mode_description=production"
)

echo %* | findstr /i "dev-advanced dev-extra" >nul 2>&1
if not errorlevel 1 (
    set "compose_files=-f docker-compose.yml -f docker-compose.dev.yml"
    set "mode_description=advanced development"
)

call :print_info "Deploying in %mode_description% mode..."

REM Stop existing container if running
docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if not errorlevel 1 (
    call :print_info "Stopping existing container..."
    docker stop "%CONTAINER_NAME%"
)

REM Remove existing container if exists
docker ps -aq -f name=%CONTAINER_NAME% >nul 2>&1
if not errorlevel 1 (
    call :print_info "Removing existing container..."
    docker rm "%CONTAINER_NAME%"
)

REM Deploy using docker compose with appropriate files
docker compose %compose_files% up -d

if errorlevel 1 (
    call :print_error "Container deployment failed!"
    exit /b 1
)

call :print_info "Container deployed successfully in %mode_description% mode!"
call :print_info "Container name: %CONTAINER_NAME%"
goto :eof

:verify_deployment
call :print_step "Verifying deployment..."

REM Wait for container to start
timeout /t 10 /nobreak >nul

REM Check if container is running
docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if errorlevel 1 (
    call :print_error "Container is not running!"
    echo Container logs:
    docker logs "%CONTAINER_NAME%"
    exit /b 1
)

call :print_info "Container is running successfully!"

REM Show container status
docker ps -f name=%CONTAINER_NAME% --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

REM Show recent logs
echo.
echo Recent logs:
docker logs --tail 20 "%CONTAINER_NAME%"

goto :eof

:show_usage
call :print_step "Deployment completed successfully!"
echo.
echo Usage Commands:
echo   View logs:     docker logs -f %CONTAINER_NAME%
echo   Stop:          docker compose down
echo   Restart:       docker compose restart
echo   Shell access:  docker exec -it %CONTAINER_NAME% /bin/bash
echo   Remove:        docker compose down -v
echo.
echo Directories:
echo   Documents:     %SCRIPT_DIR%data\documents\
echo   Outputs:       %SCRIPT_DIR%outputs\
echo   Config:        %SCRIPT_DIR%config\
echo   Logs:          %SCRIPT_DIR%logs\
goto :eof

:stop_deployment
call :print_step "Stopping deployment..."
cd /d "%SCRIPT_DIR%"
docker compose down
call :print_info "Deployment stopped."
goto :eof

:show_logs
call :print_step "Showing container logs..."
docker logs -f "%CONTAINER_NAME%"
goto :eof

:clean_deployment
call :print_step "Cleaning up deployment..."
cd /d "%SCRIPT_DIR%"

REM Stop and remove containers
docker compose down -v

REM Remove images
docker rmi "%IMAGE_NAME%:latest" 2>nul

REM Remove unused volumes
docker volume prune -f

call :print_info "Cleanup completed."
goto :eof

:show_help
echo RAG Evaluation Pipeline - Docker Deployment Script
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   -h, --help       Show this help message
echo   -b, --build      Build Docker image only
echo   -d, --deploy     Deploy container only (assumes image exists)
echo   -f, --full       Full deployment (build + deploy)
echo   -s, --stop       Stop and remove containers
echo   -l, --logs       Show container logs
echo   -c, --clean      Clean up images and containers
echo.
echo Examples:
echo   %~nx0 --full        # Complete deployment
echo   %~nx0 --build       # Build image only
echo   %~nx0 --logs        # View logs
echo   %~nx0 --stop        # Stop deployment
goto :eof

REM =============================================================================
REM Main Execution
REM =============================================================================

:main
call :print_header

if "%~1"=="" goto :full_deploy
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
if "%~1"=="-b" goto :build_only
if "%~1"=="--build" goto :build_only
if "%~1"=="-d" goto :deploy_only
if "%~1"=="--deploy" goto :deploy_only
if "%~1"=="-f" goto :full_deploy
if "%~1"=="--full" goto :full_deploy
if "%~1"=="-s" goto :stop_only
if "%~1"=="--stop" goto :stop_only
if "%~1"=="-l" goto :logs_only
if "%~1"=="--logs" goto :logs_only
if "%~1"=="-c" goto :clean_only
if "%~1"=="--clean" goto :clean_only

:full_deploy
call :print_info "Running full deployment..."
call :check_docker
call :prepare_directories
call :build_image
call :deploy_container
call :verify_deployment
call :show_usage
goto :end

:build_only
call :check_docker
call :prepare_directories
call :build_image
goto :end

:deploy_only
call :check_docker
call :prepare_directories
call :deploy_container
call :verify_deployment
call :show_usage
goto :end

:stop_only
call :stop_deployment
goto :end

:logs_only
call :show_logs
goto :end

:clean_only
call :clean_deployment
goto :end

:end
endlocal
