@echo off
REM =============================================================================
REM Docker Build Script with Proxy Support (Windows)
REM =============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "IMAGE_NAME=rag-eval-pipeline"
set "DOCKERFILE=Dockerfile"

REM Proxy configuration
set "PROXY_HOST=10.6.254.210"
set "PROXY_PORT=3128"
set "HTTP_PROXY=http://!PROXY_HOST!:!PROXY_PORT!"
set "HTTPS_PROXY=http://!PROXY_HOST!:!PROXY_PORT!"
set "NO_PROXY=localhost,127.0.0.1"

REM =============================================================================
REM Helper Functions
REM =============================================================================

:print_info
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:build_with_proxy
call :print_info "Building Docker image with proxy configuration..."
call :print_info "Proxy: !HTTP_PROXY!"

REM Update RAGAS submodule before building
call :print_info "Managing RAGAS submodule..."
if exist "manage-ragas.bat" (
    call :print_info "Using manage-ragas.bat for submodule management..."
    
    REM First check status
    call manage-ragas.bat --status
    if errorlevel 1 (
        call :print_info "Initializing RAGAS submodule..."
        call manage-ragas.bat --init
        if errorlevel 1 call :print_warning "Could not initialize RAGAS submodule"
    ) else (
        call :print_info "Updating RAGAS submodule..."
        call manage-ragas.bat --update
        if errorlevel 1 call :print_warning "Could not update RAGAS submodule"
    )
) else (
    call :print_warning "manage-ragas.bat not found, using fallback method..."
    if exist "..\ragas" (
        cd ..\ragas
        git pull origin main
        if errorlevel 1 call :print_warning "Could not update RAGAS submodule"
        cd ..\eval-pipeline
    ) else (
        cd ..
        git submodule update --init --recursive
        if errorlevel 1 call :print_warning "Could not initialize RAGAS submodule"
        cd eval-pipeline
    )
)

docker build ^
    --build-arg HTTP_PROXY="!HTTP_PROXY!" ^
    --build-arg HTTPS_PROXY="!HTTPS_PROXY!" ^
    --build-arg NO_PROXY="!NO_PROXY!" ^
    --tag "!IMAGE_NAME!:latest" ^
    --file "!DOCKERFILE!" ^
    .

if errorlevel 1 (
    call :print_error "Docker build failed"
    exit /b 1
) else (
    call :print_success "Docker image built successfully: !IMAGE_NAME!:latest"
)
goto :eof

:build_without_proxy
call :print_info "Building Docker image without proxy..."

docker build ^
    --build-arg HTTP_PROXY="" ^
    --build-arg HTTPS_PROXY="" ^
    --build-arg NO_PROXY="" ^
    --tag "!IMAGE_NAME!:latest" ^
    --file "!DOCKERFILE!" ^
    .

if errorlevel 1 (
    call :print_error "Docker build failed"
    exit /b 1
) else (
    call :print_success "Docker image built successfully: !IMAGE_NAME!:latest"
)
goto :eof

:test_connectivity
call :print_info "Testing network connectivity..."

REM Test direct connection
curl -s --connect-timeout 5 http://deb.debian.org >nul 2>&1
if errorlevel 1 (
    call :print_warning "Direct connection failed, testing proxy..."
    
    REM Test proxy connection
    curl -s --connect-timeout 5 --proxy "!HTTP_PROXY!" http://deb.debian.org >nul 2>&1
    if errorlevel 1 (
        call :print_error "Both direct and proxy connections failed"
        exit /b 2
    ) else (
        call :print_success "Proxy connection: OK"
        exit /b 1
    )
) else (
    call :print_success "Direct connection: OK"
    exit /b 0
)

:show_usage
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --proxy         Build with proxy (default)
echo   --no-proxy      Build without proxy
echo   --auto          Auto-detect if proxy is needed
echo   --test          Test connectivity only
echo   --help          Show this help message
echo.
echo Environment variables:
echo   HTTP_PROXY      Override HTTP proxy URL
echo   HTTPS_PROXY     Override HTTPS proxy URL
echo   NO_PROXY        Override no-proxy list
goto :eof

REM =============================================================================
REM Main Function
REM =============================================================================

:main
set "OPTION=%~1"
if "%OPTION%"=="" set "OPTION=--proxy"

if "%OPTION%"=="--proxy" (
    call :build_with_proxy
) else if "%OPTION%"=="--no-proxy" (
    call :build_without_proxy
) else if "%OPTION%"=="--auto" (
    call :print_info "Auto-detecting network configuration..."
    call :test_connectivity
    if errorlevel 2 (
        call :print_error "Network connectivity issues detected"
        exit /b 1
    ) else if errorlevel 1 (
        call :build_with_proxy
    ) else (
        call :build_without_proxy
    )
) else if "%OPTION%"=="--test" (
    call :test_connectivity
    if errorlevel 2 (
        call :print_error "Check your network configuration"
        exit /b 1
    ) else if errorlevel 1 (
        call :print_info "Recommendation: Use --proxy"
    ) else (
        call :print_info "Recommendation: Use --no-proxy"
    )
) else if "%OPTION%"=="--help" (
    call :show_usage
) else (
    call :print_error "Unknown option: %OPTION%"
    call :show_usage
    exit /b 1
)

goto :eof

REM Run main function
call :main %*
