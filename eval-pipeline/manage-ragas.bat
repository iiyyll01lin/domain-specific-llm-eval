@echo off
REM =============================================================================
REM RAGAS Submodule Management Script (Windows)
REM =============================================================================

setlocal enabledelayedexpansion

REM Get script directory and project root
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\"

REM Function to print colored output (limited colors in cmd)
goto :skip_functions

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

:init_ragas_submodule
call :print_info "Initializing RAGAS submodule..."

cd /d "%PROJECT_ROOT%"

REM Check if .gitmodules exists
if not exist ".gitmodules" (
    call :print_error ".gitmodules file not found. Creating it..."
    (
        echo [submodule "ragas"]
        echo 	path = ragas
        echo 	url = https://github.com/explodinggradients/ragas.git
        echo 	branch = main
    ) > .gitmodules
    git add .gitmodules
)

REM Check if ragas directory exists and if it's a placeholder
if exist "ragas\README_PLACEHOLDER.md" (
    call :print_warning "Found placeholder RAGAS directory, removing it..."
    rmdir /s /q ragas
)

REM Try to initialize submodule
git submodule update --init --recursive ragas
if errorlevel 1 (
    call :print_error "Failed to initialize submodule. Trying alternative method..."
    
    REM Alternative: clone directly
    git clone https://github.com/explodinggradients/ragas.git ragas
    if errorlevel 1 (
        call :print_error "Failed to clone RAGAS repository. Check your network connection."
        exit /b 1
    ) else (
        call :print_success "Successfully cloned RAGAS repository"
        git add ragas .gitmodules
        call :print_info "RAGAS submodule initialized. Consider committing with: git commit -m 'Add RAGAS submodule'"
    )
) else (
    call :print_success "RAGAS submodule initialized successfully"
)

git submodule update ragas
if errorlevel 1 (
    call :print_error "Failed to update RAGAS submodule"
    exit /b 1
)

call :print_success "RAGAS submodule initialized"
goto :eof

:update_ragas_submodule
call :print_info "Updating RAGAS submodule..."

cd /d "%PROJECT_ROOT%"

if not exist "ragas" (
    call :print_error "RAGAS submodule not found. Run with --init first."
    exit /b 1
)

cd ragas
git fetch origin
if errorlevel 1 (
    call :print_error "Failed to fetch from origin"
    exit /b 1
)

git checkout main
if errorlevel 1 (
    call :print_error "Failed to checkout main branch"
    exit /b 1
)

git pull origin main
if errorlevel 1 (
    call :print_error "Failed to pull from origin"
    exit /b 1
)

cd /d "%PROJECT_ROOT%"
git add ragas

call :print_success "RAGAS submodule updated to latest version"
goto :eof

:check_ragas_status
call :print_info "Checking RAGAS submodule status..."

cd /d "%PROJECT_ROOT%"

if not exist "ragas" (
    call :print_warning "RAGAS submodule not found"
    exit /b 1
)

REM Check if ragas is a git repository
if not exist "ragas\.git" (
    call :print_warning "RAGAS directory exists but is not a git submodule"
    exit /b 1
)

cd ragas

REM Get current commit and branch
for /f %%i in ('git rev-parse HEAD') do set "current_commit=%%i"
for /f %%i in ('git rev-parse --abbrev-ref HEAD') do set "current_branch=%%i"

call :print_info "Current RAGAS commit: !current_commit!"
call :print_info "Current RAGAS branch: !current_branch!"

REM Check if there are updates available
git fetch origin >nul 2>&1
for /f %%i in ('git rev-parse origin/main') do set "latest_commit=%%i"

if not "!current_commit!"=="!latest_commit!" (
    call :print_warning "RAGAS submodule is behind latest version"
    call :print_info "Current: !current_commit!"
    call :print_info "Latest:  !latest_commit!"
    call :print_info "Run with --update to update"
) else (
    call :print_success "RAGAS submodule is up to date"
)

cd /d "%PROJECT_ROOT%"
goto :eof

:reset_ragas_submodule
call :print_info "Resetting RAGAS submodule..."

cd /d "%PROJECT_ROOT%"

if exist "ragas" (
    call :print_info "Removing existing RAGAS directory..."
    rmdir /s /q ragas
)

REM Remove from git if it exists
git rm -f ragas >nul 2>&1
git rm -f .gitmodules >nul 2>&1

REM Re-add the submodule
git submodule add https://github.com/explodinggradients/ragas.git ragas
if errorlevel 1 (
    call :print_error "Failed to add RAGAS submodule"
    exit /b 1
)

git submodule init ragas
git submodule update ragas

call :print_success "RAGAS submodule reset successfully"
goto :eof

:show_usage
echo RAGAS Submodule Management Script (Windows)
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --init      Initialize RAGAS submodule
echo   --update    Update RAGAS submodule to latest version
echo   --status    Check RAGAS submodule status
echo   --reset     Reset RAGAS submodule (clean reinstall)
echo   --help      Show this help message
echo.
echo Examples:
echo   %~nx0 --init      # First time setup
echo   %~nx0 --update    # Update to latest RAGAS
echo   %~nx0 --status    # Check current status
goto :eof

:skip_functions

REM Main execution
set "command=%~1"
if "%command%"=="" set "command=--status"

if "%command%"=="--init" (
    call :init_ragas_submodule
) else if "%command%"=="--update" (
    call :update_ragas_submodule
) else if "%command%"=="--status" (
    call :check_ragas_status
) else if "%command%"=="--reset" (
    call :reset_ragas_submodule
) else if "%command%"=="--help" (
    call :show_usage
) else (
    call :print_error "Unknown option: %command%"
    call :show_usage
    exit /b 1
)

REM Check for errors
if errorlevel 1 (
    call :print_error "Command failed with error code !errorlevel!"
    exit /b !errorlevel!
)

endlocal
