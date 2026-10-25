@echo off
setlocal EnableDelayedExpansion
rem Build AVBD 3D on Windows.
rem
rem Usage:  build.bat [release^|debug] [clean] [test] [run]
rem   release   optimised build in build\release (default)
rem   debug     debug build in build\debug
rem   clean     delete the build folder first
rem   test      run the test suite after building
rem   run       launch the demo after building

cd /d "%~dp0"

rem Started by double-click (no arguments): re-run in a window that stays open so errors can be read.
if not "%~1"=="" goto :no_relaunch
echo %cmdcmdline% | find /i "/c" >nul
if not errorlevel 1 if not defined AVBD3D_KEEP_OPEN (
    set "AVBD3D_KEEP_OPEN=1"
    cmd /k ""%~f0" %*"
    exit /b
)
:no_relaunch

set "CONFIG=Release"
set "DIR=build\release"
set "CLEAN="
set "TEST="
set "RUN="
:args
if "%~1"=="" goto :args_done
if /i "%~1"=="release" ( set "CONFIG=Release" & set "DIR=build\release" & goto :next )
if /i "%~1"=="debug"   ( set "CONFIG=Debug" & set "DIR=build\debug" & goto :next )
if /i "%~1"=="clean"   ( set "CLEAN=1" & goto :next )
if /i "%~1"=="test"    ( set "TEST=1" & goto :next )
if /i "%~1"=="run"     ( set "RUN=1" & goto :next )
echo [build] unknown argument: %~1
echo usage: build.bat [release^|debug] [clean] [test] [run]
exit /b 2
:next
shift
goto :args
:args_done

rem MSVC is only on PATH inside a vcvars64 environment.
if defined VCINSTALLDIR goto :have_msvc
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS="
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if not defined VCVARS (
    echo [build] ERROR: MSVC not found. Install "Desktop development with C++" in the Visual Studio Installer.
    exit /b 1
)
call "%VCVARS%" >nul
if errorlevel 1 (
    echo [build] ERROR: vcvars64.bat failed.
    exit /b 1
)

:have_msvc
for %%t in (cmake ninja) do (
    where %%t >nul 2>&1
    if errorlevel 1 (
        echo [build] ERROR: %%t is not on PATH.
        exit /b 1
    )
)
if not defined VULKAN_SDK (
    echo [build] ERROR: VULKAN_SDK is not set. Install the Vulkan SDK from https://vulkan.lunarg.com/
    exit /b 1
)
if not exist "%VULKAN_SDK%\Bin\slangc.exe" (
    echo [build] ERROR: slangc.exe not found in %VULKAN_SDK%\Bin. Reinstall the Vulkan SDK with Slang.
    exit /b 1
)
if not exist "external\SDL\CMakeLists.txt" (
    echo [build] Fetching submodules...
    git submodule update --init --recursive || exit /b 1
)

if defined CLEAN if exist "%DIR%" (
    echo [build] Removing %DIR%
    rmdir /s /q "%DIR%"
)

echo [build] Configuring %CONFIG% in %DIR%
cmake -B "%DIR%" -S . -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% || exit /b 1
echo [build] Building
cmake --build "%DIR%" || exit /b 1

if defined TEST (
    echo [build] Testing
    ctest --test-dir "%DIR%" --output-on-failure || exit /b 1
)
if defined RUN (
    echo [build] Launching the demo
    "%DIR%\avbd3d_demo.exe"
)
echo [build] Done: %CD%\%DIR%
endlocal
