@echo off
REM ============================================================================
REM  Kratos -> GitHub: commit the current code and push to main.
REM  Run this ON YOUR machine (it uses YOUR GitHub login). 'main' is already the
REM  only branch, so this just commits + pushes (no force, no branch surgery).
REM ============================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title Kratos - commit and push

echo [1/4] Sanity check: compiling all Python sources...
python -m compileall -q kratos setup_models.py kratos.py
if %errorlevel% neq 0 (
    echo   ERROR: code does not compile - NOT committing/pushing. Fix it first.
    pause
    exit /b
)
echo   OK.

echo [2/4] Committing current changes...
git add -A
git commit -m "Fix coder path-doubling: resolve emitted paths to the real project file (no stray nested copies); harden delete path resolution"
if %errorlevel% neq 0 echo   (nothing new to commit - continuing)

echo [3/4] Syncing with origin/main...
git pull --no-rebase origin main
if %errorlevel% neq 0 (
    echo   ERROR: pull failed (resolve conflicts, then re-run). Nothing pushed.
    pause
    exit /b
)

echo [4/4] Pushing to origin/main...
git push origin main
if %errorlevel% neq 0 (
    echo   ERROR: push failed (check your GitHub login / network).
    pause
    exit /b
)

echo.
echo ============================================================
echo   DONE. main is pushed.
echo ============================================================
git log --oneline -3
echo.
pause
endlocal
