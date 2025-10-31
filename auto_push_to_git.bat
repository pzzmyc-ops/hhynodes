@echo off
chcp 65001 >nul
echo ========================================
echo    自动推送到Git仓库
echo    (此脚本不会被上传到仓库)
echo ========================================
echo.

:: 切换到脚本所在目录（用于定位脚本相对路径）
cd /d "%~dp0"

:: 识别仓库根目录，并切换到仓库根目录执行后续操作
for /f "delims=" %%i in ('git rev-parse --show-toplevel 2^>nul') do set repo_root=%%i
if "%repo_root%"=="" (
    echo ❌ 错误: 未检测到Git仓库！
    echo 请在Git仓库内运行此脚本。
    pause
    exit /b 1
)
cd /d "%repo_root%"

echo 仓库根目录: %CD%
echo.

:: 检查并从Git仓库中删除此脚本文件（如果存在）
echo 🗑️  检查是否需要从仓库中移除此脚本...
set script_rel=custom_nodes/hhynodes/auto_push_to_git.bat
git ls-files --error-unmatch "%script_rel%" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo 🚫 从Git仓库中移除auto_push_to_git.bat文件...
    git rm --cached "%script_rel%"
    echo ✅ 文件已从仓库中移除，但本地保留
    echo.
)

echo 📋 检查Git状态...
git status --porcelain
if %ERRORLEVEL% neq 0 (
    echo ❌ Git状态检查失败！
    pause
    exit /b 1
)

echo 🔄 拉取远程最新代码...
git pull
if %ERRORLEVEL% neq 0 (
    echo ⚠️  拉取失败！请手动解决冲突后再运行此脚本。
    pause
    exit /b 1
)
echo ✅ 代码已同步到最新版本
echo.

echo 📁 添加所有更改的文件（仓库范围）...
echo 🚫 排除此脚本文件本身...
git add -A --ignore-errors
git reset HEAD "%script_rel%" 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ 文件添加失败！
    pause
    exit /b 1
)

:: 检查是否有文件需要提交
git diff --cached --quiet
if %ERRORLEVEL% equ 0 (
    echo ℹ️  没有文件需要提交，仓库已是最新状态。
    echo.
    echo 📊 当前状态:
    git status
    pause
    exit /b 0
)

echo.
echo 💾 准备提交更改...
echo.
echo 📝 请输入本次提交的说明信息:
echo    (例如: 修复了QwenVL检测的尺寸问题, 添加了新的图片处理功能等)
echo    (留空则使用默认的时间戳信息)
echo.
set /p user_commit_msg="提交信息: "

:: 如果用户没有输入，使用默认的时间戳信息
if "%user_commit_msg%"=="" (
    for /f "tokens=1-4 delims=/ " %%a in ('date /t') do (
        set mydate=%%a-%%b-%%c
    )
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do (
        set mytime=%%a:%%b
    )
    set commit_msg=Auto update: %mydate% %mytime%
    echo 使用默认提交信息: %commit_msg%
) else (
    set commit_msg=%user_commit_msg%
    echo 使用自定义提交信息: %commit_msg%
)

echo.
git commit -m "%commit_msg%"
if %ERRORLEVEL% neq 0 (
    echo ❌ 提交失败！
    pause
    exit /b 1
)

echo.
echo 🚀 推送到远程仓库...
:: 检查当前分支名
for /f "tokens=*" %%i in ('git branch --show-current 2^>nul') do set current_branch=%%i
if "%current_branch%"=="" (
    echo 设置默认分支为main...
    git checkout -b main
    set current_branch=main
)

echo 当前分支: %current_branch%
git push
set push_result=%ERRORLEVEL%
if %push_result% neq 0 (
    echo ⚠️  首次推送失败，尝试设置上游分支...
    echo 正在执行: git push --set-upstream origin %current_branch%
    git push --set-upstream origin %current_branch%
    set upstream_result=%ERRORLEVEL%
    if %upstream_result% neq 0 (
        echo ❌ 推送失败！
        echo 可能的原因：
        echo - 网络连接问题
        echo - 远程仓库权限问题
        echo - 需要先拉取远程更改
        echo.
        echo 请检查网络连接或先执行 git pull 同步远程更改。
        pause
        exit /b 1
    ) else (
        echo ✅ 成功设置上游分支并推送！
    )
)

echo.
echo ✅ 成功！文件已更新到Git仓库
echo 📊 提交信息: %commit_msg%
echo 🌐 仓库地址: 
git remote get-url origin
echo.
echo ========================================
echo    推送完成！
echo ========================================
pause
