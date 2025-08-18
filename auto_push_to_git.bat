@echo off
chcp 65001 >nul
echo ========================================
echo    自动推送到Git仓库
echo ========================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

echo 当前工作目录: %CD%
echo.

:: 检查是否是git仓库
if not exist ".git" (
    echo ❌ 错误: 当前目录不是Git仓库！
    echo 请确保在Git仓库根目录下运行此脚本。
    pause
    exit /b 1
)

echo 📋 检查Git状态...
git status --porcelain
if %ERRORLEVEL% neq 0 (
    echo ❌ Git状态检查失败！
    pause
    exit /b 1
)

echo.
echo 📁 添加所有更改的文件...
git add .
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
git push
if %ERRORLEVEL% neq 0 (
    echo ❌ 推送失败！
    echo 可能的原因：
    echo - 网络连接问题
    echo - 远程仓库权限问题
    echo - 需要先拉取远程更改
    echo.
    echo 💡 尝试强制推送? (y/N)
    set /p force_push="输入 y 强制推送，其他任意键取消: "
    if /i "%force_push%"=="y" (
        echo 🔥 执行强制推送...
        git push --force-with-lease
        if %ERRORLEVEL% neq 0 (
            echo ❌ 强制推送也失败了！
            pause
            exit /b 1
        )
    ) else (
        echo 取消推送。
        pause
        exit /b 1
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
