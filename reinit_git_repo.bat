@echo off
chcp 65001 >nul
echo ========================================
echo    重新初始化Git仓库 - 支持LFS大文件
echo    (此脚本不会被上传到仓库)
echo ========================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 识别仓库根目录
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

:: 检查Git LFS是否已安装
git lfs version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ 错误: 未检测到Git LFS！
    echo 请先安装Git LFS: https://git-lfs.github.com/
    pause
    exit /b 1
)

echo ✅ Git LFS已安装
echo.

:: 获取远程仓库URL
echo 🔍 获取远程仓库信息...
for /f "tokens=*" %%i in ('git remote get-url origin 2^>nul') do set remote_url=%%i
if "%remote_url%"=="" (
    echo ❌ 错误: 未找到远程仓库URL！
    echo 请确保已配置origin远程仓库。
    pause
    exit /b 1
)
echo 远程仓库: %remote_url%
echo.

:: 确认操作
echo ⚠️  警告: 此操作将完全删除现有Git历史并重新开始！
echo 请确保已备份重要数据。
echo.
set /p confirm="确定要继续吗？(y/N): "
if /i not "%confirm%"=="y" (
    echo 操作已取消。
    pause
    exit /b 0
)

echo.
echo 📁 创建备份...
set backup_dir=%~dp0..\hhynodes-backup-%date:~0,4%%date:~5,2%%date:~8,2%-%time:~0,2%%time:~3,2%%time:~6,2%
set backup_dir=%backup_dir: =0%
echo 备份目录: %backup_dir%
xcopy /E /I /H /Y "%CD%" "%backup_dir%" >nul
if %ERRORLEVEL% neq 0 (
    echo ⚠️  备份创建失败，但继续执行...
) else (
    echo ✅ 备份已创建: %backup_dir%
)
echo.

echo 🗑️  删除现有Git历史...
if exist ".git" (
    rmdir /S /Q ".git"
    echo ✅ .git文件夹已删除
) else (
    echo ℹ️  .git文件夹不存在
)
echo.

echo 🔧 重新初始化Git仓库...
git init
if %ERRORLEVEL% neq 0 (
    echo ❌ Git初始化失败！
    pause
    exit /b 1
)
echo ✅ Git仓库已初始化
echo.

echo 🔧 配置Git LFS...
git lfs install --local
if %ERRORLEVEL% neq 0 (
    echo ❌ LFS初始化失败！
    pause
    exit /b 1
)
echo ✅ Git LFS已配置
echo.

echo 🔧 设置大文件跟踪规则...
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "*.safetensors"
git lfs track "*.ckpt"
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.onnx"
git lfs track "*.tflite"
git lfs track "*.pb"
git lfs track "*.joblib"
git lfs track "*.npy"
git lfs track "*.npz"
git lfs track "*.tar"
git lfs track "*.tar.gz"
git lfs track "*.zip"
git lfs track "*.7z"
git lfs track "*.rar"
echo ✅ LFS跟踪规则已设置
echo.

echo 📝 创建.gitattributes文件...
echo # Git LFS tracking rules > .gitattributes
echo *.pt filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.pth filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.bin filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.safetensors filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.ckpt filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.pkl filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.h5 filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.onnx filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.tflite filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.pb filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.joblib filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.npy filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.npz filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.tar filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.tar.gz filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.zip filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.7z filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.rar filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo ✅ .gitattributes文件已创建
echo.

echo 📁 添加所有文件...
git add .
if %ERRORLEVEL% neq 0 (
    echo ❌ 文件添加失败！
    pause
    exit /b 1
)
echo ✅ 文件已添加到暂存区
echo.

echo 🚫 排除脚本文件...
git reset HEAD "custom_nodes/hhynodes/auto_push_to_git.bat" 2>nul
git reset HEAD "custom_nodes/hhynodes/fix_large_files.bat" 2>nul
git reset HEAD "custom_nodes/hhynodes/reinit_git_repo.bat" 2>nul
echo ✅ 脚本文件已排除
echo.

echo 💾 提交更改...
git commit -m "初始提交 - 支持LFS大文件"
if %ERRORLEVEL% neq 0 (
    echo ❌ 提交失败！
    pause
    exit /b 1
)
echo ✅ 提交成功
echo.

echo 🔗 添加远程仓库...
git remote add origin "%remote_url%"
if %ERRORLEVEL% neq 0 (
    echo ⚠️  远程仓库可能已存在，尝试更新...
    git remote set-url origin "%remote_url%"
)
echo ✅ 远程仓库已配置
echo.

echo 🔧 确保使用main分支...
git checkout -b main
echo ✅ 已切换到main分支
echo.

echo 🚀 推送到远程仓库...
echo 📤 推送LFS文件...
git lfs push --all origin
if %ERRORLEVEL% neq 0 (
    echo ⚠️  LFS推送失败，但继续尝试普通推送...
)

echo 📤 推送普通文件到main分支...
git push -f origin main
if %ERRORLEVEL% neq 0 (
    echo ❌ 推送失败！
    echo 可能的原因：
    echo - 网络连接问题
    echo - 远程仓库权限问题
    echo - 分支名称不匹配
    echo.
    echo 请检查网络连接和权限设置。
    pause
    exit /b 1
)
echo ✅ 推送成功
echo.

echo 🗑️  清理远程仓库的旧分支...
echo 尝试删除远程master分支（如果存在）...
git push origin --delete master 2>nul
if %ERRORLEVEL% equ 0 (
    echo ✅ 远程master分支已删除
) else (
    echo ℹ️  远程master分支不存在或已删除
)

echo 尝试删除远程main分支（如果存在旧版本）...
git push origin --delete main 2>nul
if %ERRORLEVEL% equ 0 (
    echo ✅ 旧main分支已删除，重新推送...
    git push -f origin main
) else (
    echo ℹ️  没有旧main分支需要删除
)
echo.

echo 🔍 验证LFS状态...
git lfs ls-files
echo.

echo 📊 检查仓库状态...
git status
echo.

echo ========================================
echo    ✅ 重新初始化完成！
echo ========================================
echo 📊 提交信息: 初始提交 - 支持LFS大文件
echo 🌐 仓库地址: %remote_url%
echo 💾 备份位置: %backup_dir%
echo.
echo 现在可以使用auto_push_to_git.bat正常推送了！
echo ========================================
pause
