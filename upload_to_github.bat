@echo off
chcp 65001 >nul
echo ========================================
echo 上传文件到GitHub rh分支
echo ========================================

REM 设置远程仓库
echo 正在设置远程仓库...
git remote add origin https://github.com/pzzmyc-ops/hhynodes.git

REM 添加所有文件
echo 正在添加文件...
git add .

REM 提交更改
echo 正在提交更改...
git commit -m "更新所有节点显示名称为中文"

REM 检查rh分支是否存在
echo 正在检查rh分支...
git fetch origin

REM 尝试切换到rh分支，如果不存在则创建
git checkout -b rh 2>nul || git checkout rh

REM 如果rh分支不存在，创建它
if %errorlevel% neq 0 (
    echo 创建rh分支...
    git checkout -b rh
)

REM 推送到远程rh分支
echo 正在推送到远程rh分支...
git push -u origin rh

if %errorlevel% equ 0 (
    echo ========================================
    echo 上传成功！
    echo 仓库地址: https://github.com/pzzmyc-ops/hhynodes.git
    echo 分支: rh
    echo ========================================
) else (
    echo ========================================
    echo 上传失败，请检查网络连接和权限
    echo ========================================
)

pause
