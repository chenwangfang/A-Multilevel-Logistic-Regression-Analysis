@echo off
chcp 65001 >nul
echo ============================================================
echo 验证GitHub仓库更新状态
echo ============================================================
echo.

cd /d "G:\Project\实证\关联框架\github-repo-20250828_080438"

echo 检查本地Git状态...
git status --short
echo.

echo 检查远程仓库连接...
git remote -v
echo.

echo 检查最新提交...
git log --oneline -5
echo.

echo 获取远程仓库信息...
git ls-remote --heads origin
echo.

echo ============================================================
echo.
echo 📌 请访问以下地址验证更新：
echo    https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis
echo.
echo 应该能看到：
echo    ✓ 更新的README.md（包含v2.0内容）
echo    ✓ scripts文件夹（8个Python脚本）
echo    ✓ output文件夹（JSON和图表）
echo    ✓ documentation文件夹（5个文档）
echo    ✓ requirements.txt
echo.
pause