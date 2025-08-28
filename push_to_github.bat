@echo off
chcp 65001 >nul
echo ============================================================
echo GitHub仓库更新脚本 - v2.0发布
echo ============================================================
echo.

cd /d "G:\Project\实证\关联框架\github-repo-20250828_080438"

echo [1/6] 初始化Git仓库...
git init
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Git初始化失败
    pause
    exit /b 1
)
echo ✓ Git仓库初始化成功
echo.

echo [2/6] 配置Git用户信息（如需要）...
REM 如果需要配置用户信息，取消下面两行注释并填入您的信息
REM git config user.name "Your Name"
REM git config user.email "your.email@example.com"
echo ✓ 跳过用户配置（使用全局配置）
echo.

echo [3/6] 添加所有文件...
git add .
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 添加文件失败
    pause
    exit /b 1
)
echo ✓ 所有文件已添加到暂存区
echo.

echo [4/6] 提交更改...
git commit -m "v2.0: Complete rewrite with enhanced statistical analysis and Applied Linguistics compliance" -m "- Four hypothesis testing modules (H1-H4) with both basic and advanced analyses" -m "- Statistical power analysis showing 59.8%% power for main effects" -m "- FDR-corrected p-values for multiple comparisons" -m "- Python-R cross-validation system" -m "- 1200 DPI publication-quality figures" -m "- Bilingual output (Chinese/English)"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 提交失败
    pause
    exit /b 1
)
echo ✓ 提交成功
echo.

echo [5/6] 添加远程仓库...
git remote add origin https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis.git
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ 远程仓库可能已存在，尝试更新...
    git remote set-url origin https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis.git
)
echo ✓ 远程仓库配置完成
echo.

echo [6/6] 推送到GitHub（将覆盖远程仓库）...
echo.
echo ⚠️  注意：这将强制覆盖GitHub上的所有内容！
echo     如果您想保留历史记录，请按 Ctrl+C 取消
echo     否则按任意键继续...
pause >nul

git push -u origin main --force
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠️  如果main分支不存在，尝试master分支...
    git push -u origin master --force
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ❌ 推送失败！可能的原因：
        echo    1. 需要GitHub认证（用户名/密码或Personal Access Token）
        echo    2. 网络连接问题
        echo    3. 仓库权限问题
        echo.
        echo 💡 建议：
        echo    - 使用GitHub Personal Access Token代替密码
        echo    - 在GitHub Settings - Developer settings创建token
        echo    - 确保token有repo权限
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo ✅ GitHub仓库更新成功！
echo ============================================================
echo.
echo 📌 仓库地址：
echo    https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis
echo.
echo 📊 主要更新内容：
echo    - 完整的v2.0版本实现
echo    - 四个假设的基础+高级分析
echo    - 59.8%%统计功效分析
echo    - FDR多重比较校正
echo    - 1200 DPI高质量图表
echo.
pause