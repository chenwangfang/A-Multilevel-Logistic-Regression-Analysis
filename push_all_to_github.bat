@echo off
chcp 65001 >nul
echo ============================================================
echo GitHub仓库完整更新 - v2.0 (包含补充材料)
echo ============================================================
echo.

cd /d "G:\Project\实证\关联框架\github-repo-20250828_080438"

echo 仓库内容清单：
echo ✓ README.md (中英双语)
echo ✓ 8个Python分析脚本
echo ✓ 4个JSON统计结果
echo ✓ 5个高质量图表 (1200 DPI)
echo ✓ 补充材料 (Supplementary Materials)
echo ✓ XML-JSON混合架构文档
echo ✓ 编码方案文档
echo ✓ 中英文手稿
echo.

REM 检查是否已经初始化Git
if exist ".git" (
    echo [Git] 仓库已初始化，跳过init步骤
) else (
    echo [1/6] 初始化Git仓库...
    git init
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Git初始化失败
        pause
        exit /b 1
    )
    echo ✓ Git仓库初始化成功
)
echo.

echo [2/6] 检查Git状态...
git status --short
echo.

echo [3/6] 添加所有文件（包括新添加的补充材料）...
git add .
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 添加文件失败
    pause
    exit /b 1
)
echo ✓ 所有文件已添加到暂存区
echo.

echo [4/6] 提交更改...
git commit -m "v2.0: Complete framework with supplementary materials" -m "Main features:" -m "- Four hypothesis testing (H1-H4) with basic and advanced analyses" -m "- Statistical power 59.8%% with FDR correction" -m "- Python-R cross-validation system" -m "- 1200 DPI publication figures" -m "" -m "Supplementary materials:" -m "- Complete statistical methods documentation" -m "- XML-JSON hybrid architecture specification" -m "- SPAADIA corpus coding scheme" -m "- Extended results and robustness checks"
if %ERRORLEVEL% EQU 0 (
    echo ✓ 提交成功
) else (
    echo ⚠ 没有新的更改需要提交，或提交失败
)
echo.

echo [5/6] 配置远程仓库...
git remote get-url origin >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 添加远程仓库...
    git remote add origin https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis.git
) else (
    echo 远程仓库已配置
)
echo ✓ 远程仓库准备就绪
echo.

echo ============================================================
echo ⚠️  准备推送到GitHub
echo.
echo 将要推送的内容：
echo   - 核心分析脚本 (8个文件)
echo   - 统计结果 (JSON格式)
echo   - 发表级图表 (1200 DPI)
echo   - 补充材料文档 (3个文件)
echo   - 完整文档 (中英文手稿)
echo.
echo 注意：--force 将覆盖远程仓库所有内容！
echo ============================================================
echo.
echo 按任意键开始推送，或按 Ctrl+C 取消...
pause >nul

echo [6/6] 推送到GitHub...
git push -u origin main --force
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠️ 如果main分支不存在，尝试master分支...
    git branch -M main
    git push -u origin main --force
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ❌ 推送失败！
        echo.
        echo 可能的解决方案：
        echo.
        echo 1. 使用Personal Access Token:
        echo    - 访问: https://github.com/settings/tokens
        echo    - 生成新token，勾选'repo'权限
        echo    - 用户名：您的GitHub用户名
        echo    - 密码：粘贴token（不是GitHub密码）
        echo.
        echo 2. 检查网络连接
        echo.
        echo 3. 确认仓库权限
        echo.
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
echo 📊 v2.0 版本包含：
echo    ✓ 完整的统计分析框架
echo    ✓ 补充材料和技术文档
echo    ✓ Applied Linguistics期刊合规
echo    ✓ 完整的可重现性支持
echo.
echo 📎 补充材料包括：
echo    ✓ 详细统计方法说明
echo    ✓ XML-JSON混合架构文档
echo    ✓ SPAADIA语料库编码方案
echo    ✓ 扩展结果和稳健性检验
echo.
pause