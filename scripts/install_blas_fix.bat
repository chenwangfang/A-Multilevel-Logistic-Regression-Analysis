@echo off
REM 修复BLAS库问题的批处理脚本
echo ======================================================================
echo 修复BLAS库问题
echo ======================================================================

REM 1. 更新numpy和scipy以获得正确的BLAS
echo.
echo 步骤1: 更新numpy和scipy...
pip uninstall -y numpy scipy
pip install numpy scipy --no-cache-dir

REM 2. 安装MKL（Intel Math Kernel Library）
echo.
echo 步骤2: 安装MKL支持...
pip install mkl mkl-service

REM 3. 安装预编译的科学计算包
echo.
echo 步骤3: 安装预编译包...
pip install --upgrade --force-reinstall numpy scipy scikit-learn

REM 4. 验证BLAS
echo.
echo 步骤4: 验证BLAS配置...
python -c "import numpy; print('NumPy版本:', numpy.__version__); numpy.show_config()"

echo.
echo ======================================================================
echo 修复完成！
echo ======================================================================
pause