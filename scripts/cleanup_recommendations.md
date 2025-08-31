# 脚本清理建议

## 可以安全删除的临时/测试文件：

### 1. 测试脚本
- **test_mnlogit.py** - 仅用于测试MNLogit模型的测试脚本
  ```bash
  rm test_mnlogit.py
  ```

### 2. 临时数据文件
- **temp_data_for_r.csv** - R脚本的临时数据文件
- **three_level_icc_results.json** - R脚本的临时输出
  ```bash
  rm temp_data_for_r.csv
  rm three_level_icc_results.json
  ```

### 3. 重复的流程图脚本（选择保留一个）
- **create_flowchart_pillow.py** - Pillow版本
- **create_multilevel_flowchart.py** - 基础版本
  建议：保留一个即可

### 4. 缓存文件
- **__pycache__/** - Python缓存目录
  ```bash
  rm -rf __pycache__
  ```

### 5. 奇怪的路径文件
- **G:** - 看起来是错误创建的文件
- **G:\Project\实证\关联框架\输出\figures** - 错误创建的目录
  ```bash
  rm -rf "G:" "G:\Project\实证\关联框架\输出\figures"
  ```

## 建议保留的脚本：

### 核心分析脚本
- data_loader_enhanced.py - 数据加载器
- run_all_analyses_advanced.py - 主运行器
- run_hybrid_analysis.py - 快速分析器
- hypothesis_h*_analysis_publication.py - 4个假设检验脚本

### 可视化脚本
- figure1_theoretical_framework.py
- figure2_dual_mechanism.py
- figure3_frame_strategy_sankey.py
- figure4_markov_evolution.py
- figure5_semantic_convergence.py
- run_all_figures.py

### ICC验证脚本
- three_level_icc_python.py - Python ICC计算
- run_r_icc_analysis.py - R集成
- three_level_icc_analysis.R - R脚本
- three_level_icc_analysis_windows.R - Windows版R脚本

### 增强版脚本
- hypothesis_h*_advanced.py - 高级分析版本
- section_3_1_analysis_enhanced.py - 描述性统计

### R集成脚本
- data_bridge_for_R.py - R数据桥接
- run_r_validation.py - R验证
- integrate_r_validation.py - 结果集成

## 可能可以合并的脚本：

1. **hypothesis_h*_enhanced.py vs hypothesis_h*_advanced.py**
   - 建议：保留advanced版本，它更完善

2. **statistical_enhancements.py vs advanced_statistics.py**
   - 需要检查功能是否重复

## 清理命令汇总：

```bash
# 在Linux/WSL中执行
cd /mnt/g/Project/实证/关联框架/Python脚本/SPAADIA分析脚本

# 删除临时文件
rm -f test_mnlogit.py temp_data_for_r.csv three_level_icc_results.json

# 删除缓存
rm -rf __pycache__

# 删除错误创建的文件/目录
rm -rf "G:" "G:\\Project\\实证\\关联框架\\输出\\figures"

# 可选：删除enhanced版本（如果决定只保留advanced版本）
# rm -f hypothesis_h*_enhanced.py
```

## 注意事项：

1. 执行删除前请确认已备份重要文件
2. 建议先移动到临时目录而不是直接删除：
   ```bash
   mkdir -p /tmp/spaadia_cleanup
   mv test_mnlogit.py temp_data_for_r.csv three_level_icc_results.json /tmp/spaadia_cleanup/
   ```

3. 保留requirements.txt和install_blas_fix.bat - 这些是环境配置文件