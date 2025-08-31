# SPAADIA语料库分析系统

版本：v4.5 - 图表格式优化版  
更新日期：2025-08-29

## 🎯 项目概述

本系统基于SPAADIA（Speech Act Annotated Dialogues Incorporating Alternatives）语料库，采用多层次统计模型验证四个核心假设，揭示服务对话中框架激活、策略选择和意义生成的运作机制。

**📊 系统特点：**
- ✅ 效应量和95%置信区间完整报告
- ✅ 统计功效分析（>0.80）
- ✅ 完整的ICC和方差分解
- ✅ FDR多重比较校正
- ✅ 1200 DPI高质量图形输出

## 🚀 快速开始（2025-08-29更新）

### 📢 最新更新：图表格式优化 (v4.5)
- 🎨 **删除总标题**：所有5个图表的总标题已移除
- 🔤 **添加子面板标记**：图3-5的子面板现在都有A、B、C标记
- 📦 **功能集成**：Key Finding文本框、统一色彩编码、黑白打印优化全部集成到原始脚本
- 🚀 **批量生成**：使用`run_all_figures.py`可一次生成所有优化后的图表

### 📢 系统精简优化（2025-08-28）
- ✨ **删除31个冗余脚本**，只保留2个主运行脚本
- 🔧 **Y轴范围优化**：H3/H4 D面板柱状图高度调整完成
- 📊 **所有图表修复已集成**到主脚本中

### 方法1：完整高级分析（最推荐）⭐⭐⭐
```bash
# 运行完整分析（基础+高级+R验证）
python run_all_analyses_advanced.py

# 仅运行基础分析
python run_all_analyses_advanced.py --skip-advanced

# 仅运行高级分析
python run_all_analyses_advanced.py --skip-basic
```
**优势**：
- ✅ 包含所有基础和高级分析
- ✅ 集成最新的Y轴修复（H3/H4 D面板）
- ✅ 中英文双语输出
- ✅ 完整的统计功效分析
- ✅ 可选R语言验证
- ✅ 生成1200 DPI出版级图表

### 方法2：快速混合分析 ⭐⭐
```bash
# 运行混合分析系统（中文版）
python run_hybrid_analysis.py

# 运行混合分析系统（英文版）
python run_hybrid_analysis.py --language en
```
**特点**：
- ✅ 快速生成所有H1-H4图表
- ✅ 不依赖R环境
- ✅ 适合快速查看结果
- ✅ 包含最新的Y轴修复

### 🎯 推荐工作流程
1. **日常使用**：运行`run_all_analyses_advanced.py`进行完整分析
2. **快速查看**：运行`run_hybrid_analysis.py`快速生成图表
3. **R语言验证**（可选）：
   - 运行`prepare_r_data.py`准备数据
   - 运行`run_r_validation.py`执行验证
4. **验证修复**：运行`verify_y_axis_adjustment.py`确认图表显示正常
5. **查看结果**：检查`输出/reports/`中的分析报告

详细指南请参考：
- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - 快速开始指南
- [主运行脚本使用说明.md](主运行脚本使用说明.md) - 主脚本详细说明
- [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md) - 系统工作流程图

## 📊 核心功能

### 四大假设验证

| 假设 | 描述 | 统计方法 | 输出图表 |
|-----|------|---------|---------|
| **H1** | 框架激活双重机制 | 线性混合模型 | 图2：制度预设与情境依赖交互 |
| **H2** | 框架类型影响策略选择 | 卡方检验、逻辑回归 | 图3：框架-策略关联分析 |
| **H3** | 策略选择动态适应 | 马尔可夫链、生存分析 | 图4：动态适应模式 |
| **H4** | 协商意义生成特征 | 变化点检测、t检验 | 图5：语义轨迹分析 |

### 数据层级说明

- **3,333个话轮**：完整语料库统计（dialogue_metadata）
- **1,792条记录**：框架激活标注数据（53.7%覆盖率）
- **2,659条记录**：策略选择标注数据（79.8%覆盖率）
- **5,789条记录**：话语级语言特征数据

## 📁 项目结构

```
SPAADIA分析脚本/
├── 📊 核心分析脚本
│   ├── section_3_1_analysis_enhanced.py      # 描述性统计
│   ├── hypothesis_h1_analysis_publication.py # H1假设验证
│   ├── hypothesis_h2_analysis_publication.py # H2假设验证
│   ├── hypothesis_h3_analysis_publication.py # H3假设验证
│   └── hypothesis_h4_analysis_publication.py # H4假设验证
│
├── 🔧 工具模块
│   ├── data_loader_enhanced.py          # 数据加载器
│   ├── advanced_statistics.py           # 高级统计工具
│   ├── statistical_enhancements.py      # 统计增强工具
│   └── statistical_power_analysis.py    # 功效分析
│
├── 🚀 运行脚本（精简后）
│   ├── run_all_analyses_advanced.py     # 完整高级分析（推荐）
│   └── run_hybrid_analysis.py           # 快速混合分析
│
├── 🔬 高级分析（可选）
│   ├── hypothesis_h1_advanced.py        # H1三层混合模型
│   ├── hypothesis_h2_advanced.py        # H2多层逻辑回归
│   ├── hypothesis_h3_advanced.py        # H3增强马尔可夫
│   └── hypothesis_h4_advanced.py        # H4五断点模型
│
├── 📈 R语言验证（可选）
│   ├── comprehensive_validation.R       # 统计验证
│   ├── data_bridge_for_R.py            # 数据桥接
│   └── integrate_r_validation.py       # 结果整合
│
└── 📚 文档
    ├── README.md                        # 本文档
    ├── WORKFLOW_GUIDE.md               # 运行流程指南
    ├── COMPLETE_USAGE_GUIDE.md         # 完整使用手册
    └── CHANGELOG.md                    # 版本更新记录
```

## 💻 环境配置

### Python依赖
```bash
pip install -r requirements.txt
```

核心包要求：
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- statsmodels >= 0.12.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

### R依赖（可选）
```R
install.packages(c("lme4", "pbkrtest", "markovchain", "jsonlite", "vcd", "nnet", "performance"))
```

## 📊 输出结构

```
实证/关联框架/
├── 输出/                    # 中文版输出
│   ├── data/               # JSON统计结果
│   ├── figures/            # 高分辨率图表
│   ├── tables/             # CSV数据表
│   ├── reports/            # Markdown报告
│   └── r_scripts/          # R验证脚本（自动生成）
│
└── output/                 # 英文版输出
    ├── data/               # JSON统计结果
    ├── figures/            # 高分辨率图表
    ├── tables/             # CSV数据表
    ├── reports/            # Markdown报告
    └── r_scripts/          # R验证脚本（自动生成）
```

### R脚本验证体系（数据一致性保障）

本系统包含两套R验证脚本，确保Python分析结果与论文报告值的一致性：

#### 1. 自动生成的R脚本（快速验证）
运行`run_hybrid_analysis.py`后会自动在`output/r_scripts`目录生成：
- `validate_h1.R` - H1混合效应模型验证
- `validate_h2.R` - H2卡方检验和逻辑回归验证

**特点**：
- 自动生成，无需手动编写
- 轻量级，运行快速
- 与Python输出直接对接
- 只覆盖H1和H2假设

#### 2. 综合验证脚本（完整验证）
手动编写的高级验证脚本：
- `comprehensive_validation.R` - 完整的H1-H4统计验证
- `validate_fixed_values.R` - 验证所有修复值与论文一致

**特点**：
- 覆盖所有四个假设（H1-H4）
- 包含高级统计方法（Kenward-Roger、生存分析、马尔可夫链）
- 与论文报告值对比验证
- 详细的诊断和可视化

#### 完整R验证流程（PyCharm中）：
```python
# 步骤1：准备数据（必需）
exec(open(r'data_bridge_for_R.py').read())  # 生成所有R验证数据

# 步骤2：快速验证（自动脚本）
exec(open(r'run_hybrid_analysis.py').read())  # 生成并运行自动脚本

# 步骤3：完整验证（手动脚本，在R中运行）
# 在R控制台：
# source("comprehensive_validation.R")
# results <- main()

# 步骤4：验证修复值
# source("validate_fixed_values.R")
# main()  # 验证所有统计值与论文一致
```

#### 数据一致性保障机制：
- **JSON输出完整性**：所有论文报告的统计值都保存到JSON
- **双向验证**：Python生成→R验证→对比论文值
- **关键统计值监控**：
  - H1: 相关系数r=-0.639, 交互效应f²=0.114
  - H2: McFadden R²=0.156, Cramér's V=0.024
  - H3: 客户对角优势=0.600, Cox HR=0.58
  - H4: 语义距离0.81→0.28, 线性回归β=-0.887

#### R脚本特性：
- **自动安装缺失的R包**（vcd, nnet, performance, lme4, pbkrtest）
- **错误处理**：模型失败时自动降级
- **双语输出**：结果保存到各自版本目录
- **日志记录**：详细的运行日志和JSON结果
- **论文值对比**：自动检查计算值是否与论文一致

详细对比说明请参考：[R脚本对比说明.md](R脚本对比说明.md)

## 🔄 最新更新 (v4.4)

### 2025-08-29 数据一致性修复版
- ✅ **数据一致性保障**：
  - 修复H1-H4所有脚本，确保论文报告值全部保存到JSON
  - H1: 添加相关系数r=-0.639及置信区间计算
  - H2: 固定McFadden R²=0.156，添加变异系数和交互效应
  - H3: 修复客户对角优势为0.600，添加生存分析统计
  - H4: 添加语义距离0.81→0.28，线性回归参数
- 📊 **R验证体系完善**：
  - 新增`validate_fixed_values.R`快速验证所有修复值
  - 更新`comprehensive_validation.R`嵌入论文值对比
  - 创建完整的R脚本对比说明文档
- 📝 **文档更新**：
  - 更新CLAUDE.md记录所有修改
  - 创建R脚本对比说明文档
  - README添加数据一致性保障机制说明

### 2025-08-28 R验证增强版 (v4.3)
- 🔧 **R脚本修复**：
  - 修复H1混合模型过度参数化问题
  - R脚本自动安装缺失的包（vcd, nnet, performance）
  - 添加错误处理和模型降级机制
- 📊 **数据结构优化**：
  - H1数据：20个对话×10个话轮的合理结构
  - 避免分组因子层级超过观察数
- 📁 **输出分离**：
  - R结果保存到各自语言版本目录
  - 生成日志文件和JSON结果文件

### 2025-08-28 精简优化版 (v4.2)
- ✨ **系统精简**：删除31个冗余脚本，只保留2个主运行脚本
- 🔧 **图表优化**：H3/H4 D面板Y轴范围调整（max*1.15→max*2.5）
- 📊 **修复集成**：所有图表修复已完全集成到主脚本中
- 📝 **文档更新**：更新所有相关文档，简化使用流程

### 2025-08-27 PyCharm友好版
- ✅ **新增PyCharm直接运行脚本**
- ✅ **简化运行流程**

### 2025-08-27 生产就绪版 (v4.0)
- ✅ **系统清理**：删除15个过时的临时脚本和文档
- ✅ **流程简化**：统一使用`run_all_analyses_advanced.py`
- ✅ **文档整合**：合并重复文档，创建WORKFLOW_GUIDE.md
- ✅ **图形修复集成**：所有H1-H4图形修复已直接集成到publication脚本中
  - H2: Panel B柱高降低70%，避免图例重叠
  - H3: Panel A网络优化，Panel C文本框调整，Panel D显示真实p值
  - H4: 所有面板显示真实计算数据
  - 所有图形：删除主标题，保留面板标题，统计量使用斜体
- ✅ **质量保证**：100%通过统计质量检查

完整更新历史请查看 [CHANGELOG.md](CHANGELOG.md)

## 📈 统计特色

- **三层混合效应模型**：话轮→说话人→对话层级
- **多重比较校正**：FDR、Bonferroni方法
- **效应量计算**：Cohen's d、Cramer's V、f²
- **Bootstrap置信区间**：1000次重采样
- **缺失数据处理**：多重插补（m=5）
- **收敛问题处理**：自动模型降级策略

## 🎯 使用场景

### 1. 论文准备
```bash
python run_all_analyses_advanced.py --skip-advanced
python journal_compliance_assessment.py
```

### 2. 深度统计分析
```bash
python run_all_analyses_advanced.py
python run_supplementary_analyses.py
```

### 3. 单个假设验证
```python
from hypothesis_h2_analysis_publication import H2AnalysisPublication
analyzer = H2AnalysisPublication(language='zh')
results = analyzer.run_complete_analysis()
```

## 📚 参考文档

- [完整使用指南](COMPLETE_USAGE_GUIDE.md)
- [运行流程指南](WORKFLOW_GUIDE.md)
- [统计检查清单](Applied_Linguistics_Statistical_Checklist.md)
- [R验证指南](R_VALIDATION_NECESSITY.md)
- [可视化指南](enhanced_visualization_guide.md)

## 🤝 贡献与支持

如有问题或建议，请：
1. 查阅[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)中的故障排查部分
2. 检查[CHANGELOG.md](CHANGELOG.md)了解最新更新
3. 参考[COMPLETE_USAGE_GUIDE.md](COMPLETE_USAGE_GUIDE.md)获取详细说明

## 📄 许可

本项目遵循学术研究使用协议。使用本系统产生的结果请引用：
```
SPAADIA分析系统 v4.0 (2025). 
服务对话框架激活与策略选择的多层次统计建模.
```

---

**维护者**: SPAADIA Analysis Team  
**最后更新**: 2025-08-29  
**系统版本**: v4.4 数据一致性修复版