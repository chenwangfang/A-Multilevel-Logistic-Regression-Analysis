# Construal-Driven Frame Activation and Strategy Selection in Service Dialogues: A Multilevel Statistical Analysis
# 服务对话中识解驱动的框架激活与策略选择：多层统计分析

[![Version](https://img.shields.io/badge/Version-2.1-blue)](https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis/releases)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.2%2B-276DC3)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Statistical Power](https://img.shields.io/badge/Power-59.8%25-yellow)](scripts/statistical_power_analysis.py)
[![FDR Corrected](https://img.shields.io/badge/FDR-BH_Corrected-informational)](scripts/statistical_enhancements.py)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 🌟 English Version

### Project Overview

This repository contains the complete implementation of multilevel statistical analysis for the SPAADIA corpus (35 dialogues, 3,333 turns), integrating cognitive linguistic construal theory with institutional discourse analysis. The research empirically validates how construal operations drive frame activation and strategy selection in service dialogues, revealing systematic connections between cognitive mechanisms and institutional discourse patterns. The XML-JSON hybrid annotation system enables quantification of construal phenomena for large-scale empirical analysis.

### 📊 Key Findings

- **H1**: Frame activation exhibits dual mechanisms of context dependency (β = -.317) and institutional presupposition (β = .253) with medium interaction effect (*f*² = 0.114), **ICC**_speaker = 0.425, **ICC**_dialogue = 0.000
- **H2**: Frame types significantly predict strategy selection (χ² = 62.24, *p* < .001, Cramér's *V* = 0.259), with service initiation frames showing strong preference for frame reinforcement (OR = 15.33)
- **H3**: Strategy evolution demonstrates path dependency (diagonal dominance = 0.533) with effectiveness decay (β = -.082, *p* = .001), customer decay exceeds service provider
- **H4**: Semantic distance decreases from 0.836 to 0.738 (11.7% reduction, *d* = 1.25), with key negotiation points at turns 5 and 12 (CUSUM = 0.29)

### 🚀 Quick Start

#### Installation
```bash
# Clone the repository
git clone https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis.git
cd A-Multilevel-Logistic-Regression-Analysis

# Install Python dependencies
pip install -r requirements.txt

# Optional: Install R packages for validation
Rscript -e "install.packages(c('lme4', 'pbkrtest', 'jsonlite', 'vcd', 'nnet', 'performance'))"
```

#### Run Complete Analysis
```bash
# Navigate to scripts directory
cd scripts

# Option 1: Run complete analysis with advanced statistics (Recommended)
python run_all_analyses_advanced.py

# Option 2: Quick hybrid analysis for core results
python run_hybrid_analysis.py

# Option 3: Generate all figures only
python run_all_figures.py

# Optional: Run individual figure scripts
python figure1_theoretical_framework.py  # Generate theoretical framework
python figure2_dual_mechanism.py         # Generate H1 dual mechanism
python figure3_frame_strategy_sankey.py  # Generate H2 Sankey diagram
python figure4_markov_evolution.py       # Generate H3 Markov evolution
python figure5_semantic_convergence.py   # Generate H4 semantic convergence

# Optional: Run R validation
python run_r_validation.py

# Optional: Calculate ICC separately
python three_level_icc_python.py        # Python implementation
python run_r_icc_analysis.py           # R validation (requires R)
```

#

### 📎 Supplementary Materials / 补充材料

The `documentation/` folder contains essential supplementary materials:

1. **Supplementary_Materials.md**: Complete supplementary materials including:
   - Detailed statistical methods
   - Extended results tables
   - Additional analyses
   - Robustness checks

2. **XML-JSON_Hybrid_Architecture.md**: Technical documentation of the XML-JSON hybrid triple architecture:
   - Data structure design
   - Processing pipeline
   - Integration methodology

3. **Coding_Scheme.md**: Complete coding scheme for SPAADIA corpus:
   - Frame type definitions
   - Strategy classifications
   - Annotation guidelines
   - Inter-rater reliability metrics

These documents provide comprehensive technical details supporting the main analysis.

## 📁 Repository Structure

```
├── SPAADIA/                     # Corpus data (35 dialogues)
│   ├── indices/                 # JSONL index files
│   ├── metadata/                # JSON metadata files
│   └── xml_annotations/         # XML annotation files
├── scripts/                     # Analysis scripts
│   ├── Core Analyses/
│   │   ├── hypothesis_h1_analysis_publication.py  # H1: Frame activation dual mechanisms
│   │   ├── hypothesis_h2_analysis_publication.py  # H2: Frame-strategy association
│   │   ├── hypothesis_h3_analysis_publication.py  # H3: Dynamic strategy adaptation
│   │   └── hypothesis_h4_analysis_publication.py  # H4: Semantic negotiation
│   ├── Advanced Analyses/
│   │   ├── hypothesis_h1_advanced.py             # Advanced H1 with ICC correction
│   │   ├── hypothesis_h1_enhanced.py             # Enhanced H1 analysis
│   │   ├── hypothesis_h2_advanced.py             # Advanced H2 analysis
│   │   ├── hypothesis_h2_enhanced.py             # Enhanced H2 analysis
│   │   ├── hypothesis_h2_enhanced_feature_engineering.py  # Feature engineering for H2
│   │   ├── hypothesis_h3_advanced.py             # Advanced H3 analysis
│   │   └── hypothesis_h4_advanced.py             # Advanced H4 analysis
│   ├── Figure Generation/
│   │   ├── figure1_theoretical_framework.py      # Theoretical framework diagram
│   │   ├── figure2_dual_mechanism.py             # Dual mechanism visualization
│   │   ├── figure3_frame_strategy_sankey.py      # Sankey diagram for frame-strategy
│   │   ├── figure4_markov_evolution.py           # Markov chain evolution
│   │   ├── figure5_semantic_convergence.py       # Semantic convergence patterns
│   │   ├── create_flowchart_pillow.py           # Analysis flowchart (Pillow)
│   │   ├── create_multilevel_flowchart.py       # Multilevel analysis flowchart
│   │   └── run_all_figures.py                   # Generate all figures at once
│   ├── Statistical Enhancement/
│   │   ├── statistical_power_analysis.py         # Power analysis (59.8% power)
│   │   ├── statistical_enhancements.py           # FDR correction and enhancements
│   │   ├── advanced_statistics.py                # Advanced statistical methods
│   │   ├── section_3_1_analysis_enhanced.py      # Section 3.1 descriptive statistics
│   │   └── three_level_icc_python.py            # ICC calculation module
│   ├── R Integration/
│   │   ├── run_r_validation.py                   # Python-R validation interface
│   │   ├── run_r_icc_analysis.py                # Python-R interface for ICC
│   │   ├── integrate_r_validation.py            # Integrate R validation results
│   │   ├── integrate_validation_results.py      # Consolidate validation outputs
│   │   ├── data_bridge_for_R.py                 # Prepare data for R analysis
│   │   ├── comprehensive_validation.R           # Comprehensive R validation
│   │   ├── simple_r_validation.R                # Simplified R validation
│   │   ├── three_level_icc_analysis.R          # R lme4 ICC validation
│   │   └── three_level_icc_analysis_windows.R  # Windows-compatible R script
│   ├── Main Runners/
│   │   ├── run_all_analyses_advanced.py         # Run complete analysis pipeline
│   │   └── run_hybrid_analysis.py               # Quick hybrid analysis
│   └── Data Loading/
│       └── data_loader_enhanced.py              # Enhanced data loader for SPAADIA
├── output/                      # Analysis outputs
│   ├── data/                   # JSON statistical results
│   ├── figures/                # Publication figures (1200 DPI)
│   ├── tables/                 # CSV result tables
│   └── reports/                # Markdown reports
└── documentation/
    ├── XML-JSON_Hybrid_Architecture.md
    ├── Statistical_Methods.md
    └── Coding_Scheme.md
```

### 🔬 Technical Implementation

#### Script Statistics
- **Total Scripts**: 36 (32 Python + 4 R scripts)
- **Core Analysis**: 4 publication-ready hypothesis tests
- **Advanced Analysis**: 7 enhanced versions with additional features
- **Figure Generation**: 8 scripts for publication-quality visualizations
- **Statistical Enhancement**: 5 scripts for power analysis and corrections
- **R Integration**: 9 scripts for cross-validation and integration

#### Software Environment
- **Python 3.9+**: pandas, numpy, scipy, statsmodels 0.14+, scikit-learn, matplotlib 3.5+, seaborn 0.12+
- **R 4.2+** (optional): lme4, pbkrtest, jsonlite, vcd, nnet, performance
- **Analysis Pipeline**: Hybrid Python-R approach with Python for primary analysis and R for validation

#### Statistical Methods
- **Three-level linear mixed models** with proper ICC calculation (speaker-level ICC = 0.425)
  - Variance decomposition: 57.5% turn-level, 42.5% speaker-level, 0% dialogue-level
  - ANOVA method for three-level nested structure
- **Multinomial logistic regression** with clustered robust standard errors
- **Markov chain analysis** with stationary distribution and mixing time
- **Piecewise growth curve models** with CUSUM change-point detection
- **Multiple comparison correction**: Benjamini-Hochberg FDR at *q* = 0.05
- **Power analysis**: Monte Carlo simulation (1,000 iterations), achieving 59.8% power

### 📈 Key Outputs

#### Figures (1200 DPI)
1. `comprehensive_results.jpg`: SPAADIA analysis overview
2. `figure_h1_dual_mechanism_publication.jpg`: Frame activation dual mechanisms
3. `figure_h2_frame_strategy_publication.jpg`: Frame-strategy associations
4. `figure_h3_dynamic_adaptation_publication.jpg`: Dynamic adaptation patterns
5. `figure_h4_negotiation_publication.jpg`: Semantic convergence in negotiation

#### Data Files
- Complete statistical results in JSON format
- R validation outputs
- APA-formatted tables in CSV

### 📚 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{chen2025construal,
  title={Construal-Driven Frame Activation and Strategy Selection in Service Dialogues: 
         A Multilevel Logistic Regression Analysis},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2025},
  doi={10.XXXX/XXXXX}
}

@software{spaadia_analysis_2025,
  title={SPAADIA Multilevel Statistical Analysis Framework},
  author={[Contributors]},
  year={2025},
  version={2.1},
  url={https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis},
  doi={10.5281/zenodo.XXXXXX}
}
```

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📮 Contact

- **Email**: [corresponding author email]
- **Issues**: [GitHub Issues](https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis/issues)

---

<a name="chinese"></a>
## 🌟 中文版本

### 项目概述

本仓库包含SPAADIA语料库（35个对话，3,333轮）多层统计分析的完整实现，整合认知语言学识解理论与机构话语分析。研究实证验证了识解操作如何驱动服务对话中的框架激活和策略选择，揭示了认知机制与机构话语模式之间的系统性联系。XML-JSON混合标注系统实现了识解现象的量化，支持大规模实证分析。

### 📊 主要发现

- **H1**：框架激活呈现情境依赖（β = -.317）和制度预设（β = .253）双重机制，交互效应量中等（*f*² = 0.114），**ICC**_说话人 = 0.425，**ICC**_对话 = 0.000
- **H2**：框架类型显著预测策略选择（χ² = 62.24, *p* < .001, Cramér's *V* = 0.259），服务启动框架强烈偏好框架强化策略（OR = 15.33）
- **H3**：策略演化呈现路径依赖（对角优势 = 0.533）和效力衰减（β = -.082, *p* = .001），客户衰减系数超过服务提供者
- **H4**：语义距离从0.836降至0.738（11.7%降低，*d* = 1.25），关键协商点位于第5和12轮（CUSUM = 0.29）

### 🚀 快速开始

#### 安装
```bash
# 克隆仓库
git clone https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis.git
cd A-Multilevel-Logistic-Regression-Analysis

# 安装Python依赖
pip install -r requirements.txt

# 可选：安装R包用于验证
Rscript -e "install.packages(c('lme4', 'pbkrtest', 'jsonlite', 'vcd', 'nnet', 'performance'))"
```

#### 运行完整分析
```bash
# 进入脚本目录
cd scripts

# 选项1：运行包含高级统计的完整分析（推荐）
python run_all_analyses_advanced.py

# 选项2：快速混合分析获取核心结果
python run_hybrid_analysis.py

# 选项3：仅生成所有图表
python run_all_figures.py

# 可选：运行单个图表脚本
python figure1_theoretical_framework.py  # 生成理论框架图
python figure2_dual_mechanism.py         # 生成H1双重机制图
python figure3_frame_strategy_sankey.py  # 生成H2桑基图
python figure4_markov_evolution.py       # 生成H3马尔可夫演化图
python figure5_semantic_convergence.py   # 生成H4语义收敛图

# 可选：运行R验证
python run_r_validation.py

# 可选：单独计算ICC
python three_level_icc_python.py        # Python实现
python run_r_icc_analysis.py           # R验证（需要R环境）
```

### 📁 仓库结构

```
├── SPAADIA/                     # 语料库数据（35个对话）
│   ├── indices/                 # JSONL索引文件
│   ├── metadata/                # JSON元数据文件
│   └── xml_annotations/         # XML标注文件
├── scripts/                     # 分析脚本
│   ├── 核心分析/
│   │   ├── hypothesis_h1_analysis_publication.py  # H1：框架激活双重机制
│   │   ├── hypothesis_h2_analysis_publication.py  # H2：框架-策略关联
│   │   ├── hypothesis_h3_analysis_publication.py  # H3：动态策略适应
│   │   └── hypothesis_h4_analysis_publication.py  # H4：语义协商
│   ├── 高级分析/
│   │   ├── hypothesis_h1_advanced.py             # 高级H1分析（含ICC修正）
│   │   ├── hypothesis_h1_enhanced.py             # 增强H1分析
│   │   ├── hypothesis_h2_advanced.py             # 高级H2分析
│   │   ├── hypothesis_h2_enhanced.py             # 增强H2分析
│   │   ├── hypothesis_h2_enhanced_feature_engineering.py  # H2特征工程
│   │   ├── hypothesis_h3_advanced.py             # 高级H3分析
│   │   └── hypothesis_h4_advanced.py             # 高级H4分析
│   ├── 图表生成/
│   │   ├── figure1_theoretical_framework.py      # 理论框架图
│   │   ├── figure2_dual_mechanism.py             # 双重机制可视化
│   │   ├── figure3_frame_strategy_sankey.py      # 框架-策略桑基图
│   │   ├── figure4_markov_evolution.py           # 马尔可夫链演化
│   │   ├── figure5_semantic_convergence.py       # 语义收敛模式
│   │   ├── create_flowchart_pillow.py           # 分析流程图（Pillow版）
│   │   ├── create_multilevel_flowchart.py       # 多层分析流程图
│   │   └── run_all_figures.py                   # 生成所有图表
│   ├── 统计增强/
│   │   ├── statistical_power_analysis.py         # 功效分析（59.8%功效）
│   │   ├── statistical_enhancements.py           # FDR校正和增强
│   │   ├── advanced_statistics.py                # 高级统计方法
│   │   ├── section_3_1_analysis_enhanced.py      # 第3.1节描述性统计
│   │   └── three_level_icc_python.py            # ICC计算模块
│   ├── R集成/
│   │   ├── run_r_validation.py                   # Python-R验证接口
│   │   ├── run_r_icc_analysis.py                # ICC的Python-R接口
│   │   ├── integrate_r_validation.py            # 整合R验证结果
│   │   ├── integrate_validation_results.py      # 合并验证输出
│   │   ├── data_bridge_for_R.py                 # 为R分析准备数据
│   │   ├── comprehensive_validation.R           # 综合R验证
│   │   ├── simple_r_validation.R                # 简化R验证
│   │   ├── three_level_icc_analysis.R          # R lme4 ICC验证
│   │   └── three_level_icc_analysis_windows.R  # Windows兼容R脚本
│   ├── 主运行器/
│   │   ├── run_all_analyses_advanced.py         # 运行完整分析流程
│   │   └── run_hybrid_analysis.py               # 快速混合分析
│   └── 数据加载/
│       └── data_loader_enhanced.py              # SPAADIA增强数据加载器
├── 输出/                        # 分析输出
│   ├── data/                   # JSON统计结果
│   ├── figures/                # 发表级图表（1200 DPI）
│   ├── tables/                 # CSV结果表格
│   └── reports/                # Markdown报告
└── 文档/
    ├── XML-JSON混合架构.md
    ├── 统计方法说明.md
    └── 编码方案.md
```

### 🔬 技术实现

#### 脚本统计
- **脚本总数**：36个（32个Python + 4个R脚本）
- **核心分析**：4个发表级假设检验
- **高级分析**：7个带附加功能的增强版本
- **图表生成**：8个生成发表级可视化的脚本
- **统计增强**：5个功效分析和校正脚本
- **R集成**：9个用于交叉验证和集成的脚本

#### 软件环境
- **Python 3.9+**：pandas、numpy、scipy、statsmodels 0.14+、scikit-learn、matplotlib 3.5+、seaborn 0.12+
- **R 4.2+**（可选）：lme4、pbkrtest、jsonlite、vcd、nnet、performance
- **分析管道**：Python-R混合方法，Python负责主要分析，R提供验证

#### 统计方法
- **三层线性混合模型**，正确的ICC计算（说话人层ICC = 0.425）
  - 方差分解：57.5%话轮层，42.5%说话人层，0%对话层
  - 采用ANOVA方法处理三层嵌套结构
- **多项逻辑回归**，带聚类稳健标准误
- **马尔可夫链分析**，含稳态分布和混合时间
- **分段增长曲线模型**，带CUSUM变化点检测
- **多重比较校正**：Benjamini-Hochberg FDR，*q* = 0.05
- **功效分析**：蒙特卡罗模拟（1,000次迭代），达到59.8%功效

### 📈 主要输出

#### 图表（1200 DPI）
1. `comprehensive_results.jpg`：SPAADIA分析总览
2. `figure_h1_dual_mechanism_publication.jpg`：框架激活双重机制
3. `figure_h2_frame_strategy_publication.jpg`：框架-策略关联
4. `figure_h3_dynamic_adaptation_publication.jpg`：动态适应模式
5. `figure_h4_negotiation_publication.jpg`：协商中的语义收敛

#### 数据文件
- JSON格式的完整统计结果
- R验证输出
- CSV格式的APA规范表格

### 📚 引用

如果您在研究中使用此代码或数据，请引用：

```bibtex
@article{chen2025construal,
  title={Construal-Driven Frame Activation and Strategy Selection in Service Dialogues: A Multilevel Logistic Regression Analysis},
  author={[作者姓名]},
  journal={[期刊名称]},
  year={2025},
  doi={10.XXXX/XXXXX}
}

@software{spaadia_analysis_2025,
  title={SPAADIA Multilevel Statistical Analysis Framework},
  author={[贡献者]},
  year={2025},
  version={2.1},
  url={https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis},
  doi={10.5281/zenodo.XXXXXX}
}
```

### 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

### 🤝 贡献

欢迎贡献！请：
1. Fork本仓库
2. 创建功能分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 创建Pull Request

### 📮 联系方式

- **邮箱**：[通讯作者邮箱]
- **问题**：[GitHub Issues](https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis/issues)

---

## 📝 Recent Updates / 最近更新

### Version 2.1 (2025-08-31)
- **CRITICAL FIX**: Corrected ICC calculation for three-level nested models
  - Fixed speaker_id field issue in hypothesis_h1_advanced.py
  - Implemented proper ANOVA variance decomposition method
  - Unified ICC values across all scripts (speaker ICC = 0.425, dialogue ICC = 0.000)
- **NEW**: Added dedicated ICC calculation modules
  - `three_level_icc_python.py`: Python implementation using ANOVA method
  - `run_r_icc_analysis.py`: Python-R interface for validation
  - `three_level_icc_analysis.R`: R lme4 implementation
- **IMPROVED**: Enhanced statistical transparency with complete variance decomposition reporting
- **CLEANED**: Removed temporary test files and optimized codebase structure

### 版本 2.1 (2025-08-31)
- **关键修复**：修正了三层嵌套模型的ICC计算
  - 修复了hypothesis_h1_advanced.py中的speaker_id字段问题
  - 实现了正确的ANOVA方差分解方法
  - 统一了所有脚本的ICC值（说话人ICC = 0.425，对话ICC = 0.000）
- **新增**：添加了专门的ICC计算模块
  - `three_level_icc_python.py`：使用ANOVA方法的Python实现
  - `run_r_icc_analysis.py`：用于验证的Python-R接口
  - `three_level_icc_analysis.R`：R lme4实现
- **改进**：通过完整的方差分解报告增强了统计透明度
- **清理**：删除了临时测试文件并优化了代码库结构

---

## 🔍 Statistical Transparency Statement / 统计透明度声明

This research adheres to international journal standards for statistical reporting. All analyses include:
- Effect sizes with 95% confidence intervals
- FDR-corrected p-values alongside uncorrected values
- Complete model specifications and diagnostic results
- Raw data and analysis code for reproducibility

本研究遵循国际期刊的统计报告标准。所有分析包括：
- 带95%置信区间的效应量
- FDR校正和未校正的p值
- 完整的模型规范和诊断结果
- 用于可重现性的原始数据和分析代码