# Construal-Driven Frame Activation and Strategy Selection in Service Dialogues: A Multilevel Statistical Analysis
# 服务对话中识解驱动的框架激活与策略选择：多层统计分析

[![Version](https://img.shields.io/badge/Version-2.0-blue)](https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis/releases)
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

This repository contains the complete implementation of multilevel statistical analysis for the SPAADIA corpus, integrating cognitive linguistic construal theory with institutional discourse analysis. The project validates four interconnected hypotheses about frame activation, strategy selection, and meaning negotiation mechanisms in service dialogues through rigorous statistical modeling.

### 📊 Key Findings

- **H1**: Frame activation exhibits dual mechanisms (context dependency × institutional presetting), interaction effect *f*² = 0.114
- **H2**: Limited frame-strategy association, χ²(6) = 3.32, *p* = 0.768, Cramér's *V* = 0.024
- **H3**: Moderate path dependency in strategy transitions, diagonal dominance = 0.533, mixing time = 2 turns
- **H4**: Structured negotiation dynamics with change points at turns 5 and 12, piecewise *R*² = 0.42

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

# Option 2: Quick hybrid analysis for figures only
python run_hybrid_analysis.py

# Optional: Run R validation
python run_r_validation.py
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
│   │   ├── hypothesis_h1_analysis_publication.py
│   │   ├── hypothesis_h2_analysis_publication.py
│   │   ├── hypothesis_h3_analysis_publication.py
│   │   └── hypothesis_h4_analysis_publication.py
│   ├── Advanced Analyses/
│   │   ├── hypothesis_h1_advanced.py
│   │   ├── hypothesis_h2_advanced.py
│   │   ├── hypothesis_h3_advanced.py
│   │   └── hypothesis_h4_advanced.py
│   ├── Statistical Enhancement/
│   │   ├── statistical_power_analysis.py
│   │   ├── statistical_enhancements.py
│   │   └── advanced_statistics.py
│   └── Main Runners/
│       ├── run_all_analyses_advanced.py
│       └── run_hybrid_analysis.py
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

#### Software Environment
- **Python 3.9+**: pandas, numpy, scipy, statsmodels 0.14+, scikit-learn, matplotlib 3.5+, seaborn 0.12+
- **R 4.2+** (optional): lme4, pbkrtest, jsonlite, vcd, nnet, performance
- **Analysis Pipeline**: Hybrid Python-R approach with Python for primary analysis and R for validation

#### Statistical Methods
- **Three-level linear mixed models** with Kenward-Roger approximation
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
         A Multilevel Statistical Analysis},
  author={Author Name},
  journal={Applied Linguistics},
  year={2025},
  doi={10.1093/applin/XXXXX}
}

@software{spaadia_analysis_2025,
  title={SPAADIA Multilevel Statistical Analysis Framework},
  author={Author Name},
  year={2025},
  version={2.0},
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

本仓库包含SPAADIA语料库多层统计分析的完整实现，整合认知语言学识解理论与机构话语分析。项目通过严格的统计建模验证了服务对话中框架激活、策略选择和意义协商机制的四个相互关联假设。

### 📊 主要发现

- **H1**：框架激活呈现双重机制（语境依赖×机构预设），交互效应 *f*² = 0.114
- **H2**：框架-策略关联有限，χ²(6) = 3.32, *p* = 0.768, Cramér's *V* = 0.024
- **H3**：策略转换中度路径依赖，对角优势 = 0.533，混合时间 = 2个话轮
- **H4**：结构化协商动态，变化点在第5轮和第12轮，分段 *R*² = 0.42

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

# 选项2：仅生成图表的快速混合分析
python run_hybrid_analysis.py

# 可选：运行R验证
python run_r_validation.py
```

### 📁 仓库结构

```
├── SPAADIA/                     # 语料库数据（35个对话）
│   ├── indices/                 # JSONL索引文件
│   ├── metadata/                # JSON元数据文件
│   └── xml_annotations/         # XML标注文件
├── scripts/                     # 分析脚本
│   ├── 核心分析/
│   │   ├── hypothesis_h1_analysis_publication.py
│   │   ├── hypothesis_h2_analysis_publication.py
│   │   ├── hypothesis_h3_analysis_publication.py
│   │   └── hypothesis_h4_analysis_publication.py
│   ├── 高级分析/
│   │   ├── hypothesis_h1_advanced.py
│   │   ├── hypothesis_h2_advanced.py
│   │   ├── hypothesis_h3_advanced.py
│   │   └── hypothesis_h4_advanced.py
│   ├── 统计增强/
│   │   ├── statistical_power_analysis.py
│   │   ├── statistical_enhancements.py
│   │   └── advanced_statistics.py
│   └── 主运行器/
│       ├── run_all_analyses_advanced.py
│       └── run_hybrid_analysis.py
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

#### 软件环境
- **Python 3.9+**：pandas、numpy、scipy、statsmodels 0.14+、scikit-learn、matplotlib 3.5+、seaborn 0.12+
- **R 4.2+**（可选）：lme4、pbkrtest、jsonlite、vcd、nnet、performance
- **分析管道**：Python-R混合方法，Python负责主要分析，R提供验证

#### 统计方法
- **三层线性混合模型**，带Kenward-Roger近似
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
  title={服务对话中识解驱动的框架激活与策略选择：多层统计分析},
  author={作者姓名},
  journal={Applied Linguistics},
  year={2025},
  doi={10.1093/applin/XXXXX}
}

@software{spaadia_analysis_2025,
  title={SPAADIA多层统计分析框架},
  author={作者姓名},
  year={2025},
  version={2.0},
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

## 🔍 Statistical Transparency Statement / 统计透明度声明

This research adheres to Applied Linguistics journal standards for statistical reporting. All analyses include:
- Effect sizes with 95% confidence intervals
- FDR-corrected p-values alongside uncorrected values
- Complete model specifications and diagnostic results
- Raw data and analysis code for reproducibility

本研究遵循Applied Linguistics期刊的统计报告标准。所有分析包括：
- 带95%置信区间的效应量
- FDR校正和未校正的p值
- 完整的模型规范和诊断结果
- 用于可重现性的原始数据和分析代码