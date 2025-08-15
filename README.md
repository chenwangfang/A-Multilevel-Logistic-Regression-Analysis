# A Multilevel Logistic Regression Analysis / 多层逻辑回归分析

[![Version](https://img.shields.io/github/v/release/chenwangfang/A-Multilevel-Logistic-Regression-Analysis)](https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis/releases)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.2%2B-276DC3)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Statistical Power](https://img.shields.io/badge/Power-%3E80%25-success)](Python脚本/SPAADIA分析脚本/power_analysis.py)
[![FDR Corrected](https://img.shields.io/badge/FDR-Corrected-informational)](Python脚本/SPAADIA分析脚本/fdr_correction.py)

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 🌟 English Version

### Project Overview

This repository contains a comprehensive statistical analysis framework for the SPAADIA (Speech Act Annotated Dialogues Incorporating Alternatives) corpus, implementing multilevel statistical models to validate four core hypotheses about frame activation, strategy selection, and meaning generation mechanisms in service dialogues.

### 🆕 Latest Updates (2025-08-15)

#### Supplementary Analysis Modules
To ensure statistical completeness and rigor, four supplementary analysis modules have been added:

1. **Power Analysis** (`power_analysis.py`)
   - Monte Carlo simulation to evaluate statistical power
   - Ensures adequate sample size to detect medium effects (Cohen's d = 0.5)

2. **FDR Multiple Comparison Correction** (`fdr_correction.py`)
   - Benjamini-Hochberg method to control false discovery rate
   - Distinguishes between critical theoretical tests and exploratory analyses

3. **Sensitivity Analysis** (`sensitivity_analysis.py`)
   - Three-dimensional evaluation of result robustness
   - Semantic distance methods, threshold parameters, model structures

4. **Integrated Runner** (`run_supplementary_analyses.py`)
   - One-click execution of all supplementary analyses
   - Generates comprehensive reports

### 🚀 Quick Start

#### Installation
```bash
# Clone the repository
git clone https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis.git
cd A-Multilevel-Logistic-Regression-Analysis

# Install dependencies
pip install -r requirements.txt
```

#### Run Complete Analysis
```bash
# Navigate to scripts directory
cd Python脚本/SPAADIA分析脚本

# Run main analyses (basic + advanced)
python run_all_analyses_advanced.py

# Run supplementary analyses (NEW)
python run_supplementary_analyses.py

# Optional: R validation
Rscript comprehensive_validation.R
```

### 📊 Core Features

- **Four Hypothesis Testing**:
  - H1: Dual mechanisms of frame activation
  - H2: Frame-driven strategy selection
  - H3: Path dependency in strategy evolution
  - H4: Semantic convergence in meaning negotiation

- **Advanced Statistical Methods**:
  - Three-level linear mixed models
  - Multinomial logistic regression with clustered robust standard errors
  - Markov chain analysis
  - Piecewise growth curve models

- **Rigorous Statistical Controls**:
  - Statistical power analysis
  - FDR correction for multiple comparisons
  - Comprehensive sensitivity analyses
  - Cross-validation with R

### 📁 Repository Structure

```
├── Python脚本/
│   └── SPAADIA分析脚本/
│       ├── Core Analyses/
│       │   ├── hypothesis_h1_analysis.py
│       │   ├── hypothesis_h2_analysis.py
│       │   ├── hypothesis_h3_analysis.py
│       │   └── hypothesis_h4_analysis.py
│       ├── Advanced Analyses/
│       │   ├── hypothesis_h1_advanced.py
│       │   ├── hypothesis_h2_advanced.py
│       │   ├── hypothesis_h3_advanced.py
│       │   └── hypothesis_h4_advanced.py
│       ├── Supplementary Analyses (NEW)/
│       │   ├── power_analysis.py
│       │   ├── fdr_correction.py
│       │   ├── sensitivity_analysis.py
│       │   └── run_supplementary_analyses.py
│       └── Utilities/
│           ├── data_loader_enhanced.py
│           └── advanced_statistics.py
├── 背景资料/
│   └── SPAADIA分析上下文/
│       └── 2.4小节_cn_updated.md
└── 输出/output/
    ├── data/      # JSON statistical results
    ├── tables/    # CSV data tables
    ├── figures/   # High-resolution plots (1200 DPI)
    └── reports/   # Markdown reports
```

### 🔬 Technical Implementation

- **Languages**: Python 3.9+, R 4.2+ (for validation)
- **Key Libraries**: 
  - Python: pandas, numpy, statsmodels, scikit-learn, lifelines
  - R: lme4, pbkrtest, markovchain, nnet
- **Output**: Bilingual (Chinese/English) results

### 📈 Key Results

- Statistical power > 80% for main effects with current sample size (118 dialogues)
- FDR-corrected p-values for all hypothesis tests
- Robust results across different methodological choices
- Cross-validated findings between Python and R

### 📝 Documentation

- [Statistical Modeling Methods](背景资料/SPAADIA分析上下文/2.4小节_cn_updated.md)
- [Running Instructions](Python脚本/SPAADIA分析脚本/脚本运行说明.md)
- [Technical Reference](Python脚本/SPAADIA分析脚本/TECHNICAL_REFERENCE.md)

### 🤝 Contributing

Issues and pull requests are welcome. Please ensure:
- Code follows existing style conventions
- New features include appropriate tests
- Documentation is updated accordingly

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### 📚 Citation

If you use this code in your research, please cite:
```bibtex
@software{spaadia_analysis_2025,
  title = {A Multilevel Logistic Regression Analysis Framework for SPAADIA Corpus},
  author = {xxx},
  year = {2025},
  url = {https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis}
}
```

---

<a name="chinese"></a>
## 🌟 中文版本

### 项目概述

本仓库包含SPAADIA（Speech Act Annotated Dialogues Incorporating Alternatives）语料库的综合统计分析框架，实现多层统计模型以验证服务对话中框架激活、策略选择和意义生成机制的四个核心假设。

### 🆕 最新更新 (2025-08-15)

#### 补充分析模块
为确保统计分析的完整性和严谨性，新增了四个补充分析模块：

1. **统计功效分析** (`power_analysis.py`)
   - 蒙特卡洛模拟评估统计功效
   - 确保样本量足以检测中等效应（Cohen's d = 0.5）

2. **FDR多重比较校正** (`fdr_correction.py`)
   - Benjamini-Hochberg方法控制错误发现率
   - 区分关键理论检验和探索性分析

3. **敏感性分析** (`sensitivity_analysis.py`)
   - 三维度评估结果稳健性
   - 语义距离方法、阈值参数、模型结构

4. **综合运行脚本** (`run_supplementary_analyses.py`)
   - 一键运行所有补充分析
   - 生成综合报告

### 🚀 快速开始

#### 安装
```bash
# 克隆仓库
git clone https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis.git
cd A-Multilevel-Logistic-Regression-Analysis

# 安装依赖
pip install -r requirements.txt
```

#### 运行完整分析
```bash
# 进入脚本目录
cd Python脚本/SPAADIA分析脚本

# 运行主分析（基础+高级）
python run_all_analyses_advanced.py

# 运行补充分析（新增）
python run_supplementary_analyses.py

# 可选：R语言验证
Rscript comprehensive_validation.R
```

### 📊 核心功能

- **四个假设检验**：
  - H1：框架激活的双重机制
  - H2：框架驱动的策略选择
  - H3：策略演化的路径依赖
  - H4：意义协商的语义收敛

- **高级统计方法**：
  - 三层线性混合模型
  - 带聚类稳健标准误的多项逻辑回归
  - 马尔可夫链分析
  - 分段增长曲线模型

- **严格的统计控制**：
  - 统计功效分析
  - 多重比较的FDR校正
  - 全面的敏感性分析
  - R语言交叉验证

### 📁 仓库结构

```
├── Python脚本/
│   └── SPAADIA分析脚本/
│       ├── 核心分析/
│       │   ├── hypothesis_h1_analysis.py
│       │   ├── hypothesis_h2_analysis.py
│       │   ├── hypothesis_h3_analysis.py
│       │   └── hypothesis_h4_analysis.py
│       ├── 高级分析/
│       │   ├── hypothesis_h1_advanced.py
│       │   ├── hypothesis_h2_advanced.py
│       │   ├── hypothesis_h3_advanced.py
│       │   └── hypothesis_h4_advanced.py
│       ├── 补充分析（新增）/
│       │   ├── power_analysis.py
│       │   ├── fdr_correction.py
│       │   ├── sensitivity_analysis.py
│       │   └── run_supplementary_analyses.py
│       └── 工具模块/
│           ├── data_loader_enhanced.py
│           └── advanced_statistics.py
├── 背景资料/
│   └── SPAADIA分析上下文/
│       └── 2.4小节_cn_updated.md
└── 输出/
    ├── data/      # JSON统计结果
    ├── tables/    # CSV数据表格
    ├── figures/   # 高分辨率图表（1200 DPI）
    └── reports/   # Markdown报告
```

### 🔬 技术实现

- **编程语言**：Python 3.9+，R 4.2+（用于验证）
- **主要依赖库**：
  - Python: pandas, numpy, statsmodels, scikit-learn, lifelines
  - R: lme4, pbkrtest, markovchain, nnet
- **输出格式**：中英文双语结果

### 📈 关键结果

- 当前样本量（118个对话）下主效应的统计功效 > 80%
- 所有假设检验的p值经过FDR校正
- 不同方法选择下结果稳健
- Python和R交叉验证结果一致

### 📝 文档说明

- [统计建模方法](背景资料/SPAADIA分析上下文/2.4小节_cn_updated.md)
- [脚本运行说明](Python脚本/SPAADIA分析脚本/脚本运行说明.md)
- [技术参考文档](Python脚本/SPAADIA分析脚本/TECHNICAL_REFERENCE.md)

### 🤝 贡献指南

欢迎提交Issue和Pull Request。请确保：
- 代码遵循现有的风格规范
- 新功能包含适当的测试
- 相应更新文档

### 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件

### 📚 引用方式

如果您在研究中使用了本代码，请引用：
```bibtex
@software{spaadia_analysis_2025,
  title = {SPAADIA语料库多层逻辑回归分析框架},
  author = {xxx},
  year = {2025},
  url = {https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis}
}
```

### 🌟 主要特色

1. **完整的统计分析流程**：从数据预处理到最终报告生成
2. **双重验证机制**：Python主分析 + R语言验证
3. **补充分析模块**：功效分析、FDR校正、敏感性分析
4. **中英文双语输出**：适合国际发表
5. **高质量可视化**：1200 DPI出版级图表
6. **完全可重现**：固定随机种子，详细日志

### 🔄 更新历史

- **2025-08-15**：添加补充分析模块，修正文档不一致
- **2025-08-02**：完成R验证系统
- **2025-08-01**：修复H2/H4分析错误
- **2025-01-31**：H1高级分析V2版本重写

### 💡 使用建议

1. **首次使用**：先运行`run_all_analyses_advanced.py`确保环境正确
2. **完整分析**：依次运行主分析和补充分析
3. **论文发表**：使用FDR校正后的p值报告结果
4. **方法部分**：参考2.4小节的详细方法描述

### 🛠️ 故障排除

如遇到问题，请检查：
1. Python版本是否≥3.9
2. 所有依赖包是否正确安装
3. 数据文件路径是否正确
4. 查看`输出/logs/`中的日志文件

### 📮 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [提交Issue](https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis/issues)

---

**注**：本项目持续更新中，欢迎关注和Star！⭐