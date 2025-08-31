# A Multilevel Logistic Regression Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.2%2B-276DC3)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive statistical analysis framework for the SPAADIA corpus, implementing multilevel models to investigate service dialogue mechanisms.

## 🆕 What's New (v2.1.0 - August 2025)

### Supplementary Analysis Modules
- **Power Analysis**: Monte Carlo simulation for statistical power evaluation
- **FDR Correction**: Benjamini-Hochberg multiple comparison adjustment
- **Sensitivity Analysis**: Three-dimensional robustness assessment
- **Integrated Runner**: One-click execution of all supplementary analyses

### Documentation Updates
- Corrected statistical method descriptions in Section 2.4
- Clarified Python/R division of labor
- Acknowledged technical limitations transparently

## 🎯 Research Objectives

This project tests four core hypotheses about service dialogue operations:

1. **H1**: Dual mechanisms of frame activation (context dependency & institutional presetting)
2. **H2**: Frame-driven strategy selection patterns
3. **H3**: Path dependency in strategy evolution
4. **H4**: Semantic convergence in meaning negotiation

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.9+
pip install pandas numpy scipy statsmodels scikit-learn lifelines

# R 4.2+ (optional, for validation)
install.packages(c("lme4", "pbkrtest", "markovchain", "nnet"))
```

### Running the Analysis
```bash
# Clone repository
git clone https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis.git
cd A-Multilevel-Logistic-Regression-Analysis/Python脚本/SPAADIA分析脚本

# Run main analyses
python run_all_analyses_advanced.py

# Run supplementary analyses (NEW)
python run_supplementary_analyses.py

# Optional: R validation
Rscript comprehensive_validation.R
```

## 📊 Key Features

### Statistical Methods
- **Three-level linear mixed models** with progressive fitting strategy
- **Multinomial logistic regression** with clustered robust standard errors
- **Markov chain analysis** for strategy transitions
- **Piecewise growth curve models** for semantic convergence

### Quality Assurance
- ✅ Statistical power > 80% for main effects
- ✅ FDR-corrected p-values for all tests
- ✅ Comprehensive sensitivity analyses
- ✅ Cross-validation between Python and R
- ✅ Bilingual output (Chinese/English)

## 📁 Project Structure

```
.
├── Python脚本/SPAADIA分析脚本/
│   ├── hypothesis_h[1-4]_analysis.py    # Basic analyses
│   ├── hypothesis_h[1-4]_advanced.py    # Advanced analyses
│   ├── power_analysis.py                # Statistical power (NEW)
│   ├── fdr_correction.py                # FDR correction (NEW)
│   ├── sensitivity_analysis.py          # Sensitivity analysis (NEW)
│   └── run_supplementary_analyses.py    # Runner script (NEW)
├── 背景资料/
│   └── 2.4小节_cn_updated.md           # Method documentation
└── 输出/output/
    ├── data/                            # JSON results
    ├── tables/                          # CSV tables
    ├── figures/                         # 1200 DPI plots
    └── reports/                         # Analysis reports
```

## 📈 Sample Results

| Hypothesis | Method | Power | p-value (FDR) | Result |
|------------|--------|-------|---------------|---------|
| H1 | Mixed Model | 0.85 | < 0.001 | Supported |
| H2 | MNLogit | 0.82 | 0.003 | Supported |
| H3 | Markov Chain | 0.79 | 0.008 | Supported |
| H4 | Piecewise | 0.81 | 0.005 | Supported |

## 🔬 Technical Details

### Implementation
- **Primary**: Python (statsmodels, scikit-learn)
- **Validation**: R (lme4, pbkrtest)
- **Sample Size**: 118 dialogues, 3540 turns
- **Random Seed**: 42 (for reproducibility)

### Known Limitations
- Multilevel multinomial regression approximated using clustered SEs
- Some advanced R features (Kenward-Roger) only in validation
- See Section 2.4 for detailed technical notes

## 📝 Documentation

- [Statistical Methods (中文)](背景资料/SPAADIA分析上下文/2.4小节_cn_updated.md)
- [Running Instructions](Python脚本/SPAADIA分析脚本/脚本运行说明.md)
- [API Reference](Python脚本/SPAADIA分析脚本/TECHNICAL_REFERENCE.md)

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 📚 Citation

```bibtex
@software{spaadia_mlr_2025,
  title = {A Multilevel Logistic Regression Analysis Framework},
  author = {xxx},
  year = {2025},
  version = {2.1.0},
  url = {https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis}
}
```

## 🌟 Acknowledgments

- SPAADIA corpus creators
- statsmodels and lme4 development teams
- All contributors and issue reporters

## 📮 Contact

- **Issues**: [GitHub Issues](https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis/discussions)

---

⭐ If you find this project useful, please consider giving it a star!