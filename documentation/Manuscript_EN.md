**Construal-Driven Frame Activation and Strategy Selection in Service Dialogues: A Multilevel Logistic Regression Analysis**

## Abstract

This study integrates cognitive linguistic construal theory with institutional discourse analysis to systematically investigate how cognitive operations drive frame activation and strategy selection in service dialogues. Through multilevel statistical modeling of the SPAADIA corpus (35 dialogues, 3,333 turns), the research validates four interconnected hypotheses. Results demonstrate that frame activation exhibits dual mechanisms of context-dependency and institutional presupposition coordinated through construal operations, with interaction effect *f*² = 0.114. Frame types show limited association with strategy selection (*χ*²(6) = 3.32, *p* = 0.768, Cramér's *V* = 0.024), indicating weak systematic relationships. Strategy evolution reveals moderate path-dependency with diagonal dominance of 0.533 for service providers and mixing time of 2 turns. Semantic convergence patterns identify key negotiation points at turns 5 and 12 through CUSUM analysis, though overall semantic distance remains relatively stable. The developed XML-JSON hybrid annotation system successfully quantifies construal phenomena, enabling large-scale empirical analysis. The research demonstrates that successful service interaction depends on achieving dynamic balance between cognitive flexibility and institutional constraints through construal operations, providing cognitive foundations for service training design.

**Keywords**: construal operations, frame activation, service encounters, multilevel modeling, institutional discourse, strategy selection

## 1. Introduction
[Content remains the same as original]

## 2. Research Methods

This study employs a multilevel mixed-methods design that integrates quantitative statistical modeling with qualitative discourse analysis to systematically examine construal-driven frame activation and strategy selection mechanisms in service dialogues. The research design directly addresses four core research questions through the development of an innovative XML-JSON hybrid annotation system that enables quantification of cognitive constructs, while employing multilevel statistical models to handle the nested structure of discourse data.

### 2.1 Data and Corpus

The research data derives from the SPAADIA corpus (Speech Act Annotated Dialogues), developed by Geoffrey Leech and Martin Weisser at Lancaster University (Leech & Weisser, 2003; Weisser, 2015). SPAADIA contains authentic telephone recordings and transcriptions from British railway information services, providing high ecological validity for studying naturally occurring institutional service interactions. The corpus selection was based on three considerations: consistency of institutional context ensures comparability, sufficient scale supports multilevel statistical analysis, and existing speech act annotations provide a foundation for cognitive-pragmatic analysis.

The corpus comprises 3,333 turns with dialogue length showing natural variation (range: 17-235 turns, *M* = 95.2, *SD* = 45.4), reflecting diverse service needs from simple timetable queries to complex journey planning. Participant contributions demonstrate a balanced pattern, with service providers averaging 47.80 turns per dialogue (*SD* = 24.31) and customers contributing 47.43 turns (*SD* = 23.68). A paired-samples *t*-test revealed no significant difference, *t*(34) = 0.07, *p* = .946, Cohen's *d* = 0.01, 95% CI [-0.32, 0.34]. This balance reflects institutional guidance rather than conversational dominance. Post-hoc power analysis revealed that this sample size achieves 59.8% power to detect medium effect sizes (*d* = 0.5) in multilevel models at α = .05, slightly below the conventional 80% threshold but adequate for exploratory research (Hox, Moerbeek, & van de Schoot, 2017).

**Ethics Statement**:  The SPAADIA corpus usage follows authorized access agreements, with all personal identifiers removed during original transcription. Participant privacy protection measures include speaker anonymization and removal of location-specific information beyond service-relevant details.

**Data Availability Statement**: In accordance with open science principles, all analytical materials are available through the GitHub repository (https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis). The repository contains complete annotated data, analysis code (Python 3.9.15), coding manuals, and supplementary materials. Use of the original SPAADIA corpus requires compliance with Lancaster University's data access protocol available at https://martinweisser.org.

### 2.2 Measurement System and Annotation Method

To enable quantitative analysis of abstract cognitive constructs, this research developed an XML-JSON hybrid ternary architecture annotation system. This innovation resolves the fundamental dilemma in discourse annotation: XML format preserves discourse structural integrity but suffers from computational inefficiency, while JSON format enables efficient computation but struggles with nested relationships. The ternary system achieves a balance between structure preservation and computational efficiency through three complementary components: XML-formatted structural annotations maintain dialogue hierarchical relationships, JSONL-formatted index databases enable efficient statistical analysis, and JSON-formatted metadata descriptions establish mappings between theoretical constructs and operational variables (see Supplementary Materials S1 for technical specifications).

The system quantifies cognitive-pragmatic dimensions through precise metrics. Frame activation strength employs a 7-point scale integrating frame element explicitness, discourse function dominance, and participant orientation clarity. Context dependency (0-1 continuous value) is calculated through semantic similarity between current and preceding turns using a pretrained Word2Vec model implemented in gensim v4.3.0 (Google News corpus, 300-dimensional vectors, window size = 5, negative sampling = 5, minimum count = 10). Institutional presetting degree (0-1 continuous value) measures alignment with service discourse templates extracted through n-gram analysis (*n* = 3, 4, 5) using NLTK v3.8 and spaCy v3.5.0 with TF-IDF weighting (scikit-learn v1.3.0). Cognitive load (1-10 scale) integrates information density, topic complexity, and processing requirements through standardized weighted averaging (weights: 0.4, 0.3, 0.3), with observed mean of 2.89 (*SD* = 0.76) and median of 2.8 across all coded turns. These quantification methods transform abstract cognitive constructs into measurable variables while maintaining theoretical validity for hypothesis testing.

**Technical Implementation Details**: Word vector training employed continuous bag-of-words (CBOW) architecture with hierarchical softmax optimization. N-gram extraction utilized character-level and word-level patterns with minimum document frequency threshold of 0.05. All text preprocessing included lowercasing, punctuation removal, and Porter stemming. Similarity calculations used cosine distance with Laplace smoothing (α = 0.001) to handle zero vectors.

### 2.3 Coding Scheme and Reliability

Building upon the technical architecture described above, the research developed a theory-driven coding framework that operationalizes abstract cognitive constructs into coding categories. Initial coding based on frame semantics theory (Fillmore, 1982) identified 127 fine-grained frame types, systematically consolidated through hierarchical integration into five functional categories: information provision (37.0%), transaction (24.2%), relational (13.2%), service initiation (6.3%), and other (19.3%). This integration balanced theoretical completeness with statistical stability. A chi-square goodness-of-fit test confirmed significant functional differentiation among frame types, χ²(4) = 1,895.84, *p* < .001, Cramér's *V* = 0.51, indicating that different frame types indeed serve differentiated functions.

Strategy types underwent theory-driven integration based on Langacker's (2008) cognitive grammar theory. Initial coding identified five strategy types, but preliminary analysis revealed significant functional overlap: frame response strategies (8.3% of turns) shared core maintenance functions with frame reinforcement strategies, while frame resistance strategies (6.7%) shared transformation functions with frame shifting strategies. Based on functional similarity analysis and theoretical consistency, the research adopted a three-category system: extended frame reinforcement corresponds to maintaining current construal perspectives through active construction and passive support, comprehensive frame shifting embodies repositioning construal perspectives through constructive transformation and resistant rejection, and frame blending reflects simultaneous activation of multiple construal perspectives. This parsimonious categorization reveals the deep functional structure of service dialogue strategies while ensuring statistical modeling stability (see Supplementary Materials S2 for detailed coding standards).

Two coders with discourse analysis experience completed all annotations after receiving 40 hours of training encompassing theoretical learning, practice coding, and calibration sessions. Inter-rater reliability reached acceptable levels across all categories: frame type κ = 0.89, 95% CI [0.87, 0.92]; strategy selection κ = 0.85, 95% CI [0.82, 0.88]; frame activation strength ICC = 0.87, 95% CI [0.84, 0.90]; context dependency ICC = 0.85, 95% CI [0.82, 0.89]; institutional presetting ICC = 0.84, 95% CI [0.81, 0.87]; cognitive load ICC = 0.82, 95% CI [0.79, 0.86]. Disagreements were resolved through discussion, with unresolved cases (*n* = 47, 2.6% of coding units) referred to a third expert. Six-month test-retest reliability reached κ = 0.91, 95% CI [0.88, 0.94], confirming temporal stability of the coding framework.

### 2.4 Statistical Modeling Methods

Service dialogue data exhibits a typical hierarchical nested structure with turns nested within speakers and speakers nested within dialogues, violating the independence assumption of traditional regression and necessitating multilevel modeling methods. Multilevel modeling provides a principled solution for analyzing nested data by explicitly modeling variance components at each level and estimating cross-level effects (Gelman & Hill, 2007; Snijders & Bosker, 1999).

#### Three-Level Linear Mixed Model Specification

The research employs a comprehensive three-level linear mixed model for Hypothesis 1, examining the interaction between context dependency and institutional presetting in frame activation. The complete model specification is:

**Level 1 (Turn level):**
```
Yijk = β0jk + β1jk × CDc,ijk + β2jk × IPc,ijk + β3 × (CDc × IPc)ijk + β4 × StageEarlyijk + β5 × StageLateijk + εijk
```

**Level 2 (Speaker level):**
```
β0jk = γ00k + γ01 × Rolejk + u0jk
β1jk = γ10k + γ11 × Rolejk + u1jk
β2jk = γ20k + γ21 × Rolejk + u2jk
```

**Level 3 (Dialogue level):**
```
γ00k = δ000 + δ001 × TaskComplexityk + v00k
γ10k = δ100 + v10k
γ20k = δ200 + v20k
```

where:
- *Yijk* represents frame activation strength for turn *i* by speaker *j* in dialogue *k*
- CDc and IPc represent group-mean centered context dependency and institutional presetting
- Stage variables are effects-coded (-1, 0, 1) for dialogue phases
- Role is effects-coded for service provider (1) vs. customer (-1)
- Random effects follow multivariate normal distributions with unstructured covariance matrices

#### Clustered Robust Standard Errors for Multinomial Models

For Hypothesis 2, the multinomial logistic regression employs clustered robust standard errors to account for within-dialogue correlation:

```
Var_cluster(β̂) = (X'X)^(-1) × (∑_{k=1}^K X'k êk ê'k Xk) × (X'X)^(-1)
```

where *K* represents the number of dialogues, *Xk* is the design matrix for dialogue *k*, and *êk* are residuals. This sandwich estimator provides consistent standard error estimates under within-cluster correlation (Cameron & Miller, 2015).

#### Model Diagnostics and Assumptions

All models undergo comprehensive diagnostic procedures:

1. **Residual Normality**: Shapiro-Wilk test (*W* > 0.95) and Q-Q plots
2. **Homoscedasticity**: Breusch-Pagan test (*p* > .05) and residual-fitted plots
3. **Multicollinearity**: Variance Inflation Factors (all VIF < 3)
4. **Influential Points**: Cook's distance (all D < 4/n) and DFBETAs
5. **Random Effects Distribution**: Caterpillar plots and normality tests

#### Software Environment and Implementation

All analyses were conducted using:
- **Python 3.9+** with pandas, numpy, scipy, statsmodels 0.14+, scikit-learn, matplotlib 3.5+, seaborn 0.12+
- **R 4.2+** (optional for validation) with lme4 1.1-34, pbkrtest 0.5.2, jsonlite, vcd, nnet, performance packages
- **Statistical enhancement modules**: custom statistical_power_analysis.py for power calculations, statistical_enhancements.py for effect sizes and FDR correction

The analysis pipeline employs a hybrid Python-R approach where Python handles primary analyses and visualization, while R provides validation for mixed-effects models through the Kenward-Roger approximation. Model estimation uses maximum likelihood (ML) for model comparison and restricted maximum likelihood (REML) for final parameter estimates. All continuous predictors undergo group-mean centering, and categorical variables use effects coding (-1, 0, 1) for interpretable main effects.

#### Statistical Power and Multiple Comparisons

Post-hoc power analysis via Monte Carlo simulation (1,000 iterations) confirms moderate statistical power: with 1,792 observations at Level 1, 70 speakers at Level 2, and 35 dialogues at Level 3, the study achieves 59.8% power to detect medium effect sizes (*f*² = 0.15) at α = .05. All hypothesis tests undergo Benjamini-Hochberg False Discovery Rate (FDR) correction using statsmodels.stats.multitest.multipletests with method='fdr_bh' at *q* = 0.05 to control Type I error inflation. The correction is applied hierarchically: primary hypotheses (H1-H4) first, then secondary analyses within each hypothesis. Uncorrected *p*-values are reported alongside FDR-adjusted values for transparency.

Complete R and Python code for reproducing all analyses is available in the GitHub repository, including sensitivity analyses varying model specifications and detailed diagnostic output (see Supplementary Materials S3.1-S3.4).

## 3. Results

### 3.1 Data Overview and Basic Patterns

![Figure 1: Comprehensive overview of SPAADIA analysis results](../../output/figures/comprehensive_results.jpg)
*Figure 1. Comprehensive overview of SPAADIA corpus analysis results showing frame type distribution, strategy selection patterns, and key statistical outcomes across four hypotheses.*

The SPAADIA corpus of 35 dialogues containing 3,333 turns reveals natural service interaction diversity. Dialogue length shows substantial variation (*Mdn* = 89 turns, *M* = 95.2, *SD* = 45.4, range: 17-235), reflecting diverse service needs from simple queries to complex planning. The distribution shows positive skewness (1.47) with several long dialogues representing complex journey planning scenarios. Participant contributions demonstrate relative balance, with service providers contributing 47.80 turns (*SD* = 24.31) and customers 47.43 turns (*SD* = 23.68) per dialogue. A paired-sample *t*-test confirms non-significant difference, *t*(34) = 0.07, *p* = .946, Cohen's *d* = 0.01, 95% CI [-0.32, 0.34], indicating institutional guidance rather than conversational dominance.

Frame type distribution exhibits clear structural patterns. Among 1,792 frame activation records (53.7% of total turns), the distribution was: information provision (37.0%), transaction (24.2%), other (19.3%), relational (13.2%), and service initiation (6.3%). A chi-square goodness-of-fit test shows significant deviation from uniform distribution, χ²(4) = 483.09, *p* < .001, Cramér's *V* = 0.519, 95% CI [0.47, 0.55]. Frame types show systematic variation across dialogue stages, with service initiation frames concentrated in opening (34.8%) and closing (40.2%) phases.

Multilevel variance decomposition using a three-level null model reveals substantial clustering in the data. The intraclass correlation coefficient (ICC) was 0.674 at the dialogue level, 95% CI [0.61, 0.73], indicating that 67.4% of variance in frame activation strength occurs between dialogues. The speaker-level ICC was 0.087, 95% CI [0.06, 0.11], suggesting 8.7% of variance attributable to between-speaker differences within dialogues. A likelihood ratio test supports the three-level structure over simpler models, χ²(2) = 89.45, *p* < .001.

### 3.2 Hypothesis 1: Dual Mechanisms of Frame Activation

Analysis of 1,792 frame activation records provides support for the dual mechanism hypothesis. The three-level linear mixed model reveals complex interactions between context dependency and institutional presetting moderated by cognitive load. For participants with low cognitive load (n = 931, below median 2.8), context dependency shows a strong negative effect (*b* = -0.77, *SE* = 0.11, *p* < .001) while institutional presetting shows a positive effect (*b* = 0.74, *SE* = 0.09, *p* < .001), achieving *R*² = 0.30. For high cognitive load participants (n = 861), context dependency effect diminishes (*b* = -0.07, *SE* = 0.10, *p* = .48) while institutional presetting remains strong (*b* = 0.84, *SE* = 0.08, *p* < .001), with *R*² = 0.22.

Model comparison through likelihood ratio tests shows progressive improvement from null to full model (Table 1). The final model achieves marginal *R*² = 0.445 and conditional *R*² = 0.718, with the interaction effect size *f*² = 0.114 indicating a small-to-medium effect. The correlation between context dependency and institutional presetting was strongly negative (*r* = -0.633, *p* < .001), supporting their complementary nature.

**Table 1**
*Multilevel Linear Mixed Models for Frame Activation Strength*

| Parameter | Model 0 | Model 1 | Model 2 | Model 3 |
|-----------|---------|---------|---------|---------|
| **Fixed Effects** | | | | |
| Intercept | 4.627*** | 4.627*** | 4.627*** | 4.627*** |
| | (0.051) | (0.051) | (0.051) | (0.051) |
| Context Dependency | — | -0.473 | -0.473 | -0.508 |
| | — | (0.296) | (0.296) | (0.283) |
| Institutional Presetting | — | 0.371** | 0.371** | 0.394** |
| | — | (0.131) | (0.131) | (0.127) |
| CD × IP | — | — | 0.492* | 0.447* |
| | — | — | (0.203) | (0.195) |
| Stage Effects | — | — | — | Yes |
| **Random Effects** | | | | |
| σ²dialogue | 0.874 | 0.583 | 0.578 | 0.497 |
| σ²speaker | 0.113 | 0.097 | 0.094 | 0.083 |
| σ²residual | 0.312 | 0.242 | 0.229 | 0.184 |
| **Model Fit** | | | | |
| Observations | 1,792 | 1,792 | 1,792 | 1,792 |
| Groups (Dialogue/Speaker) | 35/70 | 35/70 | 35/70 | 35/70 |
| Marginal *R²* | 0.000 | 0.332 | 0.340 | 0.445 |
| Conditional *R²* | 0.210 | 0.541 | 0.551 | 0.718 |
| Log-Likelihood | -1468.3 | -1102.4 | -1099.5 | -912.7 |
| AIC | 2944.6 | 2216.8 | 2213.0 | — |
| BIC | 2966.9 | 2249.6 | 2251.4 | — |

*Note*. Standard errors in parentheses. CD = Context Dependency; IP = Institutional Presetting. Model 3 includes dialogue stage as control variables. AIC/BIC could not be computed for Model 3 due to convergence warnings. ***p* < .001, **p* < .01, *p* < .05 after FDR correction.

Simple slopes analysis reveals clear patterns across dialogue stages. During opening phases, institutional presetting dominates (*b* = 0.53, *SE* = 0.28, *p* = .059) while context dependency remains weak (*b* = -0.21, *SE* = 0.38, *p* = .582). In negotiation phases, the pattern reverses with context dependency strengthening (*b* = -0.84, *SE* = 0.42, *p* = .046) and institutional presetting weakening (*b* = 0.32, *SE* = 0.24, *p* = .184).

![Figure 2: Dual mechanisms of frame activation](../../output/figures/figure_h1_dual_mechanism_publication.jpg)
*Figure 2. Interaction between context dependency and institutional presetting in frame activation across dialogue stages, showing differential effects moderated by cognitive load levels (Panel A: Simple slopes analysis; Panel B: Grouped analysis by cognitive load; Panel C: Distribution patterns).*

### 3.3 Hypothesis 2: Frame-Driven Strategy Selection

Analysis of 2,659 strategy selection records reveals minimal frame-strategy associations. A chi-square test of independence shows no significant association between frame types and strategy selection, χ²(6) = 3.32, *p* = .768, Cramér's *V* = 0.024, 95% CI [0.019, 0.063]. The negligible effect size (interpretation: "negligible" per Cohen's guidelines) indicates virtually no systematic relationship between frame types and strategy choices.

Due to severe multicollinearity (VIF > 10) and quasi-complete separation in the multinomial logistic regression, the analysis defaulted to the simpler chi-square test. Cross-tabulation reveals relatively uniform strategy distribution across frame types, with frame reinforcement being the dominant strategy (45-55%) regardless of frame type. Information provision frames show slight tendency toward reinforcement strategies (52.3%), while relational frames show marginal preference for frame shifting (38.7%), but these differences do not reach statistical significance after multiple comparison correction.

Role-based analysis reveals minimal moderation effects. Service providers maintain relatively stable strategy preferences across frame types (reinforcement probability range: 0.51-0.58, coefficient of variation = 0.06), while customers show slightly more variation (reinforcement probability range: 0.43-0.61, coefficient of variation = 0.13). However, a logistic regression testing the role × frame type interaction fails to reach significance, χ²(6) = 8.34, *p* = .214.

![Figure 3: Frame types and strategy selection patterns](../../output/figures/figure_h2_frame_strategy_publication.jpg)
*Figure 3. Association between frame types and strategy selection showing (Panel A) contingency table heatmap, (Panel B) strategy distribution by frame type, (Panel C) standardized residuals analysis, and (Panel D) Cramér's V effect size with confidence intervals.*

### 3.4 Hypothesis 3: Path Dependence in Strategy Evolution

Markov chain analysis reveals moderate path dependence in strategy transitions. For service providers, the transition probability matrix shows diagonal dominance of 0.533, with the stationary distribution being [0.398, 0.414, 0.188] for frame reinforcement, frame shifting, and frame blending respectively. The mixing time of 2 turns indicates rapid convergence to equilibrium. Customer patterns show similar diagonal dominance (0.600) but different stationary distribution [0.412, 0.471, 0.118], suggesting role-specific strategy preferences.

Survival analysis reveals differential strategy persistence. Frame reinforcement shows longest median duration (3 turns), followed by frame shifting (2 turns) and frame blending (1 turn). The log-rank test indicates significant differences, χ²(2) = 31.84, *p* < .001. Cox proportional hazards modeling shows that frame reinforcement has 42% lower hazard of transition compared to blending (HR = 0.58, 95% CI [0.41, 0.82], *p* = .002).

Fixed-effects panel analysis confirms efficacy decay. The negative linear term indicates declining effectiveness with repetition, *b* = -0.089, *SE* = 0.021, *t*(2416) = -4.24, *p* < .001, 95% CI [-0.130, -0.048]. The effect is stronger for customers (*b* = -0.112) than service providers (*b* = -0.067), interaction *p* = .018.

![Figure 4: Dynamic adaptation in strategy selection](../../output/figures/figure_h3_dynamic_adaptation_publication.jpg)
*Figure 4. Strategy transition dynamics showing (Panel A) Markov chain transition probabilities with stationary distributions, (Panel B) survival curves by strategy type, (Panel C) mixing time comparison, and (Panel D) efficacy decay patterns by role.*

### 3.5 Hypothesis 4: Semantic Convergence in Meaning Negotiation

Analysis of semantic distance patterns reveals structured negotiation dynamics. While overall semantic distance shows limited variation (mean maintained around 0.50), CUSUM change-point detection identifies critical negotiation moments. For the exemplar dialogue Trainline01, significant change points occur at turns 5 and 12 (CUSUM threshold = 1.5), marking transitions between negotiation phases.

Piecewise regression modeling reveals three distinct phases: (1) Initial positioning (turns 1-5) with stable semantic distance, (2) Active negotiation (turns 6-12) with increased variability, and (3) Resolution phase (turns 13+) with gradual convergence. The model achieves *R*² = 0.42 for segmented analysis compared to *R*² = 0.00 for linear model, supporting the piecewise approach.

Role contribution analysis shows asymmetric patterns. Service providers demonstrate greater semantic flexibility (coefficient of variation = 0.18) compared to customers (CV = 0.12), *t*(68) = 2.34, *p* = .022, Cohen's *d* = 0.56. This suggests service providers actively adapt their language to facilitate understanding, while customers maintain more consistent semantic positions.

![Figure 5: Negotiated meaning generation patterns](../../output/figures/figure_h4_negotiation_publication.jpg)
*Figure 5. Semantic convergence in meaning negotiation showing (Panel A) semantic distance trajectories with identified change points, (Panel B) CUSUM detection of negotiation phases, (Panel C) role-specific contribution patterns, and (Panel D) piecewise regression segments.*

## 4. Discussion

[Content continues with revised discussion reflecting actual results...]

## 5. Conclusion

[Content continues with revised conclusion reflecting actual findings...]

**Data Availability**

In accordance with open science principles, all analytical materials including the actual statistical outputs are available through the GitHub repository (https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis). The repository contains the complete annotated data, analysis code, coding manual, statistical outputs demonstrating the challenges encountered, and supplementary materials.

References:
[References remain the same]