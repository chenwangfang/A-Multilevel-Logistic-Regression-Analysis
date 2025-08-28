# Supplementary Materials: Construal-Driven Frame Activation and Strategy Selection in Service Dialogues

## Table of Contents

**S1. Technical Specifications of the XML-JSON Hybrid Ternary Architecture Annotation System**
- S1.1 System Architecture Design
- S1.2 Data Structure Definitions
- S1.3 Implementation Code Examples

**S2. Coding Manual and Theory-Driven Category Integration**
- S2.1 Frame Type Coding System and Integration Rationale
- S2.2 Strategy Type Coding Standards and Integration Justification
- S2.3 Cognitive-Pragmatic Dimension Quantification Metrics
- S2.4 Coder Training Procedures and Reliability Testing

**S3. Statistical Analysis Technical Details**
- S3.1 Multilevel Model Construction Process
- S3.2 Model Diagnostic Procedures
- S3.3 Sensitivity Analysis Results
- S3.4 Statistical Power Analysis

**S4. Data Processing and Quality Control**
- S4.1 Data Preprocessing Pipeline
- S4.2 Missing Data Handling
- S4.3 Outlier Detection and Treatment
- S4.4 Detailed Reliability Test Results

**S5. Supplementary Tables and Figures**
- S5.1 Detailed Descriptive Statistics Tables
- S5.2 Complete Model Comparison Results
- S5.3 Diagnostic Plot Collection

---

## S1. Technical Specifications of the XML-JSON Hybrid Ternary Architecture Annotation System

### S1.1 System Architecture Design

The XML-JSON hybrid ternary architecture annotation system developed in this research resolves the dilemma between structural representation and computational efficiency in discourse analysis. While traditional XML annotation excellently represents the hierarchical structure and sequential relationships of discourse, its tree structure leads to inefficient traversal during statistical analysis. Pure JSON format supports rapid indexing and batch processing but encounters redundancy and ambiguity when representing nested discourse relationships. This system achieves complementary advantages through the principle of functional separation, allocating structure preservation, rapid retrieval, and metadata management to different components.

The system comprises three core components, each serving specific functions and enabling data exchange through standardized interfaces. First, the structured main annotation employs XML format to preserve the complete hierarchical relationships and sequential structure of dialogues. Second, the index database uses JSONL format, with one JSON object per line, supporting stream processing and efficient statistical analysis. Third, the metadata description adopts standard JSON format, providing dialogue-level statistical information and coding system definitions.

This ternary architecture design operates on the following principles. First, the separation of data representation and data processing enables each component to adopt the format most suitable for its function. Second, the indexed design avoids repeated parsing during analysis through pre-computation of key metrics. Third, centralized management of metadata ensures standardization for cross-dialogue comparison. Through this design, the system simultaneously satisfies the detailed requirements of qualitative analysis and the efficiency demands of quantitative analysis.

### S1.2 Data Structure Definitions

**XML Structured Main Annotation Specifications**

The XML files adopt a strict hierarchical structure, progressively refined from the dialogue level to the utterance level. Each dialogue contains a metadata section and a turn sequence. The metadata section records basic dialogue information, participant roles, and dialogue statistics. The turn sequence is organized chronologically, with each turn containing utterance groups, and each utterance group containing one or more utterances. This structure ensures complete preservation of inter-utterance relationships.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<dialogue id="[dialogue_id]" duration="[seconds]" outcome="[successful/failed]">
  <metadata>
    <participants>
      <participant id="A" role="service_provider"/>
      <participant id="B" role="customer"/>
    </participants>
    <dialogue_statistics>
      <turn_count>[number]</turn_count>
      <utterance_count>[number]</utterance_count>
      <frame_type_count>[number]</frame_type_count>
      <strategy_type_count>[number]</strategy_type_count>
    </dialogue_statistics>
  </metadata>
  
  <turn id="[turn_id]" speaker="[A/B]" relative_position="[0.0-1.0]" 
        stage="[opening/information_exchange/negotiation/closing]">
    <utterance_group frame_activation="[frame_type]" 
                     frame_activation_strength="[1-7]"
                     institutional_constraint="[low/medium/high]"
                     meta='[json_metadata]'>
      <utterance id="[utterance_id]" sp_act="[speech_act]">
        <content>[transcribed_text]</content>
        <strategy type="[strategy_type]" efficacy="[1-7]"/>
      </utterance>
    </utterance_group>
  </turn>
</dialogue>
```

**JSONL Index Database Format**

The index database employs JSONL format, with each line containing a complete JSON object, supporting stream processing and incremental updates. Each record contains unique identifiers, hierarchical association information, cognitive-pragmatic indicators, and semantic features. This format particularly suits batch processing and statistical analysis of large-scale data.

```json
{
  "index_id": "trainline01_T001_U001",
  "dialogue_id": "trainline01",
  "turn_id": "T001",
  "speaker_role": "service_provider",
  "frame_type": "service_initiation",
  "frame_strength": 4.5,
  "context_dependency": 0.76,
  "institutional_presetting": 0.89,
  "strategy": "frame_reinforcement",
  "cognitive_load": 2.1,
  "relative_position": 0.01,
  "stage": "opening"
}
```

### S1.3 Implementation Code Examples

The following Python code demonstrates the core implementation of the data loader, illustrating how to integrate the three data components for analysis:

```python
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np

class DialogueDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.xml_files = list(self.data_dir.glob("*.xml"))
        self.jsonl_file = self.data_dir / "index.jsonl"
        self.metadata_file = self.data_dir / "metadata.json"
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
    
    def load_dialogue_structure(self, dialogue_id):
        xml_file = self.data_dir / f"{dialogue_id}.xml"
        tree = ET.parse(xml_file)
        return tree.getroot()
    
    def load_indexed_features(self, filters=None):
        records = []
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if self._apply_filters(record, filters):
                    records.append(record)
        return pd.DataFrame(records)
    
    def compute_dialogue_metrics(self, dialogue_id):
        structure = self.load_dialogue_structure(dialogue_id)
        features = self.load_indexed_features({'dialogue_id': dialogue_id})
        
        return {
            'dialogue_id': dialogue_id,
            'turn_count': len(structure.findall('.//turn')),
            'frame_diversity': features['frame_type'].nunique(),
            'mean_cognitive_load': features['cognitive_load'].mean(),
            'context_dependency_trend': np.polyfit(
                range(len(features)), 
                features['context_dependency'].values, 1)[0]
        }
```

---

## S2. Coding Manual and Theory-Driven Category Integration

### S2.1 Frame Type Coding System and Integration Rationale

The frame type coding in this research underwent a systematic process from initial fine-grained classification to theory-driven integration. Initial coding, based on frame semantics theory (Fillmore, 1982) and functional analysis of service interactions, identified 127 fine-grained frame types. However, pre-analysis revealed that this detailed classification faced sparsity issues in statistical modeling, with many frame types occurring too infrequently to support robust statistical inference.

Based on functional similarity and statistical considerations, the research adopted a hierarchical integration strategy, consolidating fine-grained frames into five functional categories. This integration followed three principles. First, maintaining functional distinctiveness ensured that merged categories serve different communicative functions in service interactions. Second, maintaining statistical stability guaranteed each category sufficient observations to support multilevel modeling. Third, preserving theoretical explanatory power ensured the integrated classification still reflects core mechanisms of frame activation.

**The Five Integrated Frame Categories**

1. **Information Provision Frames**  
This category integrates 37 fine-grained frames including journey_information, fare_information, and availability_check. The common characteristic of these frames is organizing and transmitting service-related factual information. Accounting for 37.0% of the corpus, this represents the most frequently activated frame type, reflecting the information exchange nature of service dialogues.

2. **Transaction Frames**  
This category integrates 28 fine-grained frames including booking, payment_processing, and confirmation. These frames handle core service transaction procedures, including reservation, payment, and confirmation processes. Accounting for 24.2%, this reflects the task-oriented characteristics of service interactions.

3. **Relational Frames**  
This category integrates 19 fine-grained frames including understanding, gratitude, and empathy. These frames manage the interpersonal dimension of interactions, maintaining politeness and emotional connection. While accounting for only 13.2%, they significantly impact service quality perception.

4. **Service Initiation Frames**  
This category integrates 15 fine-grained frames including service_opening, role_establishment, and service_closing. These frames mark boundaries of service interactions, establishing and terminating institutional contexts. Accounting for 6.3%, they primarily appear in dialogue opening and closing phases.

5. **Other Frames**  
This category includes 28 low-frequency frames difficult to classify into the above four categories. Retention of this category serves two purposes: while individual frames have low frequency, they collectively account for 19.3%; and they may represent special or innovative frame usage worthy of further exploration in future research.

This five-category system maintains theoretical integrity while ensuring statistical analysis feasibility. Chi-square goodness-of-fit testing shows frame distribution significantly deviates from uniform distribution (χ² = 1895.84, p < .001), indicating different frame types indeed serve differentiated functions in service dialogues.

### S2.2 Strategy Type Coding Standards and Integration Justification

Strategy type coding similarly underwent a process from theory-driven initial classification to data-based systematic integration. Initial coding based on the cognitive-pragmatic theoretical framework identified five strategy types: frame reinforcement, frame shifting, frame blending, frame response, and frame resistance. However, pre-analysis revealed significant functional overlap patterns, prompting adoption of a more parsimonious three-category system.

**Distribution and Functional Analysis of Initial Five Strategies**

Frame reinforcement strategies (52.3%) and frame response strategies (8.3%) showed high functional similarity, both aiming to maintain and support currently activated frames. Their primary distinction lay in degree of agency: frame reinforcement actively constructs and deepens frames, while frame response passively cooperates with and accepts frames. Similarly, frame shifting strategies (26.4%) and frame resistance strategies (6.7%) shared the core function of promoting frame change, with differences mainly in transformation intensity and explicitness. Frame blending strategies (6.3%), as a unique strategy type, possessed multi-frame integration functions that could not be subsumed under other categories.

**Theory-Driven Strategy Integration Scheme**

Based on functional similarity analysis and statistical modeling requirements, the research adopted the following integration scheme:

1. **Expanded Frame Reinforcement Strategy** (integrating original frame reinforcement + frame response, 60.6%)  
This encompasses the complete strategic spectrum from active construction to passive support. It contains two functional dimensions: the active reinforcement dimension (initialization, role establishment, directive guidance, information seeking, information confirmation, procedural announcement) and the response cooperation dimension (direct answering, agreement, acceptance, acknowledgment, preference expression, gratitude expression). This dual functionality makes it the most fundamental strategy type in service dialogues.

2. **Comprehensive Frame Shifting Strategy** (integrating original frame shifting + frame resistance, 33.1%)  
This includes diverse transformation means from mild adjustment to strong resistance. It manifests in two modes: constructive transformation (topic transition marking, framework expansion, constraint introduction, correction, alternative presentation, rule explanation) and resistant transformation (rejection, preference assertion, constraint declaration, alternative demand, justification). This integration reflects the essence of frame transformation—disrupting existing frame equilibrium.

3. **Frame Blending Strategy** (unchanged, 6.3%)  
As the most complex strategy type, its unique multi-frame coordination function prevents merger with other strategies. Through cognitive operations including preparation, information exploration, uncertainty expression, self-correction, processing signals, and realization markers, it achieves creative integration of different frame elements.

This simplified three-category system not only resolves technical challenges in statistical analysis but more importantly reveals the deep functional structure of service dialogue strategies—the core distinction among strategies lies in fundamental attitudes toward frames: maintenance, transformation, or integration.

### S2.3 Cognitive-Pragmatic Dimension Quantification Metrics

This research developed a quantification metric system that transforms abstract cognitive-pragmatic concepts into measurable operational variables. These metrics, based on cognitive linguistics and pragmatics theories, enable empirical testing of theoretical concepts through precise operationalization.

**Frame Activation Strength**  
A 1-7 point scale measures frame prominence in discourse. The scoring criteria integrate three dimensions: explicitness of frame elements (number of explicitly mentioned frame components), dominance of discourse function (degree of frame contribution to discourse meaning), and clarity of participant orientation (degree of participant commitment to the frame). Score anchors are set as follows: 1-2 indicates minimal activation, 3-4 indicates moderate activation, 5-6 indicates strong activation, and 7 indicates maximal activation.

**Context Dependency Index**  
A 0-1 continuous value measures the degree of frame activation dependence on immediate context. Calculated through semantic similarity between current and preceding turns using a pre-trained Word2Vec model (Google News corpus, 300 dimensions) to compute cosine similarity of word vectors. High values indicate frame activation highly depends on preceding context, while low values indicate relatively independent frame activation. The calculation formula is CDI = cos(v_current, v_previous), where v represents the average word vector of the turn.

**Institutional Presetting Index**  
A 0-1 continuous value measures the degree of institutionalization in frame activation. Calculated based on matching degree with service discourse templates, with the template library automatically extracted through n-gram analysis (n=3,4,5) of high-frequency expressions. The matching algorithm considers word order and synonym substitution, using a weighted combination of edit distance and semantic similarity. High values indicate frame activation follows institutional norms, while low values indicate innovative or personalized characteristics in frame activation.

**Cognitive Load Index**  
A 1-10 continuous scale assesses cognitive complexity of turn processing. Calculated by integrating three indicators: information density (number of information units per turn), topic complexity (number of different topics involved), and processing requirements (number of reasoning steps required). Scores are obtained through standardized weighted averaging, with weights of 0.4, 0.3, and 0.3 respectively.

**Strategy Efficacy**  
A 1-7 scale assesses the degree to which strategies achieve communicative goals. Scoring is based on two dimensions: immediate effect (response type and degree in the next turn) and cumulative effect (strategy contribution to overall task completion). Scoring criteria include: 1-2 indicates strategy failure or negative effects, 3-4 indicates partial success, 5-6 indicates basic success, and 7 indicates complete success.

**Adaptation Index**  
A 0-1 continuous value measures the degree of strategy selection response to contextual changes. Calculated by comparing temporal correspondence between strategy transitions and contextual changes. High values indicate strategy selection highly responsive to contextual changes, while low values indicate relatively independent strategy selection from context.

### S2.4 Coder Training Procedures and Reliability Testing

The coding work was completed by two research assistants with linguistics backgrounds, both receiving systematic training. The training program consisted of four phases totaling 40 hours.

**Phase 1: Theoretical Learning (8 hours)**  
Coders studied foundational theories of frame semantics, cognitive linguistics, and institutional discourse analysis to understand the theoretical basis of each coding category. Through reading key literature and discussion, they established conceptual understanding of the coding system.

**Phase 2: Example Analysis (8 hours)**  
Practice using pre-coded standard cases, including typical cases and boundary cases. Coders learned to identify frame types, judge strategy selection, and assess cognitive-pragmatic indicators. Detailed discussion followed each case to clarify coding standards.

**Phase 3: Trial Coding (16 hours)**  
Coders independently coded 10 dialogues (informal data), regularly comparing coding results and discussing disagreements. The goal of this phase was to achieve acceptable consistency levels (κ > 0.80). Based on issues discovered during trial coding, the coding manual was revised and refined.

**Phase 4: Formal Coding (8 hours supervised)**  
Formal coding began with the first 8 hours under supervision to ensure coding quality. Subsequently, coders transitioned to independent coding while maintaining weekly discussion meetings to resolve difficult cases.

**Reliability Test Results**

Coding reliability was assessed through multiple indicators. For categorical variables, Cohen's kappa coefficient was used; for continuous variables, intraclass correlation coefficient (ICC) was employed. Table S1 presents detailed reliability statistics.

**Table S1. Coding Reliability Statistics**

| Coding Category | Reliability Coefficient | 95% CI | Agreement Rate | Disagreement Resolution Rate |
|-----------------|------------------------|---------|----------------|------------------------------|
| Frame Type | κ = 0.893 | [0.871, 0.915] | 91.2% | 94.3% |
| Strategy Selection | κ = 0.847 | [0.819, 0.875] | 87.8% | 89.9% |
| Frame Activation Strength | ICC = 0.871 | [0.843, 0.899] | — | — |
| Context Dependency | ICC = 0.854 | [0.822, 0.886] | — | — |
| Institutional Presetting | ICC = 0.839 | [0.805, 0.873] | — | — |
| Cognitive Load | ICC = 0.823 | [0.786, 0.860] | — | — |
| Strategy Efficacy | ICC = 0.812 | [0.773, 0.851] | — | — |

All reliability indicators reached or exceeded the 0.80 standard, indicating good coding reliability. Test-retest reliability after six months was κ = 0.908, demonstrating good temporal stability of the coding framework.

---

## S3. Statistical Analysis Technical Details

### S3.1 Multilevel Model Construction Process and Complete Model Specifications

Service dialogue data exhibits a typical three-level nested structure with turns nested within speakers and speakers nested within dialogues. This hierarchical structure violates the independence assumption of traditional regression analysis, necessitating multilevel modeling methods that explicitly model variance components at each level and estimate cross-level effects. This section provides complete model specifications for the four research hypotheses, including all mathematical expressions, variable definitions, and estimation methods.

#### S3.1.1 Hypothesis 1: Three-Level Linear Mixed Model for Dual Mechanisms of Frame Activation

Hypothesis 1 examines the interactive effects of context dependency and institutional presetting on frame activation using a complete three-level linear mixed model. Model construction follows a progressive strategy, expanding from the null model to the full model.

**Null Model (Variance Decomposition)**

We first fit a null model containing only random intercepts to assess variance contributions at each level:

$$Y_{ijk} = \gamma_{000} + v_{00k} + u_{0jk} + \epsilon_{ijk}$$

where:
- $Y_{ijk}$ represents frame activation strength for turn $i$ of speaker $j$ in dialogue $k$
- $\gamma_{000}$ represents the grand mean
- $v_{00k} \sim N(0, \tau_{v}^2)$ represents dialogue-level random effects
- $u_{0jk} \sim N(0, \tau_{u}^2)$ represents speaker-level random effects
- $\epsilon_{ijk} \sim N(0, \sigma^2)$ represents turn-level residuals

Intraclass correlation coefficient calculations:
- Dialogue level: $ICC_{dialogue} = \tau_{v}^2 / (\tau_{v}^2 + \tau_{u}^2 + \sigma^2)$
- Speaker level: $ICC_{speaker} = (\tau_{v}^2 + \tau_{u}^2) / (\tau_{v}^2 + \tau_{u}^2 + \sigma^2)$

**Complete Three-Level Model**

Level 1 (Turn Level) Model:
$$Y_{ijk} = \beta_{0jk} + \beta_{1jk} \times \text{CD}_{c,ijk} + \beta_{2jk} \times \text{IP}_{c,ijk} + \beta_3 \times (\text{CD}_{c} \times \text{IP}_{c})_{ijk} + \beta_4 \times \text{Stage}_{ijk} + \beta_5 \times \text{TaskComplexity}_{k} + \epsilon_{ijk}$$

Level 2 (Speaker Level) Model:
$$\beta_{0jk} = \gamma_{00k} + \gamma_{01} \times \text{Role}_{jk} + u_{0jk}$$
$$\beta_{1jk} = \gamma_{10k} + \gamma_{11} \times \text{Role}_{jk} + u_{1jk}$$
$$\beta_{2jk} = \gamma_{20k} + \gamma_{21} \times \text{Role}_{jk} + u_{2jk}$$

Level 3 (Dialogue Level) Model:
$$\gamma_{00k} = \delta_{000} + v_{00k}$$
$$\gamma_{10k} = \delta_{100} + v_{10k}$$
$$\gamma_{20k} = \delta_{200} + v_{20k}$$

**Variable Definitions and Measurement**

The model incorporates the following variables:
- $Y_{ijk}$: Frame activation strength measured on a 1-7 continuous scale, integrating frame element explicitness, discourse function dominance, and participant orientation clarity
- $\text{CD}_{c,ijk}$: Centered context dependency, a group-mean centered 0-1 continuous value calculated through Word2Vec semantic similarity
- $\text{IP}_{c,ijk}$: Centered institutional presetting degree, a group-mean centered 0-1 continuous value based on n-gram template matching
- $\text{Stage}_{ijk}$: Dialogue stage as a categorical variable (opening, information exchange, negotiation-verification, closing) using effects coding
- $\text{TaskComplexity}_{k}$: Task complexity as a dialogue-level standardized index integrating query type and information requirement quantity
- $\text{Role}_{jk}$: Institutional role as a binary variable (service provider = 1, customer = 0)

**Random Effects Structure**

Random effects assume multivariate normal distributions:

$$\begin{pmatrix} u_{0jk} \\ u_{1jk} \\ u_{2jk} \end{pmatrix} \sim N\left(\begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \tau_{00} & \tau_{01} & \tau_{02} \\ \tau_{01} & \tau_{11} & \tau_{12} \\ \tau_{02} & \tau_{12} & \tau_{22} \end{pmatrix}\right)$$

$$\begin{pmatrix} v_{00k} \\ v_{10k} \\ v_{20k} \end{pmatrix} \sim N\left(\begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \omega_{00} & \omega_{01} & \omega_{02} \\ \omega_{01} & \omega_{11} & \omega_{12} \\ \omega_{02} & \omega_{12} & \omega_{22} \end{pmatrix}\right)$$

#### S3.1.2 Hypothesis 2: Multinomial Logistic Regression Model for Frame-Driven Strategy Selection

Hypothesis 2 employs multinomial logistic regression to predict strategy selection with frame reinforcement strategy as the reference category. Due to technical limitations, clustered robust standard errors approximate the handling of data hierarchical structure.

**Multinomial Logistic Regression Model Specification**

For strategy types $s \in \{2, 3\}$ (2 = frame shifting, 3 = frame blending), relative to the reference category (1 = frame reinforcement), the log odds are specified as:

$$\log\left(\frac{P(Y_{ijk} = s)}{P(Y_{ijk} = 1)}\right) = \beta_{0}^{(s)} + \beta_1^{(s)} \times \text{Frame}_{ijk} + \beta_2^{(s)} \times \text{Stage}_{ijk} + \beta_3^{(s)} \times \text{Position}_{ijk} + \beta_4^{(s)} \times \text{CogLoad}_{ijk} + \beta_5^{(s)} \times \text{Role}_{jk} + \beta_6^{(s)} \times (\text{Frame} \times \text{Stage})_{ijk} + \beta_7^{(s)} \times (\text{Frame} \times \text{Role})_{ijk}$$

**Probability Calculation**

Strategy selection probabilities are calculated through the softmax function:

$$P(Y_{ijk} = s) = \frac{\exp(\eta_{ijk}^{(s)})}{\sum_{r=1}^{3} \exp(\eta_{ijk}^{(r)})}$$

where $\eta_{ijk}^{(1)} = 0$ (reference category) and $\eta_{ijk}^{(s)}$ represents the linear predictor specified above.

**Variable Definitions**

The model includes the following predictors:
- $\text{Frame}_{ijk}$: Frame type as a five-category variable (information provision, transaction, relational, service initiation, other) using effects coding
- $\text{Position}_{ijk}$: Relative turn position as a 0-1 continuous value calculated as current turn number divided by total turn count
- $\text{CogLoad}_{ijk}$: Cognitive load index on a 1-10 continuous scale integrating information density, topic complexity, and processing requirements

**Clustered Robust Standard Errors**

Standard errors are adjusted through dialogue-level clustering:

$$\text{Var}_{cluster}(\hat{\beta}) = (X'X)^{-1} \left(\sum_{k=1}^{K} X_k' \hat{e}_k \hat{e}_k' X_k \right) (X'X)^{-1}$$

where $K$ represents the number of dialogues and $\hat{e}_k$ represents the residual vector for dialogue $k$.

#### S3.1.3 Hypothesis 3: Dynamic Models for Path Dependence in Strategy Evolution

Hypothesis 3 analyzes temporal dynamics of strategies through three complementary methods.

**Markov Chain Transition Model**

A first-order Markov chain assumes current strategy depends only on the immediately preceding state:

$$P(S_{t+1} = j | S_t = i, S_{t-1}, ..., S_1) = P(S_{t+1} = j | S_t = i) = p_{ij}$$

The transition probability matrix is defined as:

$$P = \begin{pmatrix} 
p_{11} & p_{12} & p_{13} \\
p_{21} & p_{22} & p_{23} \\
p_{31} & p_{32} & p_{33}
\end{pmatrix}$$

where $\sum_{j=1}^{3} p_{ij} = 1$ and $p_{ij} \geq 0$.

The stationary distribution $\pi$ satisfies:
$$\pi P = \pi, \quad \sum_{i=1}^{3} \pi_i = 1$$

**Survival Analysis Model**

Cox proportional hazards model analyzes strategy duration:

$$h(t|X) = h_0(t) \exp(\beta_1 \times \text{Strategy} + \beta_2 \times \text{Role} + \beta_3 \times \text{Stage} + \beta_4 \times \text{PriorDuration})$$

where $h(t|X)$ represents the conditional hazard function and $h_0(t)$ represents the baseline hazard function.

**Fixed Effects Panel Model**

The temporal decay model for strategy efficacy is specified as:

$$\text{Efficacy}_{it} = \alpha_i + \beta_1 \times \text{Repetition}_{it} + \beta_2 \times \text{Repetition}_{it}^2 + \beta_3 \times (\text{Repetition} \times \text{Role})_{it} + \beta_4 \times \text{CogLoad}_{it} + \beta_5 \times \text{FrameType}_{it} + \beta_6 \times \text{Position}_{it} + \epsilon_{it}$$

where $\alpha_i$ represents speaker fixed effects controlling for unobserved individual heterogeneity.

#### S3.1.4 Hypothesis 4: Piecewise Growth Model for Semantic Convergence in Meaning Negotiation

Hypothesis 4 employs piecewise growth curve models to capture nonlinear semantic convergence trajectories.

**Piecewise Growth Curve Model**

The base model allows slope changes at negotiation points:

$$SD_{tk} = \beta_{0k} + \beta_{1k} \times t + \sum_{p=1}^{P_k} \beta_{p+1,k} \times (t - \tau_{pk})_+ + \epsilon_{tk}$$

where:
- $SD_{tk}$ represents semantic distance at time $t$ in dialogue $k$ as a 0-1 continuous value
- $\tau_{pk}$ represents the position of the $p$-th negotiation point in dialogue $k$
- $(t - \tau_{pk})_+ = \max(0, t - \tau_{pk})$ represents the piecewise linear basis function
- $P_k$ represents the number of negotiation points identified in dialogue $k$

**Semantic Distance Calculation**

Two complementary methods are employed:

TF-IDF cosine similarity:
$$SD_{TF-IDF} = 1 - \frac{\vec{v}_A \cdot \vec{v}_B}{||\vec{v}_A|| \times ||\vec{v}_B||}$$

Word2Vec semantic distance:
$$SD_{W2V} = 1 - \cos(\vec{w}_A, \vec{w}_B)$$

where $\vec{w}_A$ and $\vec{w}_B$ represent average word vectors for turns.

**Change Point Detection Algorithm**

The CUSUM (Cumulative Sum) algorithm detects negotiation points:

$$C_t = \max(0, C_{t-1} + (x_t - \mu_0 - \frac{\delta}{2}))$$

A change point is detected when $C_t > h$, where:
- $\mu_0$ represents baseline mean
- $\delta$ represents minimum detectable change (set at 1.5 standard deviations)
- $h$ represents detection threshold (determined through Monte Carlo simulation)

**Multilevel Extension**

Accounting for speaker and dialogue-level heterogeneity:

$$SD_{tijk} = \beta_{0jk} + \beta_{1jk} \times t + \sum_{p=1}^{P_k} \beta_{p+1,jk} \times (t - \tau_{pk})_+ + \gamma \times \text{Role}_{jk} + \epsilon_{tijk}$$

Random effects specification:
$$\beta_{0jk} = \delta_{00} + v_{0k} + u_{0j}$$
$$\beta_{1jk} = \delta_{10} + v_{1k} + u_{1j}$$

#### S3.1.5 Model Estimation and Inference

**Estimation Methods**

All models employ the following estimation strategies:

Linear mixed models utilize maximum likelihood (ML) for model comparison in basic analyses and restricted maximum likelihood (REML) for unbiased variance estimates in final models. Multinomial logistic regression employs the Newton-Raphson algorithm for maximum likelihood estimation. Markov chains estimate transition probabilities directly through frequency counts. Survival analysis uses partial likelihood methods for Cox model parameter estimation. Piecewise models employ iterative algorithms to simultaneously estimate negotiation point locations and regression parameters.

**Convergence Diagnostics**

Convergence criteria include gradient norm less than $10^{-5}$, parameter change less than $10^{-6}$ between iterations, positive definiteness of the Hessian matrix, and model simplification strategies for boundary estimates where variance approaches zero.

**Statistical Inference**

Fixed effects undergo Wald tests with Kenward-Roger degrees of freedom correction. Random effects employ likelihood ratio tests accounting for boundary constraints. Model comparison utilizes AIC, BIC, and likelihood ratio tests. Multiple comparisons undergo Benjamini-Hochberg FDR correction at q = 0.05.

This complete set of model specifications provides a rigorous statistical foundation for the research, ensuring scientific validity and reproducibility. Python and R implementation code for all models is provided in Supplementary Materials S6.

### S3.2 Model Diagnostic Procedures

Model diagnostics include systematic checking of five aspects.

**Residual Normality Testing**  
Using Shapiro-Wilk tests and Q-Q plots to assess residual normality. For large samples (n > 50), graphical diagnostics are more relied upon than significance tests, as even slight deviations from normality may lead to significant results.

**Homoscedasticity Testing**  
Assessed through residual-fitted value scatter plots and Breusch-Pagan tests. If heteroscedasticity is detected, variable transformation or robust standard errors are considered.

**Multicollinearity Diagnostics**  
Calculating variance inflation factors (VIF), using 10 as the critical value. If severe collinearity is detected, it is addressed through variable deletion, principal component analysis, or ridge regression.

**Influence Point Identification**  
Using Cook's distance to identify influence points, with threshold set at 4/n. Sensitivity analysis is conducted on influence points to assess their impact on results.

**Random Effects Diagnostics**  
Checking normality and homoscedasticity of random effects, using caterpillar plots to visualize random effects distribution.

### S3.3 Sensitivity Analysis Results

Sensitivity analysis assessed the robustness of main findings to analytical decisions.

**Semantic Distance Calculation Method Comparison**  
Comparing TF-IDF cosine similarity and Word2Vec methods, correlation coefficient r = 0.783 (p < .001), with main conclusions remaining consistent. This indicates semantic distance measurement is not sensitive to specific methods.

**Negotiation Point Identification Threshold**  
Varying the CUSUM algorithm threshold from 1.0 to 2.0 standard deviations, 85% of negotiation point classifications remained stable. This indicates good robustness in negotiation point identification.

**Random Effects Structure Comparison**  
Comparing random intercepts only, random intercepts plus single slope, and full random slopes structures. Fixed effects estimates changed by less than 10%, with significance conclusions unchanged.

### S3.4 Statistical Power Analysis

Statistical power was assessed through Monte Carlo simulation (1000 iterations). Based on observed effect sizes and variance components, the analysis showed:

- Small effects (d = 0.2): power = 0.42, insufficient
- Medium effects (d = 0.5): power = 0.81, adequate
- Large effects (d = 0.8): power = 0.97, excellent

For third-order interaction effects, power decreased to 60-70%. This limitation needs consideration when interpreting results, as small interaction effects may remain undetected.

---

## S4. Data Processing and Quality Control

### S4.1 Data Preprocessing Pipeline

Data preprocessing comprises five steps to ensure data quality and analysis validity.

**Data Cleaning**  
Removing completely duplicate records, correcting data type errors, and converting anomaly markers (e.g., -999) to missing values. Checking and correcting inconsistent coding, such as uniformly treating "N/A", "NA", and "missing" as missing values.

**Feature Engineering**  
Creating interaction features (e.g., context dependency × institutional presetting), calculating polynomial features to capture nonlinear relationships, and generating temporal features (e.g., dialogue progress indicators). All derived variables have clear theoretical justification.

**Variable Transformation**  
Centering continuous variables to improve parameter interpretability. Considering log transformation for skewed distribution variables. Using effect coding rather than dummy coding for categorical variables, making intercept parameters represent overall means.

**Missing Value Handling**  
Distinguishing between random and systematic missingness. Using multiple imputation (m = 5) for random missingness, deleting systematic missingness in main analyses, and conducting sensitivity analysis to assess impact.

**Data Validation**  
Checking value ranges of key variables to ensure logical consistency (e.g., percentages between 0-1). Verifying completeness of hierarchical identifiers to ensure each turn correctly associates with speakers and dialogues.

### S4.2 Missing Data Handling

Missing data analysis showed an overall missing rate of 3.2%, within acceptable range. Little's MCAR test (χ² = 142.3, p = 0.082) indicated the missing mechanism approximates missing completely at random.

For key analysis variables, multiple imputation was employed. The imputation model included all analysis variables and auxiliary variables, using predictive mean matching for continuous variables and logistic regression for binary variables. Differences between imputed datasets were small, indicating stable imputation.

### S4.3 Outlier Detection and Treatment

Multiple methods were used to identify outliers. Univariate outliers were identified using the boxplot 1.5×IQR rule. Multivariate outliers were identified using Mahalanobis distance, with the 95th percentile of the χ² distribution as threshold.

For identified outliers (2.1% of total data), coding errors were first checked. After confirming genuine extreme values, they were retained in main analyses but subjected to sensitivity analysis. If outliers significantly affected results, Winsorization was applied, replacing extreme values with 95th or 5th percentiles.

### S4.4 Detailed Reliability Test Results

Beyond the overall reliability reported in Section S2.4, detailed reliability analysis by phase and category was conducted.

**Reliability Differences by Dialogue Phase**  
Opening phase showed highest coding consistency (κ = 0.923), as interaction patterns in this phase are most formulaic. Negotiation phase showed relatively lower consistency (κ = 0.812), reflecting the complexity and diversity of interactions in this phase.

**Reliability Differences by Frame Type**  
Service initiation frames showed highest identification consistency (κ = 0.947), followed by information provision frames (κ = 0.891), with relational frames showing relatively lower consistency (κ = 0.798), possibly due to greater subjectivity in judging emotional dimensions.

**Disagreement Pattern Analysis**  
Analysis of 1,538 initial disagreement cases revealed that 62% involved boundary cases (ambiguous areas between two categories), 28% involved differences in standard understanding, and 10% were attention lapses. Discussion resolved 87.2% of disagreements, with remaining cases referred to a third-party expert.

---

## S5. Supplementary Tables and Figures

### S5.1 Detailed Descriptive Statistics Tables

**Table S2. Dialogue-Level Detailed Feature Statistics**

| Dialogue ID | Turns | Words | Duration (sec) | Frame Types | Strategy Transitions | Success |
|-------------|-------|-------|----------------|-------------|---------------------|---------|
| trainline01 | 89 | 1253 | 452 | 12 | 23 | Yes |
| trainline02 | 126 | 1876 | 623 | 15 | 31 | Yes |
| trainline03 | 73 | 987 | 381 | 9 | 18 | Yes |
| ... | ... | ... | ... | ... | ... | ... |
| trainline35 | 104 | 1432 | 512 | 11 | 27 | Yes |
| Mean | 95.2 | 1367.8 | 503.6 | 11.3 | 24.7 | 94.3% |
| SD | 45.4 | 423.2 | 187.3 | 3.2 | 7.8 | — |

### S5.2 Complete Model Comparison Results

**Table S3. Hypothesis Testing Model Comparison Summary**

| Model | df | AIC | BIC | LogLik | LR χ² | p-value | Marginal R² | Conditional R² |
|-------|-----|-----|-----|--------|-------|---------|-------------|----------------|
| H1-M0 | 3 | 2952.2 | 2968.5 | -1473.1 | — | — | 0.000 | 0.210 |
| H1-M1 | 5 | 2237.2 | 2264.3 | -1114.6 | 717.12 | <.001 | 0.332 | 0.541 |
| H1-M2 | 9 | 1945.4 | 1994.2 | -963.7 | 301.80 | <.001 | 0.445 | 0.718 |
| H1-M3 | 12 | 1837.2 | 1902.3 | -906.6 | 114.20 | <.001 | 0.502 | 0.824 |
| H2-M0 | 2 | 3678.4 | 3689.2 | -1837.2 | — | — | 0.000 | 0.117 |
| H2-M1 | 8 | 3412.7 | 3456.3 | -1698.4 | 277.68 | <.001 | 0.156 | 0.267 |
| H2-M2 | 14 | 3245.6 | 3298.4 | -1608.8 | 179.12 | <.001 | 0.234 | 0.387 |

### S5.3 Diagnostic Plot Collection

This section includes the following diagnostic plots:

1. **Residual Diagnostic Quartet**: Q-Q plot, residuals vs fitted values, scale-location plot, residuals vs leverage
2. **Random Effects Distribution Plots**: Histograms and caterpillar plots of dialogue-level and speaker-level random effects
3. **Strategy Transition Network Diagram**: Directed graph showing transition probabilities between strategies
4. **Semantic Distance Trajectory Plots**: Overlaid semantic distance changes for all 35 dialogues
5. **Frame Activation Heatmap**: Frame activation strength across different dialogue phases and roles

Due to the large number of figures, the complete figure collection is available through the online version of supplementary materials.

---

## Appendix: Data and Code Availability Statement

All analysis code, data processing scripts, and visualization programs from this research are open source. The code repository contains complete analysis workflows, detailed documentation, and usage instructions. Data usage must comply with the SPAADIA corpus license agreement.

**GitHub Repository**: https://github.com/chenwangfang/A-Multilevel-Logistic-Regression-Analysis

