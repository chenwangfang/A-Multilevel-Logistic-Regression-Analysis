# H3: Path Dependence and Dynamic Adaptation in Strategy Evolution (Advanced)

## 分析摘要

本分析使用高级统计方法验证了H3假设：服务对话中的策略选择表现出显著的路径依赖性，同时保持动态适应能力。

## 主要发现

### 1. 马尔可夫链分析（增强版）

- 混合时间: 0.28 步
- 最稳定策略: Frame Blending (稳态概率 = 0.347)
- 一阶马尔可夫性假设检验通过（p > 0.05）
- 策略转换表现出明显的惯性和周期性

### 2. 生存分析结果
- 强化策略的中位生存时间最长（7.3个话轮）
- 抵抗策略的中位生存时间最短（3.2个话轮）
- Cox模型显示对话位置显著影响策略持续性（HR = 0.85, p < 0.01）

### 3. 路径依赖的非线性效应
- 策略重复的效应呈现倒U型曲线
- 转折点出现在重复12.5次左右
- 过度重复导致转换概率上升

### 4. 网络分析
- 策略转换网络密度 = 0.72
- 强化和响应策略具有最高的中心性
- 存在明显的策略转换循环

### 5. Bootstrap验证

- persistence: 0.477, 95% BCa CI=[0.441, 0.520]
- entropy: 1.076, 95% BCa CI=[1.074, 1.078]

### 6. 置换检验
- 路径依赖性显著（p < 0.001）
- 观察到的自相关远高于随机期望

## 方法学创新

1. **增强的马尔可夫分析**：计算混合时间和稳态分布
2. **生存分析**：量化策略持续性
3. **网络分析**：揭示策略转换的结构特征
4. **非线性建模**：捕捉路径依赖的复杂性

## 理论贡献

1. 证实了策略选择的历史依赖性
2. 发现了适应性和稳定性的平衡机制
3. 揭示了策略演化的网络结构
4. 支持了有限理性决策理论

## 统计结果

### Table 9. Strategy Transition Probability Matrix and Stationary Distribution
见 tables/table9_transition_matrix_advanced.csv

### Table 10. Survival Analysis Results for Strategy Persistence
见 tables/table10_survival_analysis_advanced.csv

## 图形展示

### Figure 4. Dynamic Characteristics of Strategy Evolution
见 figures/figure4_strategy_evolution_advanced.jpg

---
生成时间：2025-08-31 07:44:28
分析版本：高级版（含增强马尔可夫、生存分析、网络分析）
