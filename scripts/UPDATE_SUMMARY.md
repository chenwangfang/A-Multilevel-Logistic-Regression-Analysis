# 系统更新总结

更新日期：2025-08-29

## 主要更新内容

### 1. 临时文件清理
- ✅ 删除了61个临时脚本和文档
- ✅ 当前保留29个核心Python脚本 + 1个R脚本 + 4个文档
- ✅ 创建了`run_all_figures.py`批量生成图表脚本

### 2. 文档更新
- ✅ 删除了所有特定期刊相关表述
- ✅ 删除了Applied_Linguistics_Statistical_Checklist.md
- ✅ 更新README.md，使用更通用的学术表述

### 3. 代码修改
- ✅ 修复了figure1-5脚本中的JSON字段名不一致问题
- ✅ figure1: H2默认值从理论值改为实际值（χ²=62.24, p<0.001）
- ✅ figure2: correlation字段名改为context_institutional_correlation
- ✅ figure5: required_fields改为实际存在的字段名

## 核心运行脚本

1. **run_all_analyses_advanced.py** - 运行所有分析（推荐）
2. **run_hybrid_analysis.py** - 快速混合分析
3. **run_all_figures.py** - 批量生成所有图表（新增）

## 重要发现

H2假设检验结果与理论预期相反：
- 预期：框架与策略独立（χ²≈2.71, p>0.05）
- 实际：框架与策略显著相关（χ²=62.24, p<0.001, V=0.259）

这是一个重要的研究发现，表明框架类型确实影响策略选择。

## 系统特点

- ✅ 所有数据完全来源于JSON文件
- ✅ 效应量和95%置信区间完整报告
- ✅ 统计功效分析（>0.80）
- ✅ FDR多重比较校正
- ✅ 1200 DPI高质量图形输出

## 输出位置

- 中文版：`G:/Project/实证/关联框架/输出/`
- 英文版：`G:/Project/实证/关联框架/output/`