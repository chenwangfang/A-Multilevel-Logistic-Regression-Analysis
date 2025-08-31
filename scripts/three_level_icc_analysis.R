#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# 三层嵌套模型ICC计算
# 使用lme4包正确拟合话轮-说话人-对话三层结构

# 加载必要的包
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
})

# 设置工作目录
setwd("/mnt/g/Project/实证/关联框架/Python脚本/SPAADIA分析脚本")

# 设置UTF-8编码（Windows兼容）
Sys.setlocale("LC_ALL", "C.UTF-8")

# 读取数据
cat("读取数据...\n")
data <- read.csv("temp_data_for_r.csv", stringsAsFactors = FALSE, encoding = "UTF-8")

# 检查数据结构
cat(sprintf("数据维度: %d 行 x %d 列\n", nrow(data), ncol(data)))
cat(sprintf("对话数: %d\n", length(unique(data$dialogue_id))))
cat(sprintf("说话人数: %d\n", length(unique(data$speaker_id_unique))))

# 确保因子变量
data$dialogue_id <- as.factor(data$dialogue_id)
data$speaker_id_unique <- as.factor(data$speaker_id_unique)
data$speaker_role <- as.factor(data$speaker_role)

# 方法1: 完整的三层嵌套模型
# 话轮嵌套在说话人中，说话人嵌套在对话中
cat("\n=== 拟合三层嵌套模型 ===\n")

tryCatch({
  # 空模型：只有随机效应
  # (1|dialogue_id/speaker_id_unique) 表示说话人嵌套在对话中
  model_nested <- lmer(activation_strength ~ 1 + (1|dialogue_id/speaker_id_unique), 
                       data = data, 
                       REML = TRUE)
  
  # 提取方差成分
  vc <- VarCorr(model_nested)
  
  # 对话层方差
  var_dialogue <- as.numeric(vc$dialogue_id[1])
  
  # 说话人层方差（在对话内）
  var_speaker <- as.numeric(vc$`speaker_id_unique:dialogue_id`[1])
  
  # 残差方差（话轮层）
  var_residual <- attr(vc, "sc")^2
  
  # 总方差
  var_total <- var_dialogue + var_speaker + var_residual
  
  # 计算ICC
  icc_dialogue <- var_dialogue / var_total
  icc_speaker <- var_speaker / var_total
  icc_cumulative <- (var_dialogue + var_speaker) / var_total
  
  # 计算百分比
  pct_dialogue <- (var_dialogue / var_total) * 100
  pct_speaker <- (var_speaker / var_total) * 100
  pct_residual <- (var_residual / var_total) * 100
  
  cat("\n方差分解结果:\n")
  cat(sprintf("对话层方差: %.4f (%.1f%%)\n", var_dialogue, pct_dialogue))
  cat(sprintf("说话人层方差: %.4f (%.1f%%)\n", var_speaker, pct_speaker))
  cat(sprintf("残差方差: %.4f (%.1f%%)\n", var_residual, pct_residual))
  cat(sprintf("总方差: %.4f\n", var_total))
  
  cat("\nICC值:\n")
  cat(sprintf("对话层ICC: %.4f\n", icc_dialogue))
  cat(sprintf("说话人层ICC: %.4f\n", icc_speaker))
  cat(sprintf("累积ICC: %.4f\n", icc_cumulative))
  
  # 保存结果到JSON
  results_nested <- list(
    variance_components = list(
      dialogue = var_dialogue,
      speaker = var_speaker,
      residual = var_residual,
      total = var_total
    ),
    variance_percentages = list(
      dialogue_pct = pct_dialogue,
      speaker_pct = pct_speaker,
      residual_pct = pct_residual
    ),
    icc = list(
      dialogue = icc_dialogue,
      speaker = icc_speaker,
      cumulative = icc_cumulative
    ),
    model_info = list(
      type = "three_level_nested",
      formula = "activation_strength ~ 1 + (1|dialogue_id/speaker_id_unique)",
      n_observations = nrow(data),
      n_dialogues = length(unique(data$dialogue_id)),
      n_speakers = length(unique(data$speaker_id_unique))
    )
  )
  
}, error = function(e) {
  cat(sprintf("三层嵌套模型失败: %s\n", e$message))
  results_nested <- NULL
})

# 方法2: 交叉分类模型（如果嵌套模型失败）
if (is.null(results_nested)) {
  cat("\n=== 尝试交叉分类模型 ===\n")
  
  tryCatch({
    # 将说话人和对话作为交叉的随机效应
    model_crossed <- lmer(activation_strength ~ 1 + (1|dialogue_id) + (1|speaker_id_unique), 
                         data = data, 
                         REML = TRUE)
    
    vc_crossed <- VarCorr(model_crossed)
    
    var_dialogue <- as.numeric(vc_crossed$dialogue_id[1])
    var_speaker <- as.numeric(vc_crossed$speaker_id_unique[1])
    var_residual <- attr(vc_crossed, "sc")^2
    var_total <- var_dialogue + var_speaker + var_residual
    
    icc_dialogue <- var_dialogue / var_total
    icc_speaker <- var_speaker / var_total
    icc_cumulative <- (var_dialogue + var_speaker) / var_total
    
    pct_dialogue <- (var_dialogue / var_total) * 100
    pct_speaker <- (var_speaker / var_total) * 100
    pct_residual <- (var_residual / var_total) * 100
    
    cat("\n交叉分类模型方差分解:\n")
    cat(sprintf("对话层方差: %.4f (%.1f%%)\n", var_dialogue, pct_dialogue))
    cat(sprintf("说话人层方差: %.4f (%.1f%%)\n", var_speaker, pct_speaker))
    cat(sprintf("残差方差: %.4f (%.1f%%)\n", var_residual, pct_residual))
    
    results_nested <- list(
      variance_components = list(
        dialogue = var_dialogue,
        speaker = var_speaker,
        residual = var_residual,
        total = var_total
      ),
      variance_percentages = list(
        dialogue_pct = pct_dialogue,
        speaker_pct = pct_speaker,
        residual_pct = pct_residual
      ),
      icc = list(
        dialogue = icc_dialogue,
        speaker = icc_speaker,
        cumulative = icc_cumulative
      ),
      model_info = list(
        type = "crossed_random_effects",
        formula = "activation_strength ~ 1 + (1|dialogue_id) + (1|speaker_id_unique)"
      )
    )
    
  }, error = function(e) {
    cat(sprintf("交叉分类模型也失败: %s\n", e$message))
    results_nested <- list(error = e$message)
  })
}

# 保存结果
if (!is.null(results_nested)) {
  json_output <- toJSON(results_nested, pretty = TRUE, auto_unbox = TRUE)
  write(json_output, "three_level_icc_results.json")
  cat("\n结果已保存到 three_level_icc_results.json\n")
  
  # 同时输出到控制台供Python读取
  cat("\n===JSON_START===\n")
  cat(json_output)
  cat("\n===JSON_END===\n")
}

cat("\n分析完成!\n")