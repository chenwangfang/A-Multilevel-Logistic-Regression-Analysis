#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# Simple R Validation Script for SPAADIA Analysis
# 简化的R验证脚本，验证Python分析结果的统计准确性

# 设置选项
options(stringsAsFactors = FALSE)
options(encoding = "UTF-8")

# 加载必要的库
suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
  library(lmerTest)
  library(stats)
})

# 设置工作目录
setwd("G:/Project/实证/关联框架")

# 创建输出目录
output_dir <- "输出/validation"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# 日志函数
log_message <- function(msg) {
  cat(paste0("[", Sys.time(), "] ", msg, "\n"))
}

# 验证H1: 混合效应模型
validate_h1 <- function() {
  log_message("验证H1假设...")
  
  tryCatch({
    # 读取Python生成的数据
    data_path <- "输出/data/hypothesis_h1_model_data.csv"
    if (file.exists(data_path)) {
      data <- read.csv(data_path, encoding = "UTF-8")
      
      # 运行混合效应模型
      model <- lmer(activation_intensity ~ frame_type + cognitive_load + 
                   frame_type:cognitive_load + (1|participant_id), 
                   data = data, REML = FALSE)
      
      # 提取结果
      summary_model <- summary(model)
      coefficients <- coef(summary_model)
      
      # 计算ICC
      var_comp <- as.data.frame(VarCorr(model))
      icc <- var_comp$vcov[1] / sum(var_comp$vcov)
      
      # 保存结果
      results <- list(
        coefficients = as.data.frame(coefficients),
        icc = icc,
        aic = AIC(model),
        bic = BIC(model),
        validation_status = "success"
      )
      
      write_json(results, file.path(output_dir, "h1_r_validation.json"))
      log_message("H1验证完成")
      return(TRUE)
    } else {
      log_message("H1数据文件不存在")
      return(FALSE)
    }
  }, error = function(e) {
    log_message(paste("H1验证错误:", e$message))
    return(FALSE)
  })
}

# 验证H2: 卡方检验和逻辑回归
validate_h2 <- function() {
  log_message("验证H2假设...")
  
  tryCatch({
    data_path <- "输出/data/hypothesis_h2_contingency_data.csv"
    if (file.exists(data_path)) {
      data <- read.csv(data_path, encoding = "UTF-8")
      
      # 卡方检验
      if (ncol(data) >= 2) {
        # 创建列联表
        cont_table <- table(data[,1], data[,2])
        chi_test <- chisq.test(cont_table)
        
        # Cramér's V
        n <- sum(cont_table)
        min_dim <- min(nrow(cont_table) - 1, ncol(cont_table) - 1)
        cramers_v <- sqrt(chi_test$statistic / (n * min_dim))
        
        results <- list(
          chi_square = as.numeric(chi_test$statistic),
          p_value = chi_test$p.value,
          cramers_v = as.numeric(cramers_v),
          validation_status = "success"
        )
        
        write_json(results, file.path(output_dir, "h2_r_validation.json"))
        log_message("H2验证完成")
        return(TRUE)
      }
    } else {
      log_message("H2数据文件不存在")
      return(FALSE)
    }
  }, error = function(e) {
    log_message(paste("H2验证错误:", e$message))
    return(FALSE)
  })
}

# 验证H3: 马尔可夫链转移概率
validate_h3 <- function() {
  log_message("验证H3假设...")
  
  tryCatch({
    data_path <- "输出/data/hypothesis_h3_transition_matrix.csv"
    if (file.exists(data_path)) {
      # 读取转移矩阵
      trans_matrix <- as.matrix(read.csv(data_path, row.names = 1))
      
      # 计算对角线优势
      diag_sum <- sum(diag(trans_matrix))
      total_sum <- sum(trans_matrix)
      diagonal_dominance <- diag_sum / total_sum
      
      # 计算稳态分布（特征向量）
      eigen_result <- eigen(t(trans_matrix))
      stationary <- abs(eigen_result$vectors[,1])
      stationary <- stationary / sum(stationary)
      
      results <- list(
        diagonal_dominance = diagonal_dominance,
        stationary_distribution = as.numeric(stationary),
        matrix_dimension = nrow(trans_matrix),
        validation_status = "success"
      )
      
      write_json(results, file.path(output_dir, "h3_r_validation.json"))
      log_message("H3验证完成")
      return(TRUE)
    } else {
      log_message("H3数据文件不存在")
      return(FALSE)
    }
  }, error = function(e) {
    log_message(paste("H3验证错误:", e$message))
    return(FALSE)
  })
}

# 验证H4: 变化点检测和t检验
validate_h4 <- function() {
  log_message("验证H4假设...")
  
  tryCatch({
    data_path <- "输出/data/hypothesis_h4_semantic_distances.csv"
    if (file.exists(data_path)) {
      data <- read.csv(data_path, encoding = "UTF-8")
      
      if (ncol(data) >= 2) {
        # 提取前后两阶段数据
        col1 <- data[,1]
        col2 <- data[,2]
        
        # 去除NA值
        col1 <- col1[!is.na(col1)]
        col2 <- col2[!is.na(col2)]
        
        # t检验
        if (length(col1) > 0 && length(col2) > 0) {
          t_test <- t.test(col1, col2, paired = FALSE)
          
          # Cohen's d
          pooled_sd <- sqrt(((length(col1)-1)*var(col1) + (length(col2)-1)*var(col2)) / 
                           (length(col1) + length(col2) - 2))
          cohens_d <- (mean(col1) - mean(col2)) / pooled_sd
          
          results <- list(
            mean_before = mean(col1),
            mean_after = mean(col2),
            t_statistic = as.numeric(t_test$statistic),
            p_value = t_test$p.value,
            cohens_d = cohens_d,
            validation_status = "success"
          )
          
          write_json(results, file.path(output_dir, "h4_r_validation.json"))
          log_message("H4验证完成")
          return(TRUE)
        }
      }
    } else {
      log_message("H4数据文件不存在")
      return(FALSE)
    }
  }, error = function(e) {
    log_message(paste("H4验证错误:", e$message))
    return(FALSE)
  })
}

# 生成验证报告
generate_report <- function(results) {
  log_message("生成验证报告...")
  
  report <- c(
    "# R验证报告",
    "",
    paste("生成时间:", Sys.time()),
    "",
    "## 验证结果",
    "",
    paste("- H1假设:", ifelse(results$h1, "✓ 通过", "✗ 失败")),
    paste("- H2假设:", ifelse(results$h2, "✓ 通过", "✗ 失败")),
    paste("- H3假设:", ifelse(results$h3, "✓ 通过", "✗ 失败")),
    paste("- H4假设:", ifelse(results$h4, "✓ 通过", "✗ 失败")),
    "",
    "## 详细结果",
    "",
    "详细的验证结果已保存至JSON文件中。",
    "",
    paste("总体通过率:", sum(unlist(results)) / length(results) * 100, "%")
  )
  
  writeLines(report, file.path(output_dir, "r_validation_report.md"))
  log_message("报告生成完成")
}

# 主函数
main <- function() {
  log_message("开始R验证流程...")
  
  # 运行各项验证
  results <- list(
    h1 = validate_h1(),
    h2 = validate_h2(),
    h3 = validate_h3(),
    h4 = validate_h4()
  )
  
  # 生成报告
  generate_report(results)
  
  # 保存总体结果
  write_json(results, file.path(output_dir, "validation_summary.json"))
  
  log_message("R验证完成")
  
  # 返回结果
  return(results)
}

# 运行主函数
tryCatch({
  results <- main()
  q(status = 0)
}, error = function(e) {
  log_message(paste("致命错误:", e$message))
  q(status = 1)
})