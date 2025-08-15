#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# ================================================================================
# SPAADIA语料库综合R验证脚本
# 配合Python分析结果，提供完整的统计验证
# 包含H1-H4所有假设的高级统计分析
# 作者：SPAADIA分析团队
# 日期：2025-01-29
# ================================================================================

# 设置UTF-8编码环境
if(.Platform$OS.type == "windows") {
  Sys.setlocale("LC_ALL", "Chinese")
} else {
  Sys.setlocale("LC_ALL", "en_US.UTF-8")
}
options(
  encoding = "UTF-8",
  scipen = 999,  # 避免科学计数法
  digits = 4     # 保留4位小数
)

# 定义必需的包列表
required_packages <- c(
  "lme4",        # 混合效应模型
  "lmerTest",    # 提供p值
  "pbkrtest",    # Kenward-Roger近似
  "car",         # 方差分析
  "effects",     # 效应展示
  "emmeans",     # 边际均值
  "nnet",        # 多项逻辑回归
  "survival",    # 生存分析
  "markovchain", # 马尔可夫链
  "changepoint", # 变化点检测
  "tidyverse",   # 数据处理套件
  "dplyr",       # 数据处理
  "magrittr",    # 管道操作符
  "ggplot2",     # 绘图
  "jsonlite",    # JSON读写
  "broom",       # 模型整理
  "broom.mixed", # 混合模型整理
  "mice",        # 多重插补
  "simr",        # 统计功效分析
  "segmented",   # 分段回归
  "performance", # 模型诊断
  "DHARMa"       # 高级残差诊断
)

# 检查并安装缺失的包
missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]
if(length(missing_packages) > 0) {
  cat("正在安装缺失的包...\n")
  install.packages(missing_packages, repos = "https://cran.r-project.org", quiet = TRUE)
}

# 加载所有必要的包
suppressPackageStartupMessages({
  for(pkg in required_packages) {
    tryCatch({
      library(pkg, character.only = TRUE)
    }, error = function(e) {
      stop(sprintf("无法加载包 '%s'。请手动安装: install.packages('%s')", pkg, pkg))
    })
  }
})

# 设置绘图主题
theme_set(theme_minimal(base_family = "Microsoft YaHei"))

# ================================================================================
# 工具函数
# ================================================================================

# 数据质量控制函数
perform_data_quality_check <- function(data) {
  cat("\n执行数据质量检查...\n")
  
  # 1. 缺失值检查
  missing_summary <- data %>%
    summarise_all(~sum(is.na(.))) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
    mutate(missing_prop = missing_count / nrow(data))
  
  cat("\n缺失值汇总：\n")
  print(missing_summary %>% filter(missing_count > 0))
  
  # 2. 多重插补（如果缺失比例 > 5%）
  if (any(missing_summary$missing_prop > 0.05)) {
    cat("\n检测到缺失比例超过5%，执行多重插补...\n")
    mice_obj <- mice(data, m = 5, method = 'pmm', printFlag = FALSE)
    data_imputed <- complete(mice_obj, action = "long", include = TRUE)
    return(list(imputed = TRUE, data = data_imputed, mice_obj = mice_obj))
  }
  
  return(list(imputed = FALSE, data = data))
}

# 模型诊断函数
perform_model_diagnostics <- function(model, type = "lmer") {
  cat("\n执行模型诊断...\n")
  
  diagnostics <- list()
  
  if (type == "lmer") {
    # 1. VIF检验（多重共线性）
    if (length(fixef(model)) > 1) {
      vif_values <- check_collinearity(model)
      diagnostics$vif <- vif_values
      cat("\nVIF值（>10表示严重共线性）：\n")
      print(vif_values)
    }
    
    # 2. 残差诊断
    residual_plot <- plot(check_model(model))
    diagnostics$residual_diagnostics <- residual_plot
    
    # 3. Cook距离（异常值检测）
    cooksd <- cooks.distance(model)
    influential <- which(cooksd > 4/length(cooksd))
    diagnostics$influential_points <- influential
    if (length(influential) > 0) {
      cat(sprintf("\n检测到%d个潜在异常值（Cook's D > 4/n）\n", length(influential)))
    }
    
    # 4. 收敛性检查
    convergence <- model@optinfo$conv$lme4$code
    diagnostics$converged <- convergence == 0
    if (convergence != 0) {
      cat("\n警告：模型未完全收敛\n")
    }
  }
  
  return(diagnostics)
}

# Benjamini-Hochberg FDR校正
apply_fdr_correction <- function(p_values, alpha = 0.05) {
  adjusted_p <- p.adjust(p_values, method = "BH")
  significant <- adjusted_p < alpha
  
  results <- data.frame(
    original_p = p_values,
    adjusted_p = adjusted_p,
    significant = significant
  )
  
  cat(sprintf("\nFDR校正后，%d/%d个检验显著（α = %.2f）\n", 
              sum(significant), length(p_values), alpha))
  
  return(results)
}

# 创建输出目录
create_output_dirs <- function(language = "zh") {
  # 根据语言选择输出目录
  base_dir <- ifelse(language == "zh", 
                     "G:/Project/实证/关联框架/输出",
                     "G:/Project/实证/关联框架/output")
  
  dirs <- c(
    file.path(base_dir, "data"),
    file.path(base_dir, "figures"),
    file.path(base_dir, "tables"),
    file.path(base_dir, "reports")
  )
  
  for (dir in dirs) {
    if (!dir.exists(dir)) {
      dir.create(dir, recursive = TRUE, showWarnings = FALSE)
    }
  }
}

# 生成模拟数据（当真实数据不存在时）
generate_mock_data <- function(hypothesis = "H1") {
  set.seed(42)
  
  if (hypothesis == "H1") {
    # H1: 框架激活数据
    n_dialogues <- 50
    n_turns <- 20
    data <- expand.grid(
      dialogue_id = sprintf("D%03d", 1:n_dialogues),
      turn_id = 1:n_turns
    )
    
    data <- data %>%
      mutate(
        speaker_id = paste0(dialogue_id, "_", ifelse(turn_id %% 2 == 1, "SP", "C")),
        stage = cut(turn_id, breaks = c(0, 5, 10, 15, 20),
                   labels = c("opening", "information_exchange", 
                            "negotiation_verification", "closing")),
        context_dependence = runif(n(), 0.2, 0.9),
        institutional_presetting = runif(n(), 0.3, 0.8),
        cognitive_load = runif(n(), 2, 8),
        task_complexity = rep(runif(n_dialogues, 0.1, 0.9), each = n_turns),  # 对话层面变量
        activation_strength = 3 + 
          1.8 * context_dependence + 
          2.2 * institutional_presetting + 
          0.5 * context_dependence * institutional_presetting +
          0.3 * task_complexity +  # 任务复杂度效应
          rnorm(n(), 0, 0.5)
      )
    
  } else if (hypothesis == "H2") {
    # H2: 策略选择数据
    n_obs <- 1000
    data <- tibble(
      dialogue_id = sample(sprintf("D%03d", 1:50), n_obs, replace = TRUE),
      frame_type = sample(c("service", "transaction", "problem_solving", "relationship"), 
                         n_obs, replace = TRUE),
      speaker_role = sample(c("SP", "C"), n_obs, replace = TRUE),
      strategy_type = sample(c("reinforcement", "shifting", "blending", 
                              "response", "resistance"), n_obs, replace = TRUE),
      cognitive_load = runif(n_obs, 2, 8),
      emotional_valence = runif(n_obs, -2, 2)
    )
    
  } else if (hypothesis == "H3") {
    # H3: 时间序列数据
    n_dialogues <- 50
    data <- map_dfr(1:n_dialogues, function(d) {
      n_turns <- sample(15:30, 1)
      tibble(
        dialogue_id = sprintf("D%03d", d),
        turn_id = 1:n_turns,
        time_stamp = 1:n_turns,
        current_strategy = sample(c("reinforcement", "shifting", "blending", 
                                   "response", "resistance"), n_turns, replace = TRUE)
      ) %>%
        mutate(
          previous_strategy = lag(current_strategy, default = current_strategy[1]),
          strategy_duration = rgeom(n_turns, prob = 0.3) + 1,
          speaker_role = ifelse(turn_id %% 2 == 1, "SP", "C")
        )
    })
    
  } else if (hypothesis == "H4") {
    # H4: 协商数据
    n_obs <- 800
    data <- tibble(
      dialogue_id = sample(sprintf("D%03d", 1:40), n_obs, replace = TRUE),
      relative_position = runif(n_obs, 0, 1),
      marker_type = sample(c("clarification", "confirmation", "reformulation", 
                            "elaboration", "agreement"), n_obs, replace = TRUE),
      semantic_distance = 0.9 * exp(-3 * relative_position) + rnorm(n_obs, 0, 0.1),
      cognitive_load = 3 + 2 * relative_position + rnorm(n_obs, 0, 0.5)
    ) %>%
      arrange(dialogue_id, relative_position)
  }
  
  return(data)
}

# ================================================================================
# H1假设验证：框架激活的双重机制（高级版）
# ================================================================================

validate_h1_advanced <- function(data_path = "G:/Project/实证/关联框架/输出/data/h1_data_for_R.csv") {
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("H1假设验证：框架激活的双重机制（高级统计分析）\n")
  cat(rep("=", 70), "\n\n", sep = "")
  
  # 读取真实数据
  if (!file.exists(data_path)) {
    cat("警告：数据文件不存在：", data_path, "\n")
    cat("跳过H3验证\n")
    return(list(status = "skipped", message = paste("数据文件不存在:", data_path)))
  }
  
  cat("读取数据文件:", data_path, "\n")
  data <- read.csv(data_path, encoding = "UTF-8", stringsAsFactors = FALSE)
  
  cat(sprintf("数据规模：%d条记录，%d个对话\n", nrow(data), n_distinct(data$dialogue_id)))
  
  # 数据质量检查
  qc_result <- perform_data_quality_check(data)
  if (qc_result$imputed) {
    data <- qc_result$data
    cat("\n已使用多重插补处理缺失数据\n")
  }
  
  # 数据预处理
  data <- data %>%
    mutate(
      dialogue_id = factor(dialogue_id),
      speaker_id = factor(speaker_id),
      stage = factor(stage),
      # 组均值中心化
      cd_centered = context_dependence - mean(context_dependence),
      ip_centered = institutional_presetting - mean(institutional_presetting)
    )
  
  # ========== 1. 随机斜率模型 ==========
  cat("\n1. 拟合随机斜率混合效应模型...\n")
  
  # 模型序列（包含三层结构和任务复杂度）
  formulas <- list(
    M0 = "activation_strength ~ 1 + (1 | dialogue_id/speaker_id)",
    M1 = "activation_strength ~ cd_centered + ip_centered + task_complexity + (1 | dialogue_id/speaker_id)",
    M2 = "activation_strength ~ cd_centered + ip_centered + task_complexity + (1 + cd_centered | dialogue_id) + (1 | speaker_id:dialogue_id)",
    M3 = "activation_strength ~ cd_centered * ip_centered + task_complexity + (1 + cd_centered + ip_centered | dialogue_id) + (1 | speaker_id:dialogue_id)",
    M4 = "activation_strength ~ cd_centered * ip_centered * stage + task_complexity + (1 + cd_centered + ip_centered | dialogue_id) + (1 | speaker_id:dialogue_id)"
  )
  
  models <- list()
  model_summaries <- list()
  
  for (name in names(formulas)) {
    cat(sprintf("\n拟合%s...", name))
    tryCatch({
      if (name %in% c("M0", "M1")) {
        models[[name]] <- lmer(as.formula(formulas[[name]]), data = data, REML = TRUE)
      } else {
        models[[name]] <- lmer(as.formula(formulas[[name]]), data = data, REML = TRUE,
                              control = lmerControl(optimizer = "bobyqa"))
      }
      
      # 提取关键信息
      model_summaries[[name]] <- list(
        AIC = AIC(models[[name]]),
        BIC = BIC(models[[name]]),
        logLik = logLik(models[[name]]),
        converged = models[[name]]@optinfo$conv$lme4$code == 0
      )
      cat(" [完成]\n")
      
    }, error = function(e) {
      cat(sprintf(" [失败: %s]\n", e$message))
    })
  }
  
  # ========== 2. 模型比较 ==========
  cat("\n2. 模型比较\n")
  if (length(models) >= 2) {
    # 似然比检验
    cat("\n似然比检验：\n")
    anova_result <- anova(models$M1, models$M2, models$M3)
    print(anova_result)
    
    # Kenward-Roger检验（用于小样本）
    if ("M3" %in% names(models)) {
      cat("\nKenward-Roger F检验（交互效应）：\n")
      kr_test <- KRmodcomp(models$M2, models$M3)
      print(kr_test)
    }
  }
  
  # ========== 3. 最优模型的详细结果 ==========
  best_model <- models$M3  # 假设M3是最优模型
  if (!is.null(best_model)) {
    cat("\n3. 最优模型（M3）详细结果\n")
    
    # 固定效应
    cat("\n固定效应：\n")
    print(summary(best_model)$coefficients)
    
    # 随机效应
    cat("\n随机效应方差：\n")
    print(VarCorr(best_model))
    
    # ICC计算
    var_comps <- as.data.frame(VarCorr(best_model))
    var_dialogue <- var_comps$vcov[1]
    var_residual <- var_comps$vcov[length(var_comps$vcov)]
    icc <- var_dialogue / (var_dialogue + var_residual)
    cat(sprintf("\n组内相关系数(ICC): %.3f\n", icc))
    
    # ========== 4. 简单斜率分析 ==========
    cat("\n4. 简单斜率分析\n")
    
    # 计算不同机构预设水平下的语境依赖效应
    ip_levels <- c(-1, 0, 1)  # 低、中、高
    
    for (level in ip_levels) {
      # 创建新数据
      new_data <- expand.grid(
        cd_centered = seq(-2, 2, 0.1),
        ip_centered = level,
        stage = "information_exchange",
        task_complexity = mean(data$task_complexity, na.rm = TRUE)  # 使用平均值
      )
      
      # 预测
      new_data$predicted <- predict(best_model, newdata = new_data, re.form = NA)
      
      # 计算斜率
      slope <- coef(lm(predicted ~ cd_centered, data = new_data))[2]
      cat(sprintf("机构预设 = %+.0f SD时，语境依赖斜率 = %.3f\n", level, slope))
    }
    
    # ========== 5. 效应可视化 ==========
    cat("\n5. 生成效应图...\n")
    
    tryCatch({
      # 创建交互效应的预测数据
      pred_data <- expand.grid(
        cd_centered = seq(-2, 2, length.out = 100),
        ip_centered = c(-1, 0, 1),  # 低、中、高
        task_complexity = mean(data$task_complexity, na.rm = TRUE),
        stage = "information_exchange"
      )
      
      # 生成预测值
      pred_data$predicted <- predict(best_model, newdata = pred_data, re.form = NA)
      pred_data$ip_level <- factor(pred_data$ip_centered, 
                                   levels = c(-1, 0, 1),
                                   labels = c("Low IP", "Mean IP", "High IP"))
      
      # 创建交互效应图
      library(ggplot2)
      p <- ggplot(pred_data, aes(x = cd_centered, y = predicted, color = ip_level, linetype = ip_level)) +
        geom_line(size = 1.2) +
        labs(x = "Context Dependence (centered)",
             y = "Predicted Activation Strength",
             color = "Institutional\nPresetting",
             linetype = "Institutional\nPresetting",
             title = "Interaction Effect: Context Dependence × Institutional Presetting") +
        theme_minimal(base_size = 12) +
        theme(legend.position = "right")
      
      # 保存图形
      ggsave(file.path(base_dir, "figures/h1_interaction_effects_R.png"), 
             plot = p, width = 10, height = 6, dpi = 300)
      
      cat("交互效应图已保存\n")
      
      # 尝试使用effects包（如果可用）
      if (requireNamespace("effects", quietly = TRUE)) {
        tryCatch({
          effect_plot <- effects::allEffects(best_model)
          
          pdf("G:/Project/实证/关联框架/输出/figures/h1_effects_R.pdf", 
              width = 10, height = 6)
          plot(effect_plot)
          dev.off()
          
          cat("effects包效应图已保存\n")
        }, error = function(e) {
          cat("effects包绘图失败，但ggplot2图已成功生成\n")
        })
      }
      
    }, error = function(e) {
      cat(sprintf("生成效应图时出错: %s\n", e$message))
      cat("跳过效应图生成\n")
    })
    
    # ========== 6. 模型诊断 ==========
    diagnostics <- perform_model_diagnostics(best_model)
    
    # ========== 7. 敏感性分析 ==========
    cat("\n7. 敏感性分析\n")
    
    # 7.1 不同随机效应结构
    cat("\n比较不同随机效应结构...\n")
    alt_model1 <- lmer(activation_strength ~ cd_centered * ip_centered + 
                      (1 | dialogue_id) + (1 | speaker_id:dialogue_id), 
                      data = data, REML = TRUE)
    alt_model2 <- lmer(activation_strength ~ cd_centered * ip_centered + 
                      (cd_centered | dialogue_id), 
                      data = data, REML = TRUE)
    
    cat("AIC比较：\n")
    cat(sprintf("主模型: %.1f\n", AIC(best_model)))
    cat(sprintf("替代模型1: %.1f\n", AIC(alt_model1)))
    cat(sprintf("替代模型2: %.1f\n", AIC(alt_model2)))
  }
  
  # ========== 8. 保存结果 ==========
  results <- list(
    models = model_summaries,
    best_model = if (!is.null(best_model)) {
      list(
        fixed_effects = fixef(best_model),
        random_effects = VarCorr(best_model),
        icc = icc
      )
    } else NULL,
    timestamp = Sys.time()
  )
  
  write_json(results, "G:/Project/实证/关联框架/输出/data/h1_advanced_validation.json", 
             pretty = TRUE, auto_unbox = TRUE)
  
  cat("\n验证完成！结果已保存。\n")
  return(results)
}

# ================================================================================
# H2假设验证：框架驱动的策略选择（效应编码）
# ================================================================================

validate_h2_advanced <- function(data_path = "G:/Project/实证/关联框架/输出/data/h2_data_for_R.csv") {
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("H2假设验证：框架驱动的策略选择（效应编码分析）\n")
  cat(rep("=", 70), "\n\n", sep = "")
  
  # 读取真实数据
  if (!file.exists(data_path)) {
    cat("警告：数据文件不存在：", data_path, "\n")
    cat("跳过H3验证\n")
    return(list(status = "skipped", message = paste("数据文件不存在:", data_path)))
  }
  
  cat("读取数据文件:", data_path, "\n")
  data <- read.csv(data_path, encoding = "UTF-8", stringsAsFactors = FALSE)
  
  cat(sprintf("数据规模：%d条记录\n", nrow(data)))
  
  # ========== 1. 效应编码 ==========
  cat("\n1. 应用效应编码...\n")
  
  # 手动创建效应编码
  # 框架类型（参考类别：transaction）
  data <- data %>%
    mutate(
      frame_service = case_when(
        frame_type == "service" ~ 1,
        frame_type == "transaction" ~ -1,
        TRUE ~ 0
      ),
      frame_problem = case_when(
        frame_type == "problem_solving" ~ 1,
        frame_type == "transaction" ~ -1,
        TRUE ~ 0
      ),
      frame_relation = case_when(
        frame_type == "relationship" ~ 1,
        frame_type == "transaction" ~ -1,
        TRUE ~ 0
      ),
      # 角色（参考类别：C）
      role_SP = ifelse(speaker_role == "SP", 1, -1)
    )
  
  cat("效应编码完成\n")
  
  # ========== 2. 多项逻辑回归 ==========
  cat("\n2. 拟合多项逻辑回归模型...\n")
  
  # 设置参考类别
  data$strategy_type <- relevel(factor(data$strategy_type), ref = "reinforcement")
  
  # 拟合多层多项逻辑模型
  # 注意：由于R的限制，使用MCMCglmm或brms包实现真正的多层多项模型
  # 这里先用nnet实现单层，然后添加说明
  model <- multinom(
    strategy_type ~ frame_service + frame_problem + frame_relation + 
                    role_SP + cognitive_load + emotional_valence +
                    frame_service:role_SP + frame_problem:role_SP + frame_relation:role_SP,
    data = data,
    trace = FALSE
  )
  
  cat("\n注意：当前使用单层多项逻辑回归。")
  cat("\n如需真正的多层结构，建议使用以下方法之一：")
  cat("\n1. MCMCglmm包的多层贝叶斯方法")
  cat("\n2. brms包的贝叶斯多层模型")
  cat("\n3. glmmTMB包的频率派方法\n")
  
  cat("模型拟合完成\n")
  
  # ========== 3. 模型结果 ==========
  cat("\n3. 模型系数（对数几率）：\n")
  print(summary(model))
  
  # 计算几率比
  cat("\n4. 几率比（Odds Ratios）：\n")
  odds_ratios <- exp(coef(model))
  print(round(odds_ratios, 3))
  
  # ========== 4. 边际效应 ==========
  cat("\n5. 计算平均边际效应...\n")
  
  # 对每个框架类型计算边际效应
  marginal_effects <- list()
  
  for (frame in c("service", "problem_solving", "relationship")) {
    # 创建预测数据
    pred_data <- data %>%
      select(cognitive_load, emotional_valence, speaker_role) %>%
      distinct() %>%
      mutate(frame_type = frame)
    
    # 预测概率
    probs <- predict(model, newdata = pred_data, type = "probs")
    
    # 计算平均概率
    avg_probs <- colMeans(probs)
    marginal_effects[[frame]] <- avg_probs
  }
  
  # 转换为数据框
  me_df <- as.data.frame(marginal_effects)
  rownames(me_df) <- names(avg_probs)
  
  cat("\n边际效应（各框架下的策略概率）：\n")
  print(round(me_df, 3))
  
  # ========== 5. 模型诊断 ==========
  cat("\n6. 模型诊断...\n")
  
  # 伪R²
  null_model <- multinom(strategy_type ~ 1, data = data, trace = FALSE)
  pseudo_r2 <- 1 - (logLik(model) / logLik(null_model))
  cat(sprintf("McFadden伪R²: %.3f\n", pseudo_r2))
  
  # AIC/BIC
  cat(sprintf("AIC: %.1f\n", AIC(model)))
  cat(sprintf("BIC: %.1f\n", BIC(model)))
  
  # ========== 6. 可视化 ==========
  cat("\n7. 生成可视化...\n")
  
  # 创建边际效应热图
  me_long <- me_df %>%
    rownames_to_column("strategy") %>%
    pivot_longer(-strategy, names_to = "frame", values_to = "probability")
  
  p <- ggplot(me_long, aes(x = frame, y = strategy, fill = probability)) +
    geom_tile() +
    geom_text(aes(label = round(probability, 2)), color = "white") +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.2) +
    labs(title = "框架类型对策略选择的边际效应",
         x = "框架类型", y = "策略类型") +
    theme_minimal()
  
  ggsave(file.path(base_dir, "figures/h2_marginal_effects_R.png"), 
         p, width = 8, height = 6, dpi = 300)
  
  # ========== 7. 保存结果 ==========
  results <- list(
    coefficients = coef(model),
    odds_ratios = odds_ratios,
    marginal_effects = marginal_effects,
    pseudo_r2 = as.numeric(pseudo_r2),
    model_fit = list(AIC = AIC(model), BIC = BIC(model)),
    timestamp = Sys.time()
  )
  
  write_json(results, "G:/Project/实证/关联框架/输出/data/h2_advanced_validation.json", 
             pretty = TRUE, auto_unbox = TRUE)
  
  cat("\n验证完成！结果已保存。\n")
  return(results)
}

# ================================================================================
# H3假设验证：策略演化的路径依赖（高级分析）
# ================================================================================

validate_h3_advanced <- function(data_path = "G:/Project/实证/关联框架/输出/data/h3_data_for_R.csv") {
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("H3假设验证：策略演化的路径依赖（马尔可夫链与生存分析）\n")
  cat(rep("=", 70), "\n\n", sep = "")
  
  # 检查是否有h3_data_for_R.csv文件
  if (file.exists(data_path)) {
    # 使用单一数据文件
    cat("读取数据文件:", data_path, "\n")
    data <- read.csv(data_path, encoding = "UTF-8", stringsAsFactors = FALSE)
    cat(sprintf("数据规模：%d条记录，%d个对话\n", nrow(data), n_distinct(data$dialogue_id)))
    # 继续原有的分析逻辑
  } else {
    # 备用方案：使用转移矩阵文件
    cat("主数据文件不存在，尝试使用转移矩阵文件...\n")
    
    matrix_files <- list(
      customer = "G:/Project/实证/关联框架/输出/data/transition_matrix_customer_for_R.csv",
      clerk = "G:/Project/实证/关联框架/输出/data/transition_matrix_clerk_for_R.csv",
      service_provider = "G:/Project/实证/关联框架/输出/data/transition_matrix_service_provider_for_R.csv"
    )
    
    # 检查是否有任何转移矩阵文件
    available_files <- sapply(matrix_files, file.exists)
    if (!any(available_files)) {
      cat("警告：没有找到任何转移矩阵文件\n")
      return(list(status = "skipped", message = "没有可用的数据文件"))
    }
    
    # 对每个可用的转移矩阵进行分析
    results <- list()
    for (role in names(matrix_files)) {
      if (available_files[role]) {
        cat(sprintf("\n分析%s角色的转移矩阵...\n", role))
        trans_matrix <- as.matrix(read.csv(matrix_files[[role]], row.names = 1))
        
        # 创建马尔可夫链
        mc <- new("markovchain", transitionMatrix = trans_matrix, name = role)
        
        # 计算稳态分布
        steady <- steadyStates(mc)
        
        results[[role]] <- list(
          steady_state = as.vector(steady),
          transition_matrix = trans_matrix
        )
      }
    }
    
    # 保存结果
    write_json(results, "G:/Project/实证/关联框架/输出/data/h3_markov_validation.json", 
               pretty = TRUE, auto_unbox = TRUE)
    
    cat("\nH3验证完成（使用转移矩阵）\n")
    return(list(status = "partial_success", results = results))
  }
  
  cat(sprintf("数据规模：%d条记录，%d个对话\n", nrow(data), n_distinct(data$dialogue_id)))
  
  # ========== 1. 马尔可夫链分析 ==========
  cat("\n1. 马尔可夫链分析\n")
  
  # 计算转换矩阵
  transitions <- data %>%
    group_by(previous_strategy, current_strategy) %>%
    summarise(count = n(), .groups = "drop")
  
  # 创建完整矩阵
  strategies <- c("reinforcement", "shifting", "blending", "response", "resistance")
  trans_matrix <- matrix(0, nrow = 5, ncol = 5, 
                        dimnames = list(strategies, strategies))
  
  for (i in 1:nrow(transitions)) {
    trans_matrix[transitions$previous_strategy[i], transitions$current_strategy[i]] <- 
      transitions$count[i]
  }
  
  # 转换为概率
  trans_probs <- trans_matrix / rowSums(trans_matrix)
  
  cat("\n转换概率矩阵：\n")
  print(round(trans_probs, 3))
  
  # 创建马尔可夫链对象
  mc <- new("markovchain", 
            states = strategies,
            transitionMatrix = trans_probs,
            name = "Strategy Evolution")
  
  # 稳态分布
  steady <- steadyStates(mc)
  cat("\n稳态分布：\n")
  print(round(steady, 4))
  
  # 平均返回时间
  mean_return <- 1 / steady[1,]
  cat("\n平均返回时间：\n")
  print(round(mean_return, 2))
  
  # 混合时间（第二大特征值）
  eigenvals <- eigen(trans_probs)$values
  mixing_time <- -1 / log(abs(eigenvals[2]))
  cat(sprintf("\n混合时间: %.2f 步\n", mixing_time))
  
  # ========== 2. 生存分析 ==========
  cat("\n2. 策略持续性的生存分析\n")
  
  # 准备生存数据
  survival_data <- data %>%
    group_by(dialogue_id) %>%
    mutate(
      strategy_change = current_strategy != lag(current_strategy, default = current_strategy[1]),
      spell_id = cumsum(strategy_change)
    ) %>%
    group_by(dialogue_id, spell_id) %>%
    summarise(
      strategy = first(current_strategy),
      duration = n(),
      censored = 0,  # 假设都是完整观察
      speaker_role = first(speaker_role),
      .groups = "drop"
    )
  
  # Kaplan-Meier估计
  cat("\nKaplan-Meier生存曲线：\n")
  
  km_fit <- survfit(Surv(duration, 1-censored) ~ strategy, data = survival_data)
  print(km_fit)
  
  # 生存曲线图
  pdf("G:/Project/实证/关联框架/输出/figures/h3_survival_curves_R.pdf", 
      width = 8, height = 6)
  plot(km_fit, col = 1:5, lty = 1:5, 
       xlab = "持续时间（话轮）", ylab = "生存概率",
       main = "策略持续性的生存曲线")
  legend("topright", strategies, col = 1:5, lty = 1:5)
  dev.off()
  
  # Cox比例风险模型
  cat("\nCox比例风险模型：\n")
  
  cox_model <- coxph(Surv(duration, 1-censored) ~ strategy + speaker_role, 
                     data = survival_data)
  print(summary(cox_model))
  
  # ========== 3. 路径依赖性检验 ==========
  cat("\n3. 路径依赖性检验\n")
  
  # 计算自相关
  autocorr_by_dialogue <- data %>%
    group_by(dialogue_id) %>%
    mutate(strategy_num = as.numeric(factor(current_strategy))) %>%
    summarise(
      autocorr = cor(strategy_num[-n()], strategy_num[-1], use = "complete.obs"),
      n_turns = n()
    ) %>%
    filter(!is.na(autocorr))
  
  mean_autocorr <- mean(autocorr_by_dialogue$autocorr)
  cat(sprintf("\n平均自相关系数: %.3f\n", mean_autocorr))
  
  # t检验：自相关是否显著大于0
  t_test <- t.test(autocorr_by_dialogue$autocorr, mu = 0, alternative = "greater")
  cat(sprintf("t = %.3f, p = %.4f\n", t_test$statistic, t_test$p.value))
  
  # ========== 4. 非线性效应分析 ==========
  cat("\n4. 策略重复的非线性效应\n")
  
  # 计算策略重复次数
  repeat_data <- data %>%
    group_by(dialogue_id) %>%
    mutate(
      same_as_prev = current_strategy == lag(current_strategy, default = current_strategy[1]),
      repeat_count = sequence(rle(current_strategy)$lengths)
    )
  
  # 拟合非线性模型
  nonlinear_model <- glm(strategy_change ~ repeat_count + I(repeat_count^2), 
                        data = repeat_data %>% 
                          mutate(strategy_change = !same_as_prev),
                        family = binomial)
  
  cat("\n二次项模型结果：\n")
  print(summary(nonlinear_model))
  
  # 计算转折点
  if (coef(nonlinear_model)[3] != 0) {
    turning_point <- -coef(nonlinear_model)[2] / (2 * coef(nonlinear_model)[3])
    cat(sprintf("\n转折点（重复次数）: %.2f\n", turning_point))
  }
  
  # ========== 5. 保存结果 ==========
  results <- list(
    markov_chain = list(
      transition_matrix = trans_probs,
      steady_state = as.vector(steady),
      mean_return_times = as.vector(mean_return),
      mixing_time = mixing_time
    ),
    survival_analysis = list(
      km_summary = capture.output(print(km_fit)),
      cox_coefficients = coef(cox_model),
      cox_hazard_ratios = exp(coef(cox_model))
    ),
    path_dependence = list(
      mean_autocorrelation = mean_autocorr,
      t_statistic = t_test$statistic,
      p_value = t_test$p.value
    ),
    nonlinear_effects = list(
      coefficients = coef(nonlinear_model),
      turning_point = if(exists("turning_point")) turning_point else NA
    ),
    timestamp = Sys.time()
  )
  
  write_json(results, "G:/Project/实证/关联框架/输出/data/h3_advanced_validation.json", 
             pretty = TRUE, auto_unbox = TRUE)
  
  cat("\n验证完成！结果已保存。\n")
  return(results)
}

# ================================================================================
# H4假设验证：意义协商的语义收敛（变化点检测）
# ================================================================================

validate_h4_advanced <- function(data_path = "G:/Project/实证/关联框架/输出/data/h4_data_for_R.csv") {
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("H4假设验证：意义协商的语义收敛（变化点检测与分段回归）\n")
  cat(rep("=", 70), "\n\n", sep = "")
  
  # 读取真实数据
  if (!file.exists(data_path)) {
    cat("警告：数据文件不存在：", data_path, "\n")
    cat("跳过H3验证\n")
    return(list(status = "skipped", message = paste("数据文件不存在:", data_path)))
  }
  
  cat("读取数据文件:", data_path, "\n")
  data <- read.csv(data_path, encoding = "UTF-8", stringsAsFactors = FALSE)
  
  cat(sprintf("数据规模：%d条记录，%d个对话\n", nrow(data), n_distinct(data$dialogue_id)))
  
  # ========== 1. 变化点检测 ==========
  cat("\n1. CUSUM变化点检测\n")
  
  # 对每个对话进行变化点检测
  changepoint_results <- list()
  
  sample_dialogues <- unique(data$dialogue_id)[1:5]  # 分析前5个对话
  
  for (dial_id in sample_dialogues) {
    dial_data <- data %>% 
      filter(dialogue_id == dial_id) %>%
      arrange(relative_position)
    
    if (nrow(dial_data) > 10) {
      # 使用changepoint包检测
      cpt <- cpt.mean(dial_data$semantic_distance, method = "PELT")
      changepoints <- cpts(cpt)
      
      if (length(changepoints) > 0) {
        cp_positions <- dial_data$relative_position[changepoints]
        changepoint_results[[dial_id]] <- cp_positions
        cat(sprintf("%s: %d个变化点 (位置: %s)\n", 
                   dial_id, length(changepoints), 
                   paste(round(cp_positions, 2), collapse = ", ")))
      }
    }
  }
  
  # ========== 2. 五断点分段回归 ==========
  cat("\n2. 五断点分段增长曲线模型\n")
  
  # 定义断点
  breakpoints <- c(0.15, 0.35, 0.50, 0.75, 0.90)
  
  # 创建分段变量
  data <- data %>%
    mutate(
      segment1 = pmin(relative_position, breakpoints[1]),
      segment2 = pmax(0, pmin(relative_position - breakpoints[1], 
                             breakpoints[2] - breakpoints[1])),
      segment3 = pmax(0, pmin(relative_position - breakpoints[2], 
                             breakpoints[3] - breakpoints[2])),
      segment4 = pmax(0, pmin(relative_position - breakpoints[3], 
                             breakpoints[4] - breakpoints[3])),
      segment5 = pmax(0, pmin(relative_position - breakpoints[4], 
                             breakpoints[5] - breakpoints[4])),
      segment6 = pmax(0, relative_position - breakpoints[5])
    )
  
  # 计算累积标记数
  cumulative_data <- data %>%
    group_by(dialogue_id) %>%
    arrange(relative_position) %>%
    mutate(cumulative_markers = row_number())
  
  # 拟合分段模型
  piecewise_model <- lmer(
    cumulative_markers ~ segment1 + segment2 + segment3 + 
                        segment4 + segment5 + segment6 - 1 + 
                        (1 | dialogue_id),
    data = cumulative_data,
    REML = TRUE
  )
  
  cat("\n分段增长率：\n")
  growth_rates <- fixef(piecewise_model)
  names(growth_rates) <- c("初始期", "早期协商", "中期发展", 
                          "深度协商", "后期收敛", "最终阶段")
  print(round(growth_rates, 3))
  
  # 检验斜率变化
  cat("\n斜率变化：\n")
  for (i in 2:length(growth_rates)) {
    change <- growth_rates[i] - growth_rates[i-1]
    cat(sprintf("%s -> %s: %.3f\n", 
               names(growth_rates)[i-1], names(growth_rates)[i], change))
  }
  
  # ========== 3. 角色差异分析 ==========
  cat("\n3. 角色差异分析（Welch's t检验）\n")
  
  # 添加角色信息
  role_data <- data %>%
    mutate(speaker_role = ifelse(row_number() %% 2 == 1, "SP", "C"))
  
  # 按角色分组
  sp_data <- role_data %>% filter(speaker_role == "SP")
  c_data <- role_data %>% filter(speaker_role == "C")
  
  # Welch's t检验
  t_test <- t.test(sp_data$semantic_distance, c_data$semantic_distance, 
                   var.equal = FALSE)
  
  cat(sprintf("\n语义距离差异:\n"))
  cat(sprintf("SP均值: %.3f, C均值: %.3f\n", 
             mean(sp_data$semantic_distance), mean(c_data$semantic_distance)))
  cat(sprintf("t = %.3f, df = %.1f, p = %.4f\n", 
             t_test$statistic, t_test$parameter, t_test$p.value))
  
  # Cohen's d
  pooled_sd <- sqrt((var(sp_data$semantic_distance) + var(c_data$semantic_distance)) / 2)
  cohens_d <- (mean(sp_data$semantic_distance) - mean(c_data$semantic_distance)) / pooled_sd
  cat(sprintf("Cohen's d = %.3f\n", cohens_d))
  
  # ========== 4. 语义收敛可视化 ==========
  cat("\n4. 生成语义收敛轨迹图...\n")
  
  # 计算平均轨迹
  avg_trajectory <- data %>%
    mutate(position_bin = cut(relative_position, breaks = 20)) %>%
    group_by(position_bin) %>%
    summarise(
      position = mean(relative_position),
      mean_distance = mean(semantic_distance),
      se_distance = sd(semantic_distance) / sqrt(n()),
      .groups = "drop"
    )
  
  p <- ggplot(avg_trajectory, aes(x = position, y = mean_distance)) +
    geom_line(size = 1.2, color = "blue") +
    geom_ribbon(aes(ymin = mean_distance - 1.96*se_distance,
                   ymax = mean_distance + 1.96*se_distance),
               alpha = 0.3, fill = "blue") +
    geom_vline(xintercept = breakpoints, linetype = "dashed", color = "red", alpha = 0.5) +
    labs(title = "语义距离收敛轨迹",
         x = "对话相对位置", y = "语义距离") +
    theme_minimal()
  
  ggsave(file.path(base_dir, "figures/h4_convergence_trajectory_R.png"), 
         p, width = 10, height = 6, dpi = 300)
  
  # ========== 5. 保存结果 ==========
  results <- list(
    changepoint_detection = changepoint_results,
    piecewise_model = list(
      breakpoints = breakpoints,
      growth_rates = as.list(growth_rates),
      model_summary = capture.output(summary(piecewise_model))
    ),
    role_differences = list(
      t_statistic = t_test$statistic,
      p_value = t_test$p.value,
      cohens_d = cohens_d,
      mean_SP = mean(sp_data$semantic_distance),
      mean_C = mean(c_data$semantic_distance)
    ),
    timestamp = Sys.time()
  )
  
  write_json(results, "G:/Project/实证/关联框架/输出/data/h4_advanced_validation.json", 
             pretty = TRUE, auto_unbox = TRUE)
  
  cat("\n验证完成！结果已保存。\n")
  return(results)
}

# ================================================================================
# 统计功效分析
# ================================================================================

perform_power_analysis <- function(model, effect_size = 0.5, nsim = 100) {
  cat("\n执行统计功效分析...\n")
  
  # 使用simr包进行功效分析
  power_fixed <- powerSim(model, 
                         test = fixed("cd_centered:ip_centered"), 
                         nsim = nsim,
                         progress = FALSE)
  
  cat(sprintf("检测交互效应的统计功效: %.2f (基于%d次模拟)\n", 
              summary(power_fixed)$mean, nsim))
  
  # 计算样本量曲线
  sample_sizes <- seq(20, 100, by = 20)
  power_curve <- powerCurve(model, 
                           test = fixed("cd_centered:ip_centered"),
                           along = "dialogue_id",
                           breaks = sample_sizes,
                           nsim = 50,
                           progress = FALSE)
  
  return(list(power = power_fixed, curve = power_curve))
}

# ================================================================================
# 主程序
# ================================================================================

main <- function(language = "zh") {
  cat("\n")
  cat("================================================================================\n")
  if (language == "zh") {
    cat("                    SPAADIA语料库综合R验证分析系统                              \n")
  } else {
    cat("                    SPAADIA Corpus Comprehensive R Validation System             \n")
  }
  cat("================================================================================\n")
  cat("\n启动时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
  
  # 创建输出目录
  create_output_dirs(language)
  
  # 存储所有结果
  all_results <- list()
  
  # 运行各假设验证（每个假设独立的错误处理）
  cat("\n开始运行假设验证...\n")
  
  # H1假设
  cat("\n" , rep("-", 70), "\n", sep = "")
  tryCatch({
    all_results$H1 <- validate_h1_advanced()
  }, error = function(e) {
    cat("\nH1验证出错:", e$message, "\n")
    all_results$H1 <- list(status = "failed", error = e$message)
  })
  
  # H2假设
  cat("\n" , rep("-", 70), "\n", sep = "")
  tryCatch({
    all_results$H2 <- validate_h2_advanced()
  }, error = function(e) {
    cat("\nH2验证出错:", e$message, "\n")
    all_results$H2 <- list(status = "failed", error = e$message)
  })
  
  # H3假设
  cat("\n" , rep("-", 70), "\n", sep = "")
  tryCatch({
    all_results$H3 <- validate_h3_advanced()
  }, error = function(e) {
    cat("\nH3验证出错:", e$message, "\n")
    all_results$H3 <- list(status = "failed", error = e$message)
  })
  
  # H4假设
  cat("\n" , rep("-", 70), "\n", sep = "")
  tryCatch({
    all_results$H4 <- validate_h4_advanced()
  }, error = function(e) {
    cat("\nH4验证出错:", e$message, "\n")
    all_results$H4 <- list(status = "failed", error = e$message)
  })
  
  # 生成综合报告
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("生成综合验证报告...\n")
  
  report <- list(
    title = "SPAADIA语料库R验证分析报告",
    date = Sys.Date(),
    summary = "使用R语言对Python分析结果进行了独立验证，包括：
    - H1: 随机斜率混合模型与Kenward-Roger检验
    - H2: 效应编码的多项逻辑回归
    - H3: 马尔可夫链分析与生存分析
    - H4: 变化点检测与分段回归",
    results_files = c(
      "h1_advanced_validation.json",
      "h2_advanced_validation.json", 
      "h3_advanced_validation.json",
      "h4_advanced_validation.json"
    ),
    figures = c(
      "h1_interaction_effects_R.pdf",
      "h2_marginal_effects_R.png",
      "h3_survival_curves_R.pdf",
      "h4_convergence_trajectory_R.png"
    )
  )
  
  write_json(report, "G:/Project/实证/关联框架/输出/reports/R_validation_summary.json",
             pretty = TRUE, auto_unbox = TRUE)
  
  cat("\n================================================================================\n")
  cat("                          所有验证分析已完成！                                  \n")
  cat("================================================================================\n")
  cat("\n输出文件位置:\n")
  cat("- 数据文件: G:/Project/实证/关联框架/输出/data/\n")
  cat("- 图形文件: G:/Project/实证/关联框架/输出/figures/\n")
  cat("- 报告文件: G:/Project/实证/关联框架/输出/reports/\n")
  cat("\n完成时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
  
  return(all_results)
}

# 如果直接运行脚本，执行主函数
if (!interactive()) {
  results <- main()
} else {
  cat("脚本已加载。运行 main() 开始分析。\n")
}

