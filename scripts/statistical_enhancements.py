#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计增强功能模块
为H2-H4假设验证脚本提供高标准统计分析支持
"""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class StatisticalEnhancements:
    """统计增强功能类，提供高质量统计分析所需的各种方法"""
    
    @staticmethod
    def cohens_d_with_ci(group1: np.ndarray, group2: np.ndarray, 
                        paired: bool = False, confidence: float = 0.95) -> Dict[str, float]:
        """
        计算Cohen's d效应量及其置信区间
        
        Parameters:
        -----------
        group1 : np.ndarray
            第一组数据
        group2 : np.ndarray
            第二组数据
        paired : bool
            是否为配对样本
        confidence : float
            置信水平（默认0.95）
            
        Returns:
        --------
        Dict包含d值、置信区间、解释等
        """
        n1 = len(group1)
        n2 = len(group2)
        
        # 处理空组的情况
        if n1 == 0 or n2 == 0:
            return {
                'd': 0,
                'ci_lower': 0,
                'ci_upper': 0,
                'interpretation': '无法计算（样本量不足）'
            }
        
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        
        if paired:
            # 配对样本
            if n1 != n2:
                raise ValueError("配对样本的样本量必须相等")
            diff = group1 - group2
            d = np.mean(diff) / np.std(diff, ddof=1)
            # 配对样本的标准误
            se_d = 1 / np.sqrt(n1)
        else:
            # 独立样本
            std1 = np.std(group1, ddof=1)
            std2 = np.std(group2, ddof=1)
            
            # 合并标准差
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            # Cohen's d
            d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # 标准误（Hedges & Olkin, 1985）
            if n1 * n2 > 0:
                se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
            else:
                se_d = 0
        
        # 计算置信区间
        z = stats.norm.ppf((1 + confidence) / 2)
        ci_lower = d - z * se_d
        ci_upper = d + z * se_d
        
        # 解释效应量大小
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return {
            'd': d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'se': se_d,
            'interpretation': interpretation,
            'n1': n1,
            'n2': n2
        }
    
    @staticmethod
    def fdr_correction(p_values: List[float], alpha: float = 0.05, 
                       method: str = 'fdr_bh') -> Dict[str, any]:
        """
        应用FDR多重比较校正（Benjamini-Hochberg方法）
        
        Parameters:
        -----------
        p_values : List[float]
            原始p值列表
        alpha : float
            显著性水平
        method : str
            校正方法，默认'fdr_bh'
            
        Returns:
        --------
        Dict包含校正后的p值、拒绝原假设的布尔值等
        """
        p_array = np.array(p_values)
        
        # 处理NaN值
        valid_mask = ~np.isnan(p_array)
        valid_p = p_array[valid_mask]
        
        if len(valid_p) == 0:
            return {
                'p_adjusted': p_array,
                'rejected': np.zeros(len(p_array), dtype=bool),
                'alpha_adjusted': alpha,
                'n_significant': 0,
                'n_tests': len(p_array)
            }
        
        # 应用FDR校正
        rejected, p_adjusted_valid, alpha_sidak, alpha_bonf = multipletests(
            valid_p, alpha=alpha, method=method, returnsorted=False
        )
        
        # 重建完整数组
        p_adjusted = np.full(len(p_array), np.nan)
        p_adjusted[valid_mask] = p_adjusted_valid
        
        rejected_full = np.zeros(len(p_array), dtype=bool)
        rejected_full[valid_mask] = rejected
        
        return {
            'p_adjusted': p_adjusted.tolist(),
            'rejected': rejected_full.tolist(),
            'alpha_adjusted': alpha,
            'alpha_sidak': alpha_sidak,
            'alpha_bonferroni': alpha_bonf,
            'n_significant': int(np.sum(rejected)),
            'n_tests': len(valid_p),
            'method': method
        }
    
    @staticmethod
    def calculate_pseudo_r2(llf: float, llnull: float, n: int) -> Dict[str, float]:
        """
        计算各种伪R²度量（用于逻辑回归等GLM模型）
        
        Parameters:
        -----------
        llf : float
            完整模型的对数似然
        llnull : float
            零模型的对数似然
        n : int
            样本量
            
        Returns:
        --------
        Dict包含McFadden's, Cox-Snell, Nagelkerke R²
        """
        # McFadden's R²
        mcfadden = 1 - (llf / llnull) if llnull != 0 else 0
        
        # Cox-Snell R²
        cox_snell = 1 - np.exp((llnull - llf) * (2/n)) if n > 0 else 0
        
        # Nagelkerke R²
        max_cox_snell = 1 - np.exp(llnull * (2/n)) if n > 0 else 1
        nagelkerke = cox_snell / max_cox_snell if max_cox_snell > 0 else 0
        
        return {
            'mcfadden': mcfadden,
            'cox_snell': cox_snell,
            'nagelkerke': nagelkerke
        }
    
    @staticmethod
    def calculate_eta_squared(f_statistic: float, df1: int, df2: int) -> Dict[str, float]:
        """
        计算η²效应量及其偏η²
        
        Parameters:
        -----------
        f_statistic : float
            F统计量
        df1 : int
            分子自由度
        df2 : int
            分母自由度
            
        Returns:
        --------
        Dict包含η²和偏η²
        """
        # η²
        eta_squared = (f_statistic * df1) / (f_statistic * df1 + df2)
        
        # 偏η²（用于重复测量或多因素ANOVA）
        partial_eta_squared = (f_statistic * df1) / (f_statistic * df1 + df2)
        
        return {
            'eta_squared': eta_squared,
            'partial_eta_squared': partial_eta_squared
        }
    
    @staticmethod
    def calculate_cramers_v(chi2: float, n: int, r: int, c: int) -> Dict[str, float]:
        """
        计算Cramér's V效应量及其置信区间
        
        Parameters:
        -----------
        chi2 : float
            卡方统计量
        n : int
            样本量
        r : int
            行数
        c : int
            列数
            
        Returns:
        --------
        Dict包含Cramér's V值和解释
        """
        # Cramér's V
        min_dim = min(r - 1, c - 1)
        v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0
        
        # 解释（基于Cohen's建议）
        if min_dim == 1:  # 2x2表
            if v < 0.1:
                interpretation = 'negligible'
            elif v < 0.3:
                interpretation = 'small'
            elif v < 0.5:
                interpretation = 'medium'
            else:
                interpretation = 'large'
        elif min_dim == 2:  # 3x3表
            if v < 0.07:
                interpretation = 'negligible'
            elif v < 0.21:
                interpretation = 'small'
            elif v < 0.35:
                interpretation = 'medium'
            else:
                interpretation = 'large'
        else:  # 更大的表
            if v < 0.05:
                interpretation = 'negligible'
            elif v < 0.15:
                interpretation = 'small'
            elif v < 0.25:
                interpretation = 'medium'
            else:
                interpretation = 'large'
        
        return {
            'v': v,
            'interpretation': interpretation,
            'df': min_dim
        }
    
    @staticmethod
    def bootstrap_ci(data: np.ndarray, statistic_func: callable, 
                    n_bootstrap: int = 1000, confidence: float = 0.95,
                    random_state: int = 42) -> Dict[str, float]:
        """
        Bootstrap方法计算置信区间
        
        Parameters:
        -----------
        data : np.ndarray
            原始数据
        statistic_func : callable
            统计量计算函数
        n_bootstrap : int
            Bootstrap样本数
        confidence : float
            置信水平
        random_state : int
            随机种子
            
        Returns:
        --------
        Dict包含点估计和置信区间
        """
        np.random.seed(random_state)
        n = len(data)
        
        # Bootstrap抽样
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, n, replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        # 计算置信区间
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        # 点估计
        point_estimate = statistic_func(data)
        
        return {
            'estimate': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'se': np.std(bootstrap_stats),
            'n_bootstrap': n_bootstrap
        }
    
    @staticmethod
    def format_p_value(p: float) -> str:
        """
        按照APA格式化p值
        
        Parameters:
        -----------
        p : float
            p值
            
        Returns:
        --------
        str: 格式化的p值字符串
        """
        if p < 0.001:
            return "p < .001"
        elif p < 0.01:
            return f"p = {p:.3f}"
        elif p < 0.05:
            return f"p = {p:.3f}"
        else:
            return f"p = {p:.3f}"
    
    @staticmethod
    def interpret_effect_size(effect_type: str, value: float) -> str:
        """
        解释各种效应量的大小
        
        Parameters:
        -----------
        effect_type : str
            效应量类型 ('d', 'r', 'eta', 'v')
        value : float
            效应量值
            
        Returns:
        --------
        str: 效应量解释
        """
        abs_value = abs(value)
        
        if effect_type == 'd':  # Cohen's d
            if abs_value < 0.2:
                return 'negligible'
            elif abs_value < 0.5:
                return 'small'
            elif abs_value < 0.8:
                return 'medium'
            else:
                return 'large'
        
        elif effect_type == 'r':  # 相关系数
            if abs_value < 0.1:
                return 'negligible'
            elif abs_value < 0.3:
                return 'small'
            elif abs_value < 0.5:
                return 'medium'
            else:
                return 'large'
        
        elif effect_type == 'eta':  # η²
            if abs_value < 0.01:
                return 'negligible'
            elif abs_value < 0.06:
                return 'small'
            elif abs_value < 0.14:
                return 'medium'
            else:
                return 'large'
        
        elif effect_type == 'v':  # Cramér's V (2x2)
            if abs_value < 0.1:
                return 'negligible'
            elif abs_value < 0.3:
                return 'small'
            elif abs_value < 0.5:
                return 'medium'
            else:
                return 'large'
        
        else:
            return 'unknown'
    
    @staticmethod
    def calculate_odds_ratio_ci(or_value: float, se_log_or: float, 
                               confidence: float = 0.95) -> Dict[str, float]:
        """
        计算优势比(OR)的置信区间
        
        Parameters:
        -----------
        or_value : float
            优势比值
        se_log_or : float
            log(OR)的标准误
        confidence : float
            置信水平
            
        Returns:
        --------
        Dict包含OR值和置信区间
        """
        z = stats.norm.ppf((1 + confidence) / 2)
        log_or = np.log(or_value) if or_value > 0 else 0
        
        log_ci_lower = log_or - z * se_log_or
        log_ci_upper = log_or + z * se_log_or
        
        ci_lower = np.exp(log_ci_lower)
        ci_upper = np.exp(log_ci_upper)
        
        return {
            'or': or_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'log_or': log_or,
            'se_log_or': se_log_or
        }