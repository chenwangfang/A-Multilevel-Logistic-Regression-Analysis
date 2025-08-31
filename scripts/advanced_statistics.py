#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级统计分析工具模块
提供随机斜率模型、效应编码、Word2Vec、CUSUM检测等高级功能
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import jieba
from typing import Dict, List, Tuple, Optional
import logging
from patsy import dmatrix
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EffectCoding:
    """效应编码（-1, 0, 1）实现"""
    
    @staticmethod
    def encode(series: pd.Series, reference_category: str = None) -> pd.DataFrame:
        """
        对分类变量进行效应编码
        
        参数:
            series: 分类变量序列
            reference_category: 参考类别（如不指定，使用最后一个类别）
        
        返回:
            效应编码的DataFrame
        """
        # 确保输入是pandas Series
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
        elif not isinstance(series, pd.Series):
            series = pd.Series(series)
            
        categories = series.unique()
        n_categories = len(categories)
        
        if reference_category is None:
            reference_category = categories[-1]
        elif reference_category not in categories:
            raise ValueError(f"参考类别 {reference_category} 不在数据中")
        
        # 创建效应编码矩阵
        encoded = pd.DataFrame(index=series.index)
        
        for i, cat in enumerate(categories):
            if cat == reference_category:
                continue
            
            col_name = f"{series.name}_{cat}"
            encoded[col_name] = 0
            
            # 当前类别编码为1
            encoded.loc[series == cat, col_name] = 1
            # 参考类别编码为-1
            encoded.loc[series == reference_category, col_name] = -1
        
        return encoded

class Word2VecSemanticDistance:
    """基于Word2Vec的语义距离计算"""
    
    def __init__(self, corpus: List[str], vector_size: int = 100, window: int = 5):
        """
        初始化Word2Vec模型
        
        参数:
            corpus: 文本语料列表
            vector_size: 词向量维度
            window: 上下文窗口大小
        """
        self.vector_size = vector_size
        self.window = window
        self.model = None
        self._train_model(corpus)
    
    def _train_model(self, corpus: List[str]):
        """训练Word2Vec模型"""
        # 中文分词
        processed_corpus = []
        for text in corpus:
            # 使用jieba分词
            words = list(jieba.cut(text))
            # 英文使用simple_preprocess
            if not any('\u4e00' <= char <= '\u9fff' for char in text):
                words = simple_preprocess(text)
            processed_corpus.append(words)
        
        # 训练模型
        self.model = Word2Vec(
            sentences=processed_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=1,
            workers=4,
            seed=42
        )
    
    def calculate_distance(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义距离
        
        参数:
            text1: 第一个文本
            text2: 第二个文本
        
        返回:
            语义距离（0-1之间，0表示完全相同，1表示完全不同）
        """
        if self.model is None:
            raise ValueError("Word2Vec模型未训练")
        
        # 分词
        words1 = list(jieba.cut(text1)) if any('\u4e00' <= c <= '\u9fff' for c in text1) else simple_preprocess(text1)
        words2 = list(jieba.cut(text2)) if any('\u4e00' <= c <= '\u9fff' for c in text2) else simple_preprocess(text2)
        
        # 获取词向量
        vectors1 = [self.model.wv[w] for w in words1 if w in self.model.wv]
        vectors2 = [self.model.wv[w] for w in words2 if w in self.model.wv]
        
        if not vectors1 or not vectors2:
            return 1.0  # 如果没有共同词汇，返回最大距离
        
        # 计算平均词向量
        mean_vec1 = np.mean(vectors1, axis=0)
        mean_vec2 = np.mean(vectors2, axis=0)
        
        # 计算余弦相似度
        cosine_sim = np.dot(mean_vec1, mean_vec2) / (np.linalg.norm(mean_vec1) * np.linalg.norm(mean_vec2))
        
        # 转换为距离（0-1）
        distance = 1 - (cosine_sim + 1) / 2
        
        return np.clip(distance, 0, 1)

class CUSUMDetection:
    """CUSUM（累积和）变化点检测"""
    
    @staticmethod
    def detect_changepoints(data: np.ndarray, threshold: float = 3.0, 
                          min_segment_length: int = 10) -> List[int]:
        """
        使用CUSUM算法检测变化点
        
        参数:
            data: 时间序列数据
            threshold: 检测阈值（标准差的倍数）
            min_segment_length: 最小段长度
        
        返回:
            变化点位置列表
        """
        n = len(data)
        
        # 标准化数据
        scaler = StandardScaler()
        standardized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # 计算累积和
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        
        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i-1] + standardized[i] - threshold/2)
            cusum_neg[i] = min(0, cusum_neg[i-1] + standardized[i] + threshold/2)
        
        # 检测超过阈值的点
        changepoints = []
        
        # 正向变化
        pos_exceed = np.where(cusum_pos > threshold)[0]
        if len(pos_exceed) > 0:
            # 找到每个超过阈值段的起始点
            segments = np.split(pos_exceed, np.where(np.diff(pos_exceed) > 1)[0] + 1)
            for segment in segments:
                if len(segment) > 0 and segment[0] > min_segment_length:
                    changepoints.append(segment[0])
        
        # 负向变化
        neg_exceed = np.where(cusum_neg < -threshold)[0]
        if len(neg_exceed) > 0:
            segments = np.split(neg_exceed, np.where(np.diff(neg_exceed) > 1)[0] + 1)
            for segment in segments:
                if len(segment) > 0 and segment[0] > min_segment_length:
                    changepoints.append(segment[0])
        
        # 去重并排序
        changepoints = sorted(list(set(changepoints)))
        
        # 确保变化点之间有最小间隔
        filtered_changepoints = []
        for cp in changepoints:
            if not filtered_changepoints or cp - filtered_changepoints[-1] >= min_segment_length:
                filtered_changepoints.append(cp)
        
        return filtered_changepoints

class MultipleImputation:
    """多重插补处理缺失数据"""
    
    def __init__(self, n_imputations: int = 5, random_state: int = 42):
        """
        初始化多重插补
        
        参数:
            n_imputations: 插补次数
            random_state: 随机种子
        """
        self.n_imputations = n_imputations
        self.random_state = random_state
    
    def impute(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        执行多重插补
        
        参数:
            df: 包含缺失值的DataFrame
        
        返回:
            插补后的DataFrame列表
        """
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        imputed_dfs = []
        
        for i in range(self.n_imputations):
            try:
                # 使用BayesianRidge作为估计器，它更稳健
                imputer = IterativeImputer(
                    estimator=BayesianRidge(),
                    random_state=self.random_state + i,
                    max_iter=10,
                    tol=0.001,
                    initial_strategy='mean',  # 使用均值作为初始策略
                    imputation_order='ascending'  # 从缺失值最少的特征开始
                )
                
                # 分离数值和分类变量
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                
                imputed_df = df.copy()
                
                # 对数值变量进行插补
                if len(numeric_cols) > 0:
                    # 检查并移除常数列（避免SVD问题）
                    numeric_data = df[numeric_cols]
                    non_constant_cols = []
                    for col in numeric_cols:
                        if numeric_data[col].nunique() > 1:
                            non_constant_cols.append(col)
                    
                    if len(non_constant_cols) > 0:
                        # 标准化数据以提高数值稳定性
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[non_constant_cols])
                        
                        # 插补
                        imputed_scaled = imputer.fit_transform(scaled_data)
                        
                        # 逆变换
                        imputed_numeric = scaler.inverse_transform(imputed_scaled)
                        imputed_df[non_constant_cols] = imputed_numeric
                
                # 对分类变量使用众数插补
                for col in categorical_cols:
                    if imputed_df[col].isnull().any():
                        # 检查是否是Categorical类型
                        if pd.api.types.is_categorical_dtype(imputed_df[col]):
                            # 获取众数
                            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else None
                            if mode_value is None or mode_value not in imputed_df[col].cat.categories:
                                # 使用第一个类别作为默认值
                                mode_value = imputed_df[col].cat.categories[0]
                            imputed_df[col].fillna(mode_value, inplace=True)
                        else:
                            # 非Categorical类型，直接填充
                            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                            imputed_df[col].fillna(mode_value, inplace=True)
                
                imputed_dfs.append(imputed_df)
                
            except Exception as e:
                print(f"插补迭代 {i+1} 失败: {e}")
                # 如果插补失败，使用简单的均值/众数插补作为后备
                imputed_df = df.copy()
                
                # 数值变量用均值
                for col in numeric_cols:
                    if imputed_df[col].isnull().any():
                        imputed_df[col].fillna(imputed_df[col].mean(), inplace=True)
                
                # 分类变量用众数
                for col in categorical_cols:
                    if imputed_df[col].isnull().any():
                        # 检查是否是Categorical类型
                        if pd.api.types.is_categorical_dtype(imputed_df[col]):
                            # 获取众数
                            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else None
                            if mode_value is None or mode_value not in imputed_df[col].cat.categories:
                                # 使用第一个类别作为默认值
                                mode_value = imputed_df[col].cat.categories[0]
                            imputed_df[col].fillna(mode_value, inplace=True)
                        else:
                            # 非Categorical类型，直接填充
                            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                            imputed_df[col].fillna(mode_value, inplace=True)
                
                imputed_dfs.append(imputed_df)
        
        return imputed_dfs
    
    @staticmethod
    def combine_results(results: List[Dict]) -> Dict:
        """
        组合多重插补的结果（Rubin's rules）
        
        参数:
            results: 每次插补的分析结果列表
        
        返回:
            组合后的结果
        """
        m = len(results)
        
        # 处理字典格式的结果
        if isinstance(results[0]['coefficients'], dict):
            # 获取所有参数名
            param_names = list(results[0]['coefficients'].keys())
            
            # 收集所有参数的系数和标准误
            all_coefs = {param: [] for param in param_names}
            all_ses = {param: [] for param in param_names}
            
            for result in results:
                for param in param_names:
                    all_coefs[param].append(result['coefficients'][param])
                    all_ses[param].append(result['std_errors'][param])
            
            # 将列表转换为数组
            for param in param_names:
                all_coefs[param] = np.array(all_coefs[param])
                all_ses[param] = np.array(all_ses[param])
            
            # 计算组合结果
            combined_coef = {}
            combined_se = {}
            df_dict = {}
            within_var_dict = {}
            between_var_dict = {}
            
            for param in param_names:
                param_coefs = all_coefs[param]
                param_ses = all_ses[param]
                param_vars = param_ses**2
                
                # 计算组合估计
                combined_coef[param] = np.mean(param_coefs)
                
                # 组内方差
                within_var = np.mean(param_vars)
                within_var_dict[param] = within_var
                
                # 组间方差
                between_var = np.var(param_coefs, ddof=1)
                between_var_dict[param] = between_var
                
                # 总方差
                total_var = within_var + (1 + 1/m) * between_var
                
                # 组合标准误
                combined_se[param] = np.sqrt(total_var)
                
                # 自由度（使用Barnard-Rubin调整）
                if total_var > 0:
                    lambda_val = (1 + 1/m) * between_var / total_var
                    if lambda_val > 0:
                        df_dict[param] = (m - 1) / lambda_val**2
                    else:
                        df_dict[param] = 1000  # 大自由度
                else:
                    df_dict[param] = 1000
            
            return {
                'coefficients': combined_coef,
                'std_errors': combined_se,
                'df': df_dict,
                'within_variance': within_var_dict,
                'between_variance': between_var_dict
            }
        else:
            # 原始数组格式处理
            coefficients = np.array([r['coefficients'] for r in results])
            variances = np.array([r['std_errors']**2 for r in results])
            
            # 计算组合估计
            combined_coef = np.mean(coefficients, axis=0)
            
            # 组内方差
            within_var = np.mean(variances, axis=0)
            
            # 组间方差
            between_var = np.var(coefficients, axis=0, ddof=1)
            
            # 总方差
            total_var = within_var + (1 + 1/m) * between_var
            
            # 组合标准误
            combined_se = np.sqrt(total_var)
            
            # 自由度（使用Barnard-Rubin调整）
            lambda_val = (1 + 1/m) * between_var / total_var
            df_old = (m - 1) / lambda_val**2
            
            return {
                'coefficients': combined_coef,
                'std_errors': combined_se,
                'df': df_old,
                'within_variance': within_var,
                'between_variance': between_var
            }

class BootstrapAnalysis:
    """Bootstrap置信区间和假设检验"""
    
    @staticmethod
    def bootstrap_ci(data: np.ndarray, statistic_func, n_bootstrap: int = 1000, 
                    confidence_level: float = 0.95, random_state: int = 42) -> Dict:
        """
        计算Bootstrap置信区间
        
        参数:
            data: 原始数据
            statistic_func: 统计量计算函数
            n_bootstrap: Bootstrap次数
            confidence_level: 置信水平
            random_state: 随机种子
        
        返回:
            包含点估计、置信区间和Bootstrap分布的字典
        """
        np.random.seed(random_state)
        n = len(data)
        
        # 计算原始统计量
        original_stat = statistic_func(data)
        
        # Bootstrap抽样
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            # 有放回抽样
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # 计算置信区间（百分位法）
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        # 计算偏差校正的置信区间（BCa方法）
        try:
            # 偏差校正参数z0
            prop_less = np.mean(bootstrap_stats < original_stat)
            # 避免极端值
            prop_less = np.clip(prop_less, 0.001, 0.999)
            z0 = stats.norm.ppf(prop_less)
            
            # Jackknife估计加速度
            jackknife_stats = []
            for i in range(n):
                jack_sample = np.delete(data, i)
                jackknife_stats.append(statistic_func(jack_sample))
            
            jackknife_mean = np.mean(jackknife_stats)
            jack_var = np.sum((jackknife_mean - jackknife_stats)**2)
            
            if jack_var > 0:
                a = np.sum((jackknife_mean - jackknife_stats)**3) / (6 * jack_var**1.5)
                # 限制a的范围，避免极端值
                a = np.clip(a, -0.5, 0.5)
            else:
                a = 0.0
            
            # BCa置信区间
            z_alpha_low = stats.norm.ppf(alpha/2)
            z_alpha_high = stats.norm.ppf(1 - alpha/2)
            
            # 计算校正后的分位数
            denom_low = 1 - a * (z0 + z_alpha_low)
            denom_high = 1 - a * (z0 + z_alpha_high)
            
            # 避免除以零
            if abs(denom_low) > 0.001 and abs(denom_high) > 0.001:
                alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_low) / denom_low)
                alpha2 = stats.norm.cdf(z0 + (z0 + z_alpha_high) / denom_high)
            else:
                # 退回到基本百分位法
                alpha1 = alpha/2
                alpha2 = 1 - alpha/2
                
            # 确保alpha值在有效范围内且不是NaN
            if np.isnan(alpha1) or np.isnan(alpha2):
                alpha1 = alpha/2
                alpha2 = 1 - alpha/2
            else:
                alpha1 = np.clip(alpha1, 0.0, 1.0)
                alpha2 = np.clip(alpha2, 0.0, 1.0)
                
            bca_ci_lower = np.percentile(bootstrap_stats, alpha1 * 100)
            bca_ci_upper = np.percentile(bootstrap_stats, alpha2 * 100)
            
        except Exception as e:
            # 如果BCa方法失败，使用基本百分位法
            print(f"BCa方法计算失败: {e}，使用基本百分位法")
            bca_ci_lower = ci_lower
            bca_ci_upper = ci_upper
        
        return {
            'estimate': original_stat,
            'ci_percentile': (ci_lower, ci_upper),
            'ci_bca': (bca_ci_lower, bca_ci_upper),
            'bootstrap_distribution': bootstrap_stats,
            'bias': np.mean(bootstrap_stats) - original_stat,
            'std_error': np.std(bootstrap_stats)
        }

class RandomSlopesModel:
    """随机斜率混合效应模型"""
    
    @staticmethod
    def format_random_effects(groups: pd.DataFrame, random_slopes: List[str]) -> str:
        """
        格式化随机效应公式
        
        参数:
            groups: 分组变量DataFrame
            random_slopes: 随机斜率变量列表
        
        返回:
            随机效应公式字符串
        """
        # 构建随机效应公式
        if len(random_slopes) == 0:
            return "1"  # 仅随机截距
        else:
            return "1 + " + " + ".join(random_slopes)
    
    @staticmethod
    def fit_with_convergence_check(model, max_iter: int = 1000) -> Tuple[object, bool]:
        """
        拟合模型并检查收敛性
        
        参数:
            model: 混合效应模型对象
            max_iter: 最大迭代次数
        
        返回:
            拟合结果和收敛标志
        """
        try:
            # 尝试完整模型
            result = model.fit(reml=True, maxiter=max_iter)
            converged = result.converged
            
            if not converged:
                logger.warning("模型未收敛，尝试简化随机效应结构")
                # 可以在这里实现模型简化逻辑
            
            return result, converged
        except Exception as e:
            logger.error(f"模型拟合失败: {e}")
            return None, False

# 导出所有工具类
__all__ = [
    'EffectCoding',
    'Word2VecSemanticDistance',
    'CUSUMDetection',
    'MultipleImputation',
    'BootstrapAnalysis',
    'RandomSlopesModel'
]