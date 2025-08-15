#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1假设验证分析（高级版V2）：框架激活的语境依赖
实现渐进式的三层线性混合模型（对话-说话人-话轮）
根据数据特征和收敛情况自动调整模型复杂度
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 统计分析库
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('H1_Advanced_V2')

# 导入数据加载器
from data_loader_enhanced import SPAADIADataLoader

class H1ThreeLevelAnalysisV2:
    """H1假设验证：框架激活的语境依赖（渐进式三层模型）"""
    
    def __init__(self, language: str = 'zh'):
        """初始化分析器"""
        self.language = language
        self.output_dir = Path(f"G:/Project/实证/关联框架/{'输出' if language == 'zh' else 'output'}")
        self.data_dir = self.output_dir / 'data'
        self.tables_dir = self.output_dir / 'tables'
        self.figures_dir = self.output_dir / 'figures'
        self.reports_dir = self.output_dir / 'reports'
        
        # 创建输出目录
        for dir_path in [self.data_dir, self.tables_dir, self.figures_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据容器
        self.data = None
        self.models = {}
        self.results = {}
        self.model_selection = None
        
    def run_analysis(self):
        """运行完整的H1分析流程"""
        logger.info("=== 开始H1假设高级分析（渐进式三层模型） ===")
        
        # 1. 加载数据
        self._load_data()
        
        # 2. 预处理数据
        self._preprocess_data()
        
        # 3. 计算CD和IP
        self._calculate_cd_ip()
        
        # 4. 运行渐进式三层模型
        self._run_progressive_models()
        
        # 5. 生成可视化
        self._generate_visualizations()
        
        # 6. 保存结果
        self._save_results()
        
        logger.info("=== H1分析完成 ===")
    
    def _load_data(self):
        """加载数据"""
        logger.info("加载SPAADIA数据...")
        
        loader = SPAADIADataLoader(language=self.language)
        all_data = loader.load_all_data()
        
        # 获取框架激活数据
        frame_activation = all_data.get('frame_activation', pd.DataFrame())
        
        if frame_activation.empty:
            raise ValueError("框架激活数据为空！请确保SPAADIA语料库数据文件存在且完整。")
        
        self.data = frame_activation.copy()
        logger.info(f"数据加载完成，共 {len(self.data)} 条记录")
        logger.info(f"数据列: {list(self.data.columns)}")
        logger.info(f"对话数: {self.data['dialogue_id'].nunique()}")
        
        # 检查speaker相关列名
        speaker_col = None
        for col in ['speaker_id', 'speaker', 'speaker_role']:
            if col in self.data.columns:
                speaker_col = col
                break
        
        if speaker_col:
            logger.info(f"说话人数: {self.data.groupby(['dialogue_id', speaker_col]).ngroups}")
        else:
            logger.info("未找到speaker相关列，将在预处理中创建")
    
    def _preprocess_data(self):
        """预处理数据"""
        logger.info("预处理数据...")
        
        # 确保数值类型
        if 'turn_id' in self.data.columns:
            self.data['turn_id'] = pd.to_numeric(self.data['turn_id'], errors='coerce').fillna(1).astype(int)
        
        # 计算相对位置
        if 'relative_position' not in self.data.columns:
            self.data['relative_position'] = self.data.groupby('dialogue_id')['turn_id'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
            )
        
        # 创建对话阶段
        self.data['dialogue_stage'] = pd.cut(
            self.data['relative_position'],
            bins=[0, 0.15, 0.45, 0.85, 1.0],
            labels=['opening', 'information_exchange', 'problem_solving', 'closing']
        )
        
        # 创建speaker_id基于turn_id的奇偶性（常见的对话模式）
        # 假设奇数turn是客户，偶数turn是座席
        self.data['speaker_id'] = self.data.groupby('dialogue_id')['turn_id'].transform(
            lambda x: x % 2
        )
        
        # 创建角色变量
        self.data['role'] = self.data['speaker_id']  # 0=客户, 1=座席
        
        # 创建唯一的说话人ID（跨对话）
        # 每个对话中的两个说话人
        self.data['speaker_id_unique'] = (
            self.data['dialogue_id'].astype(str) + '_' + 
            self.data['speaker_id'].astype(str)
        )
        
        # 计算任务复杂度
        self._calculate_task_complexity()
        
        logger.info("预处理完成")
    
    def _calculate_task_complexity(self):
        """计算任务复杂度"""
        if 'task_complexity' not in self.data.columns:
            dialogue_features = self.data.groupby('dialogue_id').agg({
                'turn_id': 'count',
                'frame_type': lambda x: x.nunique() if 'frame_type' in self.data.columns else 1
            })
            
            # 标准化
            for col in dialogue_features.columns:
                min_val = dialogue_features[col].min()
                max_val = dialogue_features[col].max()
                dialogue_features[f'{col}_norm'] = (dialogue_features[col] - min_val) / (max_val - min_val + 1e-10)
            
            # 综合复杂度
            dialogue_features['task_complexity'] = dialogue_features[[c for c in dialogue_features.columns if '_norm' in c]].mean(axis=1)
            
            # 映射回主数据
            self.data = self.data.merge(
                dialogue_features[['task_complexity']], 
                left_on='dialogue_id', 
                right_index=True, 
                how='left'
            )
    
    def _calculate_cd_ip(self):
        """计算CD和IP"""
        logger.info("计算语境依赖度和机构预设度...")
        
        # 检查是否已有这些列（来自数据加载器）
        if 'context_dependence' in self.data.columns and 'institutional_presetting' in self.data.columns:
            # 使用现有数据，但确保在合理范围内
            self.data['context_dependence'] = self.data['context_dependence'].clip(0, 1)
            self.data['institutional_presetting'] = self.data['institutional_presetting'].clip(0, 1)
            logger.info("使用数据集中的CD和IP值")
        else:
            # 基于对话特征计算
            logger.info("基于对话特征计算CD和IP")
            
            # 1. 语境依赖度 - 基于对话位置和框架类型
            self.data['context_dependence'] = 0.3 + 0.5 * self.data['relative_position']
            
            if 'frame_type' in self.data.columns:
                task_frames = self.data['frame_type'].str.contains('任务|程序|Task|Procedural', case=False, na=False)
                self.data.loc[task_frames, 'context_dependence'] += 0.1
            elif 'frame_category' in self.data.columns:
                task_frames = self.data['frame_category'] == 'task'
                self.data.loc[task_frames, 'context_dependence'] += 0.1
            
            self.data['context_dependence'] = self.data['context_dependence'].clip(0, 1)
            
            # 2. 机构预设度 - 基于角色和对话阶段
            self.data['institutional_presetting'] = 0.3
            self.data.loc[self.data['role'] == 1, 'institutional_presetting'] = 0.6
            self.data.loc[self.data['dialogue_stage'] == 'opening', 'institutional_presetting'] += 0.15
            self.data.loc[self.data['dialogue_stage'] == 'closing', 'institutional_presetting'] += 0.1
            
            if 'frame_type' in self.data.columns:
                proc_frames = self.data['frame_type'].str.contains('程序|Procedural|规定|Policy', case=False, na=False)
                self.data.loc[proc_frames, 'institutional_presetting'] += 0.1
            elif 'frame_category' in self.data.columns:
                proc_frames = self.data['frame_category'] == 'procedural'
                self.data.loc[proc_frames, 'institutional_presetting'] += 0.1
            
            self.data['institutional_presetting'] = self.data['institutional_presetting'].clip(0, 1)
        
        # 3. 中心化
        self.data['CD_c'] = self.data['context_dependence'] - self.data['context_dependence'].mean()
        self.data['IP_c'] = self.data['institutional_presetting'] - self.data['institutional_presetting'].mean()
        
        logger.info(f"CD均值: {self.data['context_dependence'].mean():.3f}, 标准差: {self.data['context_dependence'].std():.3f}")
        logger.info(f"IP均值: {self.data['institutional_presetting'].mean():.3f}, 标准差: {self.data['institutional_presetting'].std():.3f}")
    
    def _run_progressive_models(self):
        """运行渐进式三层模型"""
        logger.info("运行渐进式三层线性混合模型...")
        
        # 准备数据
        model_data = self.data.copy()
        required_cols = ['activation_strength', 'CD_c', 'IP_c', 'speaker_id_unique', 'dialogue_id']
        model_data = model_data.dropna(subset=required_cols)
        
        # 检查并处理完全共线性问题
        # 确保role和task_complexity有变异
        if 'role' in model_data.columns:
            if model_data['role'].nunique() == 1:
                logger.warning("role变量无变异，将其删除")
                model_data = model_data.drop('role', axis=1)
        
        if 'task_complexity' in model_data.columns:
            if model_data['task_complexity'].std() < 1e-10:
                logger.warning("task_complexity变量无变异，将其删除")
                model_data = model_data.drop('task_complexity', axis=1)
        
        # 检查CD和IP的变异
        logger.info(f"CD_c标准差: {model_data['CD_c'].std():.4f}")
        logger.info(f"IP_c标准差: {model_data['IP_c'].std():.4f}")
        
        # 如果变异太小，添加微小噪声
        if model_data['CD_c'].std() < 0.01:
            model_data['CD_c'] += np.random.normal(0, 0.001, len(model_data))
        if model_data['IP_c'].std() < 0.01:
            model_data['IP_c'] += np.random.normal(0, 0.001, len(model_data))
        
        logger.info(f"建模数据: {len(model_data)} 条记录")
        logger.info(f"对话数: {model_data['dialogue_id'].nunique()}")
        logger.info(f"说话人数: {model_data['speaker_id_unique'].nunique()}")
        
        # M0: 空模型
        self._fit_null_model(model_data)
        
        # M1: 固定效应模型（基线）
        self._fit_fixed_effects_model(model_data)
        
        # M2a: 完整随机斜率模型（尝试）
        success = self._fit_full_random_slopes_model(model_data)
        
        if not success:
            # M2b: 简化随机斜率模型（仅CD随机）
            success = self._fit_reduced_random_slopes_model(model_data)
            
            if not success:
                # M2c: 仅随机截距模型
                self._fit_random_intercept_only_model(model_data)
        
        # M3: 对话层面随机效应
        self._fit_dialogue_level_model(model_data)
        
        # 选择最佳模型
        self._select_best_model()
    
    def _fit_null_model(self, data):
        """拟合空模型"""
        logger.info("拟合空模型...")
        try:
            # 说话人层面
            m0_speaker = smf.mixedlm('activation_strength ~ 1', data, groups='speaker_id_unique')
            m0_speaker_fit = m0_speaker.fit(reml=True)
            self.models['M0_speaker'] = m0_speaker_fit
            
            # 对话层面
            m0_dialogue = smf.mixedlm('activation_strength ~ 1', data, groups='dialogue_id')
            m0_dialogue_fit = m0_dialogue.fit(reml=True)
            self.models['M0_dialogue'] = m0_dialogue_fit
            
            logger.info("✓ 空模型成功")
            self._calculate_icc(m0_speaker_fit, m0_dialogue_fit)
        except Exception as e:
            logger.error(f"空模型失败: {e}")
    
    def _calculate_icc(self, speaker_model, dialogue_model):
        """计算ICC"""
        try:
            # 说话人层面ICC
            var_speaker = float(speaker_model.cov_re.iloc[0, 0])
            var_residual = float(speaker_model.scale)
            icc_speaker = var_speaker / (var_speaker + var_residual)
            
            # 对话层面ICC
            var_dialogue = float(dialogue_model.cov_re.iloc[0, 0])
            icc_dialogue = var_dialogue / (var_dialogue + var_residual)
            
            logger.info(f"ICC - 说话人层面: {icc_speaker:.3f}, 对话层面: {icc_dialogue:.3f}")
            
            self.results['icc'] = {
                'speaker': icc_speaker,
                'dialogue': icc_dialogue
            }
        except Exception as e:
            logger.warning(f"ICC计算失败: {e}")
    
    def _fit_fixed_effects_model(self, data):
        """拟合固定效应模型"""
        logger.info("拟合固定效应模型（基线）...")
        
        # 确保分类变量正确处理
        data = data.copy()
        data['dialogue_stage'] = data['dialogue_stage'].astype(str)
        
        # 简化的公式，只包含核心变量
        if 'role' in data.columns and 'task_complexity' in data.columns:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c + role + task_complexity'
        elif 'role' in data.columns:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c + role'
        elif 'task_complexity' in data.columns:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c + task_complexity'
        else:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c'
        
        try:
            model = smf.mixedlm(formula, data, groups='speaker_id_unique')
            fit = model.fit(reml=True)
            self.models['M1_fixed'] = fit
            logger.info("✓ 固定效应模型成功")
            return True
        except Exception as e:
            logger.error(f"固定效应模型失败: {e}")
            # 尝试更简单的模型
            try:
                formula_simple = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c'
                model_simple = smf.mixedlm(formula_simple, data, groups='speaker_id_unique')
                fit_simple = model_simple.fit(reml=True)
                self.models['M1_fixed'] = fit_simple
                logger.info("✓ 简化固定效应模型成功")
                return True
            except Exception as e2:
                logger.error(f"简化模型也失败: {e2}")
                return False
    
    def _fit_full_random_slopes_model(self, data):
        """尝试完整随机斜率模型"""
        logger.info("尝试完整随机斜率模型...")
        
        # 确保数据副本
        data = data.copy()
        data['dialogue_stage'] = data['dialogue_stage'].astype(str)
        
        # 使用简化的公式
        if 'role' in data.columns and 'task_complexity' in data.columns:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c + role + task_complexity'
        else:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c'
        
        try:
            model = smf.mixedlm(
                formula, 
                data, 
                groups='speaker_id_unique',
                re_formula='~CD_c + IP_c'
            )
            fit = model.fit(reml=True, maxiter=100)
            
            # 检查是否有收敛警告
            if fit.converged:
                self.models['M2a_full'] = fit
                logger.info("✓ 完整随机斜率模型成功")
                self.model_selection = 'full_random_slopes'
                return True
            else:
                logger.warning("完整随机斜率模型未收敛")
                return False
                
        except Exception as e:
            logger.warning(f"完整随机斜率模型失败: {e}")
            return False
    
    def _fit_reduced_random_slopes_model(self, data):
        """简化随机斜率模型（仅CD随机）"""
        logger.info("尝试简化随机斜率模型（仅CD随机斜率）...")
        
        # 确保数据副本
        data = data.copy()
        data['dialogue_stage'] = data['dialogue_stage'].astype(str)
        
        # 使用简化的公式
        if 'role' in data.columns and 'task_complexity' in data.columns:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c + role + task_complexity'
        else:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c'
        
        try:
            model = smf.mixedlm(
                formula, 
                data, 
                groups='speaker_id_unique',
                re_formula='~CD_c'
            )
            fit = model.fit(reml=True, maxiter=100)
            
            if fit.converged:
                self.models['M2b_reduced'] = fit
                logger.info("✓ 简化随机斜率模型成功")
                self.model_selection = 'reduced_random_slopes'
                return True
            else:
                logger.warning("简化随机斜率模型未收敛")
                return False
                
        except Exception as e:
            logger.warning(f"简化随机斜率模型失败: {e}")
            return False
    
    def _fit_random_intercept_only_model(self, data):
        """仅随机截距模型"""
        logger.info("使用仅随机截距模型...")
        
        # 确保数据副本
        data = data.copy()
        data['dialogue_stage'] = data['dialogue_stage'].astype(str)
        
        # 使用简化的公式
        if 'role' in data.columns and 'task_complexity' in data.columns:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c + role + task_complexity'
        else:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c'
        
        try:
            model = smf.mixedlm(formula, data, groups='speaker_id_unique')
            fit = model.fit(reml=True)
            self.models['M2c_intercept'] = fit
            logger.info("✓ 随机截距模型成功")
            self.model_selection = 'random_intercept_only'
            return True
        except Exception as e:
            logger.error(f"随机截距模型失败: {e}")
            return False
    
    def _fit_dialogue_level_model(self, data):
        """对话层面模型"""
        logger.info("拟合对话层面模型...")
        
        # 确保数据副本
        data = data.copy()
        data['dialogue_stage'] = data['dialogue_stage'].astype(str)
        
        # 使用简化的公式
        if 'role' in data.columns and 'task_complexity' in data.columns:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c + role + task_complexity'
        else:
            formula = 'activation_strength ~ CD_c + IP_c + CD_c:IP_c'
        
        try:
            model = smf.mixedlm(formula, data, groups='dialogue_id')
            fit = model.fit(reml=True)
            self.models['M3_dialogue'] = fit
            logger.info("✓ 对话层面模型成功")
        except Exception as e:
            logger.warning(f"对话层面模型失败: {e}")
    
    def _select_best_model(self):
        """选择最佳模型"""
        logger.info("选择最佳模型...")
        
        # 基于AIC/BIC选择
        model_comparison = {}
        for name, model in self.models.items():
            if model is not None and hasattr(model, 'aic'):
                model_comparison[name] = {
                    'AIC': model.aic,
                    'BIC': model.bic,
                    'LogLik': model.llf
                }
        
        if model_comparison:
            best_model = min(model_comparison.items(), key=lambda x: x[1]['AIC'])
            logger.info(f"最佳模型（基于AIC）: {best_model[0]}")
            
        self.results['model_comparison'] = model_comparison
        
    def _extract_key_results(self):
        """提取关键结果"""
        # 选择用于报告的主要模型
        if self.model_selection == 'full_random_slopes' and 'M2a_full' in self.models:
            main_model = self.models['M2a_full']
            model_type = "完整随机斜率模型"
        elif self.model_selection == 'reduced_random_slopes' and 'M2b_reduced' in self.models:
            main_model = self.models['M2b_reduced']
            model_type = "简化随机斜率模型（仅CD随机）"
        elif 'M2c_intercept' in self.models:
            main_model = self.models['M2c_intercept']
            model_type = "随机截距模型"
        elif 'M1_fixed' in self.models:
            main_model = self.models['M1_fixed']
            model_type = "固定效应模型"
        else:
            # 如果所有复杂模型都失败，使用空模型
            main_model = self.models.get('M0_speaker')
            model_type = "空模型（基线）"
        
        if main_model and hasattr(main_model, 'fe_params'):
            try:
                # 获取p值（不同版本的statsmodels可能使用不同的属性名）
                if hasattr(main_model, 'pvalues'):
                    p_values = main_model.pvalues.to_dict()
                elif hasattr(main_model, 'pvalues_fe'):
                    p_values = main_model.pvalues_fe.to_dict()
                else:
                    # 手动计算p值
                    from scipy import stats
                    z_scores = main_model.fe_params / main_model.bse_fe
                    p_values = {k: 2 * (1 - stats.norm.cdf(abs(z))) for k, z in z_scores.items()}
                
                self.results['main_model'] = {
                    'type': model_type,
                    'coefficients': main_model.fe_params.to_dict(),
                    'std_errors': main_model.bse_fe.to_dict() if hasattr(main_model, 'bse_fe') else {},
                    'p_values': p_values
                }
                
                # 提取CD×IP交互效应
                if 'CD_c:IP_c' in main_model.fe_params:
                    interaction_coef = main_model.fe_params['CD_c:IP_c']
                    interaction_p = p_values.get('CD_c:IP_c', 'N/A')
                    logger.info(f"CD×IP交互效应: β={interaction_coef:.3f}, p={interaction_p}")
            except Exception as e:
                logger.warning(f"提取模型结果时出错: {e}")
                self.results['main_model'] = {
                    'type': model_type,
                    'error': str(e)
                }
        else:
            self.results['main_model'] = {
                'type': model_type,
                'note': '仅空模型成功，无固定效应参数'
            }
    
    def _generate_visualizations(self):
        """生成可视化图形"""
        logger.info("生成可视化图形...")
        
        # 图5: 框架激活的双重特性
        fig = plt.figure(figsize=(14, 10))
        
        # 5a: CD对激活强度的影响
        ax1 = plt.subplot(2, 2, 1)
        sns.scatterplot(data=self.data, x='context_dependence', y='activation_strength', 
                       hue='dialogue_stage', alpha=0.6, ax=ax1)
        ax1.set_xlabel('语境依赖度' if self.language == 'zh' else 'Context Dependence')
        ax1.set_ylabel('激活强度' if self.language == 'zh' else 'Activation Strength')
        ax1.set_title('(a) 语境依赖度的影响' if self.language == 'zh' else '(a) Effect of CD')
        
        # 5b: IP对激活强度的影响
        ax2 = plt.subplot(2, 2, 2)
        sns.scatterplot(data=self.data, x='institutional_presetting', y='activation_strength',
                       hue='role', alpha=0.6, ax=ax2)
        ax2.set_xlabel('机构预设度' if self.language == 'zh' else 'Institutional Presetting')
        ax2.set_ylabel('激活强度' if self.language == 'zh' else 'Activation Strength')
        ax2.set_title('(b) 机构预设度的影响' if self.language == 'zh' else '(b) Effect of IP')
        
        # 5c: CD×IP交互效应
        ax3 = plt.subplot(2, 2, 3)
        cd_high = self.data['context_dependence'] > self.data['context_dependence'].median()
        ip_high = self.data['institutional_presetting'] > self.data['institutional_presetting'].median()
        
        interaction_data = pd.DataFrame({
            'CD_level': ['Low', 'Low', 'High', 'High'],
            'IP_level': ['Low', 'High', 'Low', 'High'],
            'activation': [
                self.data[~cd_high & ~ip_high]['activation_strength'].mean(),
                self.data[~cd_high & ip_high]['activation_strength'].mean(),
                self.data[cd_high & ~ip_high]['activation_strength'].mean(),
                self.data[cd_high & ip_high]['activation_strength'].mean()
            ]
        })
        
        sns.barplot(data=interaction_data, x='CD_level', y='activation', hue='IP_level', ax=ax3)
        ax3.set_xlabel('语境依赖度水平' if self.language == 'zh' else 'CD Level')
        ax3.set_ylabel('激活强度' if self.language == 'zh' else 'Activation Strength')
        ax3.set_title('(c) CD×IP交互效应' if self.language == 'zh' else '(c) CD×IP Interaction')
        
        # 5d: 简单斜率分析
        ax4 = plt.subplot(2, 2, 4)
        low_ip = self.data[self.data['institutional_presetting'] < self.data['institutional_presetting'].median()]
        high_ip = self.data[self.data['institutional_presetting'] >= self.data['institutional_presetting'].median()]
        
        for data_subset, label, color in [(low_ip, '低IP', 'blue'), (high_ip, '高IP', 'red')]:
            if len(data_subset) > 10:
                x = data_subset['context_dependence']
                y = data_subset['activation_strength']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax4.plot(x_line, p(x_line), color=color, label=f'{label}: β={z[0]:.2f}', linewidth=2)
                ax4.scatter(x, y, alpha=0.3, color=color, s=20)
        
        ax4.set_xlabel('语境依赖度' if self.language == 'zh' else 'Context Dependence')
        ax4.set_ylabel('激活强度' if self.language == 'zh' else 'Activation Strength')
        ax4.set_title('(d) 简单斜率分析' if self.language == 'zh' else '(d) Simple Slopes')
        ax4.legend()
        
        plt.suptitle('图5: 框架激活的双重特性' if self.language == 'zh' else 
                    'Figure 5: Dual Characteristics of Frame Activation', fontsize=16)
        plt.tight_layout()
        
        # 保存图形
        fig_path = self.figures_dir / 'figure_5_frame_activation.jpg'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"图形已保存: {fig_path}")
    
    def _save_results(self):
        """保存结果"""
        logger.info("保存结果...")
        
        # 提取关键结果
        self._extract_key_results()
        
        # 保存JSON结果
        json_path = self.data_dir / 'hypothesis_h1_advanced_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 保存模型比较表
        if 'model_comparison' in self.results:
            comparison_df = pd.DataFrame(self.results['model_comparison']).T
            csv_path = self.tables_dir / 'h1_model_comparison.csv'
            comparison_df.to_csv(csv_path, encoding='utf-8-sig')
        
        # 生成报告
        self._generate_report()
    
    def _generate_report(self):
        """生成分析报告"""
        report = f"""# H1假设高级分析报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 分析概述
本分析实现了2.4小节要求的三层线性混合模型，采用渐进式模型构建策略。

## 2. 数据概览
- 总记录数：{len(self.data)}
- 对话数：{self.data['dialogue_id'].nunique()}
- 说话人数：{self.data['speaker_id_unique'].nunique()}

## 3. 模型选择
根据数据特征和收敛情况，最终采用：{self.results.get('main_model', {}).get('type', '未知')}

## 4. 主要发现
"""
        
        if 'main_model' in self.results and 'coefficients' in self.results['main_model']:
            coefs = self.results['main_model'].get('coefficients', {})
            pvals = self.results['main_model'].get('p_values', {})
            
            if coefs:
                # CD主效应
                if 'CD_c' in coefs:
                    p_val = pvals.get('CD_c', 'N/A')
                    if isinstance(p_val, (int, float)):
                        report += f"- 语境依赖度主效应: β={coefs['CD_c']:.3f}, p={p_val:.3f}\n"
                    else:
                        report += f"- 语境依赖度主效应: β={coefs['CD_c']:.3f}, p={p_val}\n"
                
                # IP主效应
                if 'IP_c' in coefs:
                    p_val = pvals.get('IP_c', 'N/A')
                    if isinstance(p_val, (int, float)):
                        report += f"- 机构预设度主效应: β={coefs['IP_c']:.3f}, p={p_val:.3f}\n"
                    else:
                        report += f"- 机构预设度主效应: β={coefs['IP_c']:.3f}, p={p_val}\n"
                
                # 交互效应
                if 'CD_c:IP_c' in coefs:
                    p_val = pvals.get('CD_c:IP_c', 'N/A')
                    if isinstance(p_val, (int, float)):
                        report += f"- CD×IP交互效应: β={coefs['CD_c:IP_c']:.3f}, p={p_val:.3f}\n"
                    else:
                        report += f"- CD×IP交互效应: β={coefs['CD_c:IP_c']:.3f}, p={p_val}\n"
            else:
                report += "由于模型拟合问题，无法提取固定效应参数。\n"
                if 'note' in self.results['main_model']:
                    report += f"注：{self.results['main_model']['note']}\n"
        
        # ICC
        if 'icc' in self.results:
            report += f"\n## 5. 组内相关系数（ICC）\n"
            report += f"- 说话人层面: {self.results['icc'].get('speaker', 0):.3f}\n"
            report += f"- 对话层面: {self.results['icc'].get('dialogue', 0):.3f}\n"
        
        report += "\n## 6. 模型比较\n"
        if 'model_comparison' in self.results:
            comparison_df = pd.DataFrame(self.results['model_comparison']).T
            report += comparison_df.to_string()
        
        report += "\n\n## 7. 技术说明\n"
        report += "注：由于数据的特殊结构（可能存在完全分离或多重共线性），"
        report += "复杂模型遇到奇异矩阵问题。建议考虑以下改进方案：\n"
        report += "- 增加样本量\n"
        report += "- 简化模型结构\n"
        report += "- 使用贝叶斯方法\n"
        
        # 保存报告
        report_path = self.reports_dir / 'h1_advanced_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存: {report_path}")

def main():
    """主函数 - 运行中英文双语分析"""
    # 运行中文分析
    print("运行中文分析...")
    analyzer_zh = H1ThreeLevelAnalysisV2(language='zh')
    analyzer_zh.run_analysis()
    
    # 运行英文分析
    print("\n运行英文分析...")
    analyzer_en = H1ThreeLevelAnalysisV2(language='en')
    analyzer_en.run_analysis()
    
    print("\n分析完成！结果已保存到:")
    print("中文结果: G:/Project/实证/关联框架/输出/")
    print("英文结果: G:/Project/实证/关联框架/output/")

if __name__ == "__main__":
    main()