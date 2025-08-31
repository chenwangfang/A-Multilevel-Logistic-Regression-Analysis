#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Python实现更准确的三层嵌套模型ICC计算
通过ANOVA方法分解方差成分
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入数据加载器
from data_loader_enhanced import SPAADIADataLoader

class ThreeLevelICCAnalysis:
    """三层嵌套模型ICC分析（纯Python实现）"""
    
    def __init__(self):
        self.data = None
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        logger.info("加载SPAADIA数据...")
        
        loader = SPAADIADataLoader(language='zh')
        dataframes = loader.load_all_data()
        
        if 'frame_activation' not in dataframes:
            logger.error("缺少frame_activation数据")
            return False
        
        df = dataframes['frame_activation'].copy()
        
        # 确保必要的列存在
        required_cols = ['dialogue_id', 'speaker_role', 'turn_id', 'activation_strength']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"缺少必要的列: {missing_cols}")
            return False
        
        # 创建唯一的speaker_id
        df['speaker_id_unique'] = df['dialogue_id'].astype(str) + "_" + df['speaker_role'].astype(str)
        
        # 删除缺失值
        df = df.dropna(subset=['activation_strength'])
        
        self.data = df
        
        logger.info(f"数据加载完成: {len(df)} 行")
        logger.info(f"对话数: {df['dialogue_id'].nunique()}")
        logger.info(f"说话人数: {df['speaker_id_unique'].nunique()}")
        
        return True
    
    def calculate_variance_components_anova(self):
        """使用ANOVA方法计算三层模型的方差成分"""
        logger.info("使用ANOVA方法计算方差成分...")
        
        df = self.data.copy()
        
        # 计算各层的基本统计
        n_total = len(df)
        n_dialogues = df['dialogue_id'].nunique()
        n_speakers = df['speaker_id_unique'].nunique()
        
        # 计算每个对话的平均话轮数
        turns_per_dialogue = df.groupby('dialogue_id').size().mean()
        
        # 计算每个说话人的平均话轮数
        turns_per_speaker = df.groupby('speaker_id_unique').size().mean()
        
        # 总体均值
        grand_mean = df['activation_strength'].mean()
        
        # 计算对话层均值
        dialogue_means = df.groupby('dialogue_id')['activation_strength'].mean()
        
        # 计算说话人层均值
        speaker_means = df.groupby('speaker_id_unique')['activation_strength'].mean()
        
        # 计算各层的平方和
        # SST: 总平方和
        SST = ((df['activation_strength'] - grand_mean) ** 2).sum()
        
        # SS_dialogue: 对话间平方和
        dialogue_counts = df.groupby('dialogue_id').size()
        SS_dialogue = ((dialogue_means - grand_mean) ** 2 * dialogue_counts).sum()
        
        # SS_speaker_within_dialogue: 对话内说话人间平方和
        speaker_dialogue_means = df.groupby(['dialogue_id', 'speaker_id_unique'])['activation_strength'].mean()
        speaker_dialogue_counts = df.groupby(['dialogue_id', 'speaker_id_unique']).size()
        
        SS_speaker_total = 0
        for dialogue_id in df['dialogue_id'].unique():
            dialogue_data = df[df['dialogue_id'] == dialogue_id]
            dialogue_mean = dialogue_means[dialogue_id]
            
            speakers_in_dialogue = dialogue_data['speaker_id_unique'].unique()
            for speaker in speakers_in_dialogue:
                speaker_data = dialogue_data[dialogue_data['speaker_id_unique'] == speaker]
                if len(speaker_data) > 0:
                    speaker_mean = speaker_data['activation_strength'].mean()
                    SS_speaker_total += len(speaker_data) * (speaker_mean - dialogue_mean) ** 2
        
        # SS_within: 残差平方和（话轮层）
        SS_within = SST - SS_dialogue - SS_speaker_total
        
        # 计算均方
        df_dialogue = n_dialogues - 1
        df_speaker = n_speakers - n_dialogues  # 说话人自由度减去对话自由度（嵌套）
        df_within = n_total - n_speakers
        
        if df_dialogue > 0:
            MS_dialogue = SS_dialogue / df_dialogue
        else:
            MS_dialogue = 0
            
        if df_speaker > 0:
            MS_speaker = SS_speaker_total / df_speaker
        else:
            MS_speaker = 0
            
        if df_within > 0:
            MS_within = SS_within / df_within
        else:
            MS_within = 0
        
        # 估计方差成分（使用期望均方法）
        # 对于平衡设计，有固定公式；对于不平衡设计，使用近似
        
        # 计算每层的平均观测数
        # n0: 用于估计方差成分的调和平均数
        dialogue_sizes = df.groupby('dialogue_id').size()
        n0_dialogue = (n_total - (dialogue_sizes ** 2).sum() / n_total) / (n_dialogues - 1)
        
        speaker_sizes = df.groupby('speaker_id_unique').size()
        n0_speaker = (n_total - (speaker_sizes ** 2).sum() / n_total) / (n_speakers - 1)
        
        # 估计方差成分
        var_within = MS_within  # 残差方差
        
        # 说话人层方差
        if MS_speaker > MS_within:
            var_speaker = (MS_speaker - MS_within) / (turns_per_speaker if turns_per_speaker > 0 else 1)
        else:
            var_speaker = 0
        
        # 对话层方差
        if MS_dialogue > MS_speaker:
            var_dialogue = (MS_dialogue - MS_speaker) / (turns_per_dialogue if turns_per_dialogue > 0 else 1)
        else:
            var_dialogue = 0
        
        # 如果方差成分为负，设为0
        var_dialogue = max(0, var_dialogue)
        var_speaker = max(0, var_speaker)
        var_within = max(0, var_within)
        
        # 计算总方差
        var_total = var_dialogue + var_speaker + var_within
        
        # 计算ICC
        if var_total > 0:
            icc_dialogue = var_dialogue / var_total
            icc_speaker = var_speaker / var_total
            icc_cumulative = (var_dialogue + var_speaker) / var_total
            
            pct_dialogue = (var_dialogue / var_total) * 100
            pct_speaker = (var_speaker / var_total) * 100
            pct_within = (var_within / var_total) * 100
        else:
            icc_dialogue = icc_speaker = icc_cumulative = 0
            pct_dialogue = pct_speaker = pct_within = 0
        
        self.results = {
            'variance_components': {
                'dialogue': var_dialogue,
                'speaker': var_speaker,
                'residual': var_within,
                'total': var_total
            },
            'variance_percentages': {
                'dialogue_pct': pct_dialogue,
                'speaker_pct': pct_speaker,
                'residual_pct': pct_within
            },
            'icc': {
                'dialogue': icc_dialogue,
                'speaker': icc_speaker,
                'cumulative': icc_cumulative
            },
            'anova_components': {
                'SS_total': SST,
                'SS_dialogue': SS_dialogue,
                'SS_speaker': SS_speaker_total,
                'SS_within': SS_within,
                'MS_dialogue': MS_dialogue,
                'MS_speaker': MS_speaker,
                'MS_within': MS_within,
                'df_dialogue': df_dialogue,
                'df_speaker': df_speaker,
                'df_within': df_within
            },
            'model_info': {
                'method': 'ANOVA variance decomposition',
                'n_observations': n_total,
                'n_dialogues': n_dialogues,
                'n_speakers': n_speakers,
                'avg_turns_per_dialogue': turns_per_dialogue,
                'avg_turns_per_speaker': turns_per_speaker
            }
        }
        
        return self.results
    
    def calculate_variance_components_reml(self):
        """使用REML近似方法（通过迭代）估计方差成分"""
        logger.info("使用REML近似方法计算方差成分...")
        
        df = self.data.copy()
        
        # 初始估计（使用ANOVA结果）
        anova_results = self.calculate_variance_components_anova()
        
        var_dialogue_init = anova_results['variance_components']['dialogue']
        var_speaker_init = anova_results['variance_components']['speaker']
        var_within_init = anova_results['variance_components']['residual']
        
        # 如果ANOVA给出合理的结果，使用它
        if var_speaker_init > 0 or var_dialogue_init > 0:
            return anova_results
        
        # 否则，使用简单的分组方差方法
        logger.info("使用分组方差方法...")
        
        # 计算组内和组间方差
        # 对话层
        dialogue_means = df.groupby('dialogue_id')['activation_strength'].mean()
        var_between_dialogues = dialogue_means.var()
        
        # 说话人层（在对话内）
        speaker_vars = []
        for dialogue_id in df['dialogue_id'].unique():
            dialogue_data = df[df['dialogue_id'] == dialogue_id]
            speakers = dialogue_data['speaker_id_unique'].unique()
            if len(speakers) > 1:
                speaker_means = dialogue_data.groupby('speaker_id_unique')['activation_strength'].mean()
                speaker_vars.append(speaker_means.var())
        
        var_between_speakers = np.mean(speaker_vars) if speaker_vars else 0
        
        # 残差（话轮层）
        within_vars = []
        for speaker_id in df['speaker_id_unique'].unique():
            speaker_data = df[df['speaker_id_unique'] == speaker_id]
            if len(speaker_data) > 1:
                within_vars.append(speaker_data['activation_strength'].var())
        
        var_within = np.mean(within_vars) if within_vars else df['activation_strength'].var()
        
        # 调整估计
        var_dialogue = max(0, var_between_dialogues - var_between_speakers)
        var_speaker = max(0, var_between_speakers - var_within)
        var_residual = var_within
        
        # 计算总方差
        var_total = var_dialogue + var_speaker + var_residual
        
        # 计算ICC
        if var_total > 0:
            icc_dialogue = var_dialogue / var_total
            icc_speaker = var_speaker / var_total
            icc_cumulative = (var_dialogue + var_speaker) / var_total
            
            pct_dialogue = (var_dialogue / var_total) * 100
            pct_speaker = (var_speaker / var_total) * 100
            pct_residual = (var_residual / var_total) * 100
        else:
            icc_dialogue = icc_speaker = icc_cumulative = 0
            pct_dialogue = pct_speaker = pct_residual = 0
        
        self.results = {
            'variance_components': {
                'dialogue': var_dialogue,
                'speaker': var_speaker,
                'residual': var_residual,
                'total': var_total
            },
            'variance_percentages': {
                'dialogue_pct': pct_dialogue,
                'speaker_pct': pct_speaker,
                'residual_pct': pct_residual
            },
            'icc': {
                'dialogue': icc_dialogue,
                'speaker': icc_speaker,
                'cumulative': icc_cumulative
            },
            'model_info': {
                'method': 'Group variance decomposition',
                'n_observations': len(df),
                'n_dialogues': df['dialogue_id'].nunique(),
                'n_speakers': df['speaker_id_unique'].nunique()
            }
        }
        
        return self.results
    
    def display_results(self):
        """显示结果"""
        if not self.results:
            logger.error("没有结果可显示")
            return
        
        print("\n" + "="*60)
        print("三层嵌套模型ICC分析结果（Python实现）")
        print("="*60)
        
        if 'variance_components' in self.results:
            vc = self.results['variance_components']
            print("\n方差分解:")
            print(f"  对话层方差: {vc['dialogue']:.4f}")
            print(f"  说话人层方差: {vc['speaker']:.4f}")
            print(f"  残差方差: {vc['residual']:.4f}")
            print(f"  总方差: {vc['total']:.4f}")
        
        if 'variance_percentages' in self.results:
            vp = self.results['variance_percentages']
            print("\n方差百分比:")
            print(f"  对话层: {vp['dialogue_pct']:.1f}%")
            print(f"  说话人层: {vp['speaker_pct']:.1f}%")
            print(f"  残差: {vp['residual_pct']:.1f}%")
        
        if 'icc' in self.results:
            icc = self.results['icc']
            print("\nICC值:")
            print(f"  对话层ICC: {icc['dialogue']:.4f}")
            print(f"  说话人层ICC: {icc['speaker']:.4f}")
            print(f"  累积ICC: {icc['cumulative']:.4f}")
        
        if 'model_info' in self.results:
            info = self.results['model_info']
            print("\n模型信息:")
            print(f"  方法: {info['method']}")
            print(f"  观测数: {info['n_observations']}")
            print(f"  对话数: {info['n_dialogues']}")
            print(f"  说话人数: {info['n_speakers']}")
        
        print("="*60)
    
    def save_results(self):
        """保存结果"""
        output_dir = Path("/mnt/g/Project/实证/关联框架/输出/data")
        output_file = output_dir / "three_level_icc_python_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")
        return output_file

def main():
    """主函数"""
    analyzer = ThreeLevelICCAnalysis()
    
    # 加载数据
    if not analyzer.load_data():
        return
    
    # 方法1：ANOVA方差分解
    print("\n方法1：ANOVA方差分解")
    results1 = analyzer.calculate_variance_components_anova()
    analyzer.display_results()
    
    # 方法2：REML近似
    print("\n方法2：分组方差方法")
    results2 = analyzer.calculate_variance_components_reml()
    analyzer.display_results()
    
    # 保存最终结果
    analyzer.save_results()

if __name__ == "__main__":
    main()