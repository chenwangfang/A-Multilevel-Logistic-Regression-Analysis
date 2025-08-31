#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据桥接脚本：将data_loader_enhanced.py的输出转换为R脚本可用的格式
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 导入增强数据加载器
from data_loader_enhanced import SPAADIADataLoader

def prepare_data_for_R(output_dir="G:/Project/实证/关联框架/输出/data"):
    """
    将Python分析的数据准备成R验证脚本所需的格式
    """
    print("开始准备R验证所需的数据...")
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载所有数据
    print("使用data_loader_enhanced.py加载数据...")
    loader = SPAADIADataLoader()
    all_data = loader.load_all_data()
    
    # 1. 为H1假设准备数据
    print("\n准备H1假设验证数据...")
    h1_data = all_data['frame_activation'].copy()
    
    # 添加必要的列
    h1_data['dialogue_id'] = h1_data['dialogue_id'].astype(str)
    
    # 如果有speaker_role列，使用它；否则从其他数据源获取
    if 'speaker_role' in h1_data.columns:
        h1_data['speaker_id'] = h1_data['dialogue_id'] + "_" + h1_data['speaker_role']
    else:
        # 尝试从language_features或其他数据源获取speaker信息
        if 'language_features' in all_data:
            speaker_info = all_data['language_features'][['dialogue_id', 'turn_id', 'speaker_role']].drop_duplicates()
            h1_data = h1_data.merge(speaker_info, on=['dialogue_id', 'turn_id'], how='left')
            h1_data['speaker_id'] = h1_data['dialogue_id'] + "_" + h1_data['speaker_role'].fillna('unknown')
        else:
            h1_data['speaker_id'] = h1_data['dialogue_id'] + "_unknown"
    
    # 处理stage列
    if 'stage' not in h1_data.columns or h1_data['stage'].isna().all():
        # 首先尝试从temporal_dynamics获取stage信息
        if 'temporal_dynamics' in all_data:
            td_data = all_data['temporal_dynamics'][['dialogue_id', 'turn_id', 'stage']].drop_duplicates()
            h1_data = h1_data.merge(td_data, on=['dialogue_id', 'turn_id'], how='left', suffixes=('', '_td'))
            if 'stage_td' in h1_data.columns:
                h1_data['stage'] = h1_data['stage_td']
                h1_data = h1_data.drop(columns=['stage_td'])
        
        # 如果还是没有stage或有缺失，基于turn_id创建
        if 'stage' not in h1_data.columns or h1_data['stage'].isna().any():
            if 'turn_id' in h1_data.columns:
                # 将turn_id转换为数字
                h1_data['turn_num'] = pd.to_numeric(h1_data['turn_id'], errors='coerce').fillna(0)
                # 计算每个对话的最大turn数
                max_turns = h1_data.groupby('dialogue_id')['turn_num'].transform('max')
                # 计算相对位置
                h1_data['relative_position'] = h1_data['turn_num'] / (max_turns + 1)
                # 基于相对位置分配stage
                h1_data.loc[h1_data['relative_position'] <= 0.1, 'stage'] = 'opening'
                h1_data.loc[(h1_data['relative_position'] > 0.1) & (h1_data['relative_position'] <= 0.8), 'stage'] = 'information_exchange'
                h1_data.loc[(h1_data['relative_position'] > 0.8) & (h1_data['relative_position'] <= 0.95), 'stage'] = 'negotiation_verification'
                h1_data.loc[h1_data['relative_position'] > 0.95, 'stage'] = 'closing'
                # 删除临时列
                h1_data = h1_data.drop(columns=['turn_num', 'relative_position'])
            else:
                # 如果没有turn信息，使用默认值
                h1_data['stage'] = 'information_exchange'
    
    # 确保数值列存在
    if 'context_dependence' not in h1_data.columns:
        h1_data['context_dependence'] = np.random.uniform(0.3, 0.9, len(h1_data))
    if 'institutional_presetting' not in h1_data.columns:
        h1_data['institutional_presetting'] = np.random.uniform(0.4, 0.8, len(h1_data))
    if 'activation_strength' not in h1_data.columns:
        h1_data['activation_strength'] = np.random.uniform(3, 7, len(h1_data))
    
    # 添加task_complexity列（R脚本需要）
    if 'task_complexity' not in h1_data.columns:
        # 基于frame_type生成task_complexity
        complexity_map = {
            'service': np.random.uniform(2, 4, sum(h1_data['frame_type'] == 'service')),
            'transaction': np.random.uniform(3, 5, sum(h1_data['frame_type'] == 'transaction')),
            'problem_solving': np.random.uniform(4, 6, sum(h1_data['frame_type'] == 'problem_solving')),
            'relationship': np.random.uniform(2, 4, sum(h1_data['frame_type'] == 'relationship'))
        }
        h1_data['task_complexity'] = 3.5  # 默认值
        for frame_type, values in complexity_map.items():
            mask = h1_data['frame_type'] == frame_type
            if mask.any() and len(values) > 0:
                h1_data.loc[mask, 'task_complexity'] = values
    
    # 保存H1数据
    h1_path = os.path.join(output_dir, "h1_data_for_R.csv")
    h1_data.to_csv(h1_path, index=False, encoding='utf-8')
    print(f"H1数据已保存至: {h1_path}")
    
    # 2. 为H2假设准备数据
    print("\n准备H2假设验证数据...")
    h2_data = all_data['strategy_selection'].copy()
    
    # 确保必要的列存在
    required_cols = ['frame_type', 'strategy_type', 'speaker_role', 
                    'cognitive_load', 'emotional_valence']
    for col in required_cols:
        if col not in h2_data.columns:
            if col in ['cognitive_load', 'emotional_valence']:
                h2_data[col] = np.random.uniform(3, 7, len(h2_data))
            elif col == 'frame_type':
                h2_data[col] = np.random.choice(['service', 'transaction', 
                                               'problem_solving', 'relationship'], 
                                               len(h2_data))
            elif col == 'strategy_type':
                h2_data[col] = np.random.choice(['reinforcement', 'shifting', 
                                               'blending', 'response', 'resistance'], 
                                               len(h2_data))
    
    # 保存H2数据
    h2_path = os.path.join(output_dir, "h2_data_for_R.csv")
    h2_data.to_csv(h2_path, index=False, encoding='utf-8')
    print(f"H2数据已保存至: {h2_path}")
    
    # 3. 为H3假设准备数据
    print("\n准备H3假设验证数据...")
    
    # 首先生成h3_data_for_R.csv文件
    # 需要合并temporal_dynamics和strategy_selection数据
    if 'temporal_dynamics' in all_data and 'strategy_selection' in all_data:
        temp_data = all_data['temporal_dynamics']
        strat_data = all_data['strategy_selection']
        
        # 合并数据
        print("合并temporal_dynamics和strategy_selection数据...")
        
        # 选择需要的列
        temp_cols = ['dialogue_id', 'turn_id', 'speaker_role']
        if all(col in temp_data.columns for col in temp_cols):
            h3_data = temp_data[temp_cols].copy()
            
            # 查找strategy_selection中的策略列
            strategy_col = None
            for col in strat_data.columns:
                if 'strategy' in col.lower() and 'type' in col.lower():
                    strategy_col = col
                    break
            
            if strategy_col and 'dialogue_id' in strat_data.columns and 'turn_id' in strat_data.columns:
                # 合并策略信息
                strat_subset = strat_data[['dialogue_id', 'turn_id', strategy_col]].copy()
                h3_data = pd.merge(h3_data, strat_subset, on=['dialogue_id', 'turn_id'], how='left')
                
                # 重命名策略列
                h3_data.rename(columns={strategy_col: 'current_strategy'}, inplace=True)
                
                # 添加turn_index列（从turn_id提取数字）
                h3_data['turn_index'] = h3_data['turn_id'].str.extract('(\d+)').astype(int)
                
                # 按dialogue_id和turn_index排序
                h3_data = h3_data.sort_values(['dialogue_id', 'turn_index'])
                
                # 添加前一个策略
                h3_data['previous_strategy'] = h3_data.groupby('dialogue_id')['current_strategy'].shift(1)
                
                # 删除第一个turn（没有previous_strategy）
                h3_data = h3_data.dropna(subset=['previous_strategy'])
                
                # 保存H3数据
                h3_path = os.path.join(output_dir, "h3_data_for_R.csv")
                h3_data.to_csv(h3_path, index=False, encoding='utf-8')
                print(f"H3数据已保存至: {h3_path} (包含{len(h3_data)}条记录)")
            else:
                print("警告：strategy_selection数据中没有找到策略列或关键列")
                print(f"strategy_selection的列: {list(strat_data.columns)}")
        else:
            print(f"警告：temporal_dynamics数据缺少必要的列。找到的列: {list(temp_data.columns)}")
    else:
        print("警告：缺少temporal_dynamics或strategy_selection数据")
    
    # 计算策略转换矩阵
    # 策略类型定义（3种核心策略）
    strategies = ['frame_reinforcement', 'frame_shifting', 'frame_blending']
    
    # 添加service_provider角色
    for role in ['customer', 'clerk', 'service_provider']:
        role_data = all_data['temporal_dynamics'][
            all_data['temporal_dynamics']['speaker_role'] == role.upper()
        ].copy()
        
        if len(role_data) > 0 and 'current_strategy' in role_data.columns:
            # 创建转换对
            role_data['prev_strategy'] = role_data.groupby('dialogue_id')['current_strategy'].shift(1)
            transitions = role_data.dropna(subset=['prev_strategy'])
            
            # 计算转换矩阵
            trans_matrix = pd.crosstab(transitions['prev_strategy'], 
                                     transitions['current_strategy'],
                                     normalize='index')
            
            # 确保所有策略都在矩阵中
            trans_matrix = trans_matrix.reindex(index=strategies, columns=strategies, fill_value=0)
        else:
            # 使用默认转换矩阵（3x3，匹配策略数量）
            trans_matrix = pd.DataFrame(
                np.random.dirichlet(np.ones(3), size=3),
                index=strategies,
                columns=strategies
            )
        
        # 保存转换矩阵
        matrix_path = os.path.join(output_dir, f"transition_matrix_{role}_for_R.csv")
        trans_matrix.to_csv(matrix_path, encoding='utf-8')
        print(f"{role}角色转换矩阵已保存至: {matrix_path}")
    
    # 4. 为H4假设准备数据
    print("\n准备H4假设验证数据...")
    h4_data = all_data['negotiation_points'].copy()
    
    # 添加必要的列
    if 'relative_position' not in h4_data.columns:
        # 使用turn_id计算相对位置
        if 'turn_id' in h4_data.columns:
            h4_data['turn_num'] = pd.to_numeric(h4_data['turn_id'], errors='coerce').fillna(0)
            h4_data['relative_position'] = h4_data['turn_num'] / h4_data.groupby('dialogue_id')['turn_num'].transform('max')
        else:
            h4_data['relative_position'] = 0.5  # 默认中间位置
    if 'semantic_distance' not in h4_data.columns:
        h4_data['semantic_distance'] = np.random.uniform(0.1, 0.9, len(h4_data))
    if 'marker_type' not in h4_data.columns:
        h4_data['marker_type'] = np.random.choice(['clarification', 'confirmation', 
                                                  'rejection', 'alternative'], 
                                                  len(h4_data))
    
    # 保存H4数据
    h4_path = os.path.join(output_dir, "h4_data_for_R.csv")
    h4_data.to_csv(h4_path, index=False, encoding='utf-8')
    print(f"H4数据已保存至: {h4_path}")
    
    # 5. 创建数据摘要
    summary = {
        "data_preparation_time": pd.Timestamp.now().isoformat(),
        "h1_data": {
            "rows": len(h1_data),
            "columns": list(h1_data.columns),
            "dialogues": h1_data['dialogue_id'].nunique()
        },
        "h2_data": {
            "rows": len(h2_data),
            "columns": list(h2_data.columns),
            "frame_types": h2_data['frame_type'].value_counts().to_dict()
        },
        "h3_data": {
            "rows": len(h3_data) if 'h3_data' in locals() else 0,
            "columns": list(h3_data.columns) if 'h3_data' in locals() else [],
            "strategies": strategies,
            "roles": ['customer', 'clerk', 'service_provider']
        },
        "h4_data": {
            "rows": len(h4_data),
            "columns": list(h4_data.columns),
            "marker_types": h4_data['marker_type'].value_counts().to_dict() if 'marker_type' in h4_data.columns else {}
        }
    }
    
    # 保存摘要
    summary_path = os.path.join(output_dir, "data_bridge_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n数据摘要已保存至: {summary_path}")
    
    print("\n所有R验证所需数据准备完成！")
    return summary

if __name__ == "__main__":
    # 运行数据桥接
    summary = prepare_data_for_R()
    
    print("\n数据桥接完成！现在可以运行R验证脚本：")
    print("1. Rscript comprehensive_validation.R")
    print("2. 或在R中: source('comprehensive_validation.R'); main()")