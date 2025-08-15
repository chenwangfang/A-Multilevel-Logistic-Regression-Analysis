#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPAADIA语料库增强版数据加载模块
支持完整的多层次分析框架
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import xml.etree.ElementTree as ET
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SPAADIA_DataLoader')

# 数据路径配置
BASE_DIR = Path(r"G:\Project\实证\关联框架")
DATA_DIR = BASE_DIR / "SPAADIA"
OUTPUT_DIR_ZH = BASE_DIR / "输出"
OUTPUT_DIR_EN = BASE_DIR / "output"

# 数据源路径
DATA_PATHS = {
    'indices': DATA_DIR / 'indices',
    'metadata': DATA_DIR / 'metadata', 
    'xml_annotations': DATA_DIR / 'xml_annotations'
}

# 框架类型映射到4大类
FRAME_TYPE_MAPPING = {
    # 服务启动框架
    'service_initiation': 'service_initiation',
    'closing': 'service_initiation',
    'closing_reciprocation': 'service_initiation',
    'closing_finalization': 'service_initiation',
    
    # 信息提供框架
    'journey_information': 'information_provision',
    'payment_method': 'information_provision',
    'discount_eligibility': 'information_provision',
    'passenger_quantity': 'information_provision',
    'journey_date': 'information_provision',
    'departure_time': 'information_provision',
    'return_information': 'information_provision',
    'location_verification': 'information_provision',
    'date_verification': 'information_provision',
    'journey_verification': 'information_provision',
    'return_journey_verification': 'information_provision',
    
    # 交易框架
    'booking_confirmation': 'transaction',
    'payment_information': 'transaction',
    'booking_reference': 'transaction',
    'payment_confirmation': 'transaction',
    'fare_options': 'transaction',
    'fare_limitation': 'transaction',
    
    # 关系框架
    'correction': 'relational',
    'correction_acceptance': 'relational',
    'understanding': 'relational',
    'satisfaction': 'relational',
    'acceptance': 'relational',
    'acknowledgment': 'relational',
    'comprehension': 'relational'
}

class SPAADIADataLoader:
    """SPAADIA语料库数据加载器"""
    
    def __init__(self, language: str = 'zh'):
        """
        初始化数据加载器
        
        Parameters:
        -----------
        language : str
            输出语言，'zh'为中文，'en'为英文
        """
        self.language = language
        self.output_dir = OUTPUT_DIR_ZH if language == 'zh' else OUTPUT_DIR_EN
        
        # 数据路径
        self.indices_path = DATA_PATHS['indices']
        self.metadata_path = DATA_PATHS['metadata']
        self.xml_path = DATA_PATHS['xml_annotations']
        
        # 数据容器
        self.raw_data = {
            'indices': {},
            'metadata': {},
            'xml': {}
        }
        
        # 处理后的数据
        self.processed_data = {}
        
        logger.info(f"SPAADIA数据加载器初始化完成 (语言: {language})")
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有数据并返回处理后的DataFrame字典
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            包含各种分析所需的DataFrame
        """
        logger.info("开始加载SPAADIA语料库数据...")
        
        # 1. 加载原始数据
        self._load_raw_data()
        
        # 2. 整合和处理数据
        self._process_data()
        
        # 3. 创建分析用DataFrame
        dataframes = self._create_analysis_dataframes()
        
        # 4. 验证数据完整性
        self._validate_data(dataframes)
        
        logger.info(f"数据加载完成，共处理 {len(self.raw_data['indices'])} 个对话")
        
        return dataframes
    
    def _load_raw_data(self):
        """加载所有原始数据"""
        # 加载JSONL索引
        self._load_jsonl_indices()
        
        # 加载JSON元数据
        self._load_json_metadata()
        
        # 加载XML标注
        self._load_xml_annotations()
    
    def _load_jsonl_indices(self):
        """加载JSONL索引文件"""
        logger.info("加载JSONL索引文件...")
        
        jsonl_files = list(self.indices_path.glob("*.jsonl"))
        logger.info(f"找到 {len(jsonl_files)} 个JSONL文件")
        
        for file_path in tqdm(jsonl_files, desc="加载索引"):
            dialogue_id = file_path.stem
            records = []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                records.append(record)
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON解析错误 {file_path}: {e}")
                
                self.raw_data['indices'][dialogue_id] = records
                
            except Exception as e:
                logger.error(f"读取文件错误 {file_path}: {e}")
    
    def _load_json_metadata(self):
        """加载JSON元数据文件"""
        logger.info("加载JSON元数据...")
        
        json_files = list(self.metadata_path.glob("*.json"))
        logger.info(f"找到 {len(json_files)} 个JSON文件")
        
        for file_path in tqdm(json_files, desc="加载元数据"):
            dialogue_id = file_path.stem
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.raw_data['metadata'][dialogue_id] = metadata
                
            except Exception as e:
                logger.error(f"读取元数据错误 {file_path}: {e}")
    
    def _load_xml_annotations(self):
        """加载XML标注文件"""
        logger.info("加载XML标注文件...")
        
        xml_files = list(self.xml_path.glob("*.xml"))
        logger.info(f"找到 {len(xml_files)} 个XML文件")
        
        for file_path in tqdm(xml_files, desc="加载XML"):
            dialogue_id = file_path.stem
            
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                self.raw_data['xml'][dialogue_id] = root
                
            except Exception as e:
                logger.error(f"解析XML错误 {file_path}: {e}")
    
    def _process_data(self):
        """处理和整合原始数据"""
        logger.info("处理和整合数据...")
        
        # 处理每个对话
        for dialogue_id in tqdm(self.raw_data['indices'].keys(), desc="处理对话"):
            self._process_dialogue(dialogue_id)
    
    def _process_dialogue(self, dialogue_id: str):
        """处理单个对话的数据"""
        # 获取原始数据
        indices = self.raw_data['indices'].get(dialogue_id, [])
        metadata = self.raw_data['metadata'].get(dialogue_id, {})
        xml_root = self.raw_data['xml'].get(dialogue_id)
        
        if not indices or not metadata:
            logger.warning(f"对话 {dialogue_id} 数据不完整")
            return
        
        # 提取各类数据
        dialogue_data = {
            'dialogue_id': dialogue_id,
            'metadata': metadata,
            'frame_activations': [],
            'strategy_selections': [],
            'temporal_dynamics': [],
            'negotiation_points': [],
            'language_features': []
        }
        
        # 处理索引数据
        for record in indices:
            record_type = record.get('type', '')
            
            if record_type == 'frame_activation':
                dialogue_data['frame_activations'].append(self._process_frame_activation(record))
            elif record_type == 'strategy_selection':
                dialogue_data['strategy_selections'].append(self._process_strategy_selection(record))
            elif record_type == 'temporal_dynamics':
                dialogue_data['temporal_dynamics'].append(self._process_temporal_dynamics(record))
            elif record_type == 'negotiation_point':
                dialogue_data['negotiation_points'].append(self._process_negotiation_point(record))
            elif record_type == 'language_features':
                dialogue_data['language_features'].append(self._process_language_features(record))
        
        # 如果索引中没有明确的类型，尝试从字段推断
        if not any(dialogue_data[key] for key in ['frame_activations', 'strategy_selections']):
            for record in indices:
                if 'frame_type' in record and 'activation_strength' in record:
                    dialogue_data['frame_activations'].append(self._process_frame_activation(record))
                if 'strategy' in record or 'strategy_type' in record:
                    dialogue_data['strategy_selections'].append(self._process_strategy_selection(record))
        
        # 从XML提取语言特征数据
        if xml_root is not None and not dialogue_data['language_features']:
            language_features = self._extract_language_features_from_xml(xml_root, dialogue_id)
            dialogue_data['language_features'].extend(language_features)
        
        # 保存处理后的数据
        self.processed_data[dialogue_id] = dialogue_data
    
    def _process_frame_activation(self, record: Dict) -> Dict:
        """处理框架激活记录"""
        # 获取原始框架类型
        frame_type = record.get('frame_type', 'unknown')
        
        # 映射到4大类
        frame_category = FRAME_TYPE_MAPPING.get(frame_type, 'other')
        
        return {
            'dialogue_id': record.get('dialogue_id'),
            'turn_id': record.get('turn_id'),
            'utterance_id': record.get('utterance_id'),
            'frame_type': frame_type,
            'frame_category': frame_category,
            'activation_strength': float(record.get('activation_strength', 0)),
            'context_dependence': float(record.get('context_dependence', 0)),
            'institutional_presetting': float(record.get('institutional_presetting', 0)),
            'cognitive_load': float(record.get('cognitive_load', 0))
        }
    
    def _process_strategy_selection(self, record: Dict) -> Dict:
        """处理策略选择记录"""
        # 策略类型可能在不同字段中
        strategy_type = record.get('strategy') or record.get('strategy_type') or record.get('type', 'unknown')
        
        # 实现5种策略到3种策略的合并
        # 根据背景资料：
        # - 框架响应(frame_response) → 框架强化(frame_reinforcement)
        # - 框架抵抗(frame_resistance) → 框架转换(frame_shifting)
        if 'reinforcement' in strategy_type or 'reinforce' in strategy_type:
            strategy_category = 'frame_reinforcement'
        elif 'response' in strategy_type:  # 框架响应合并到框架强化
            strategy_category = 'frame_reinforcement'
        elif 'shift' in strategy_type:
            strategy_category = 'frame_shifting'
        elif 'resistance' in strategy_type or 'resist' in strategy_type:  # 框架抵抗合并到框架转换
            strategy_category = 'frame_shifting'
        elif 'blend' in strategy_type:
            strategy_category = 'frame_blending'
        else:
            strategy_category = strategy_type
        
        return {
            'dialogue_id': record.get('dialogue_id'),
            'turn_id': record.get('turn_id'),
            'utterance_id': record.get('utterance_id'),
            'strategy_type': strategy_type,
            'strategy_category': strategy_category,
            'strategy_subtype': record.get('subtype', ''),
            'efficacy': float(record.get('efficacy', 0)),
            'adaptation_index': float(record.get('adaptation_index', 0))
        }
    
    def _process_temporal_dynamics(self, record: Dict) -> Dict:
        """处理时间动态记录"""
        return {
            'dialogue_id': record.get('dialogue_id'),
            'turn_id': record.get('turn_id'),
            'relative_position': float(record.get('relative_position', 0)),
            'stage': record.get('stage', ''),
            'transition_smoothness': float(record.get('transition_smoothness', 0))
        }
    
    def _process_negotiation_point(self, record: Dict) -> Dict:
        """处理协商点记录"""
        return {
            'dialogue_id': record.get('dialogue_id'),
            'turn_id': record.get('turn_id'),
            'negotiation_type': record.get('negotiation_type', ''),
            'contribution_ratio': record.get('contribution_ratio', [0.5, 0.5]),
            'semantic_distance': float(record.get('semantic_distance', 0))
        }
    
    def _process_language_features(self, record: Dict) -> Dict:
        """处理语言特征记录"""
        return {
            'dialogue_id': record.get('dialogue_id'),
            'turn_id': record.get('turn_id'),
            'utterance_id': record.get('utterance_id'),
            'speech_act': record.get('sp_act', ''),
            'polarity': record.get('polarity', ''),
            'topic': record.get('topic', ''),
            'mode': record.get('mode', '')
        }
    
    def _create_analysis_dataframes(self) -> Dict[str, pd.DataFrame]:
        """创建用于分析的DataFrame"""
        logger.info("创建分析用DataFrame...")
        
        # 准备数据列表
        all_frame_activations = []
        all_strategy_selections = []
        all_temporal_dynamics = []
        all_negotiation_points = []
        all_language_features = []
        dialogue_metadata = []
        
        # 收集所有数据
        for dialogue_id, data in self.processed_data.items():
            # 添加对话元数据
            meta = data['metadata']
            dialogue_stats = meta.get('dialogue_statistics', {})
            
            dialogue_metadata.append({
                'dialogue_id': dialogue_id,
                'turn_count': dialogue_stats.get('turn_count', 0),
                'utterance_count': dialogue_stats.get('utterance_count', 0),
                'duration_seconds': dialogue_stats.get('duration_seconds', 0),
                'word_count': dialogue_stats.get('word_count', 0),
                'service_provider_turns': dialogue_stats.get('speaker_distribution', {}).get('service_provider', 0),
                'customer_turns': dialogue_stats.get('speaker_distribution', {}).get('customer', 0)
            })
            
            # 收集各类数据
            all_frame_activations.extend(data['frame_activations'])
            all_strategy_selections.extend(data['strategy_selections'])
            all_temporal_dynamics.extend(data['temporal_dynamics'])
            all_negotiation_points.extend(data['negotiation_points'])
            all_language_features.extend(data['language_features'])
        
        # 创建DataFrame
        dataframes = {
            'dialogue_metadata': pd.DataFrame(dialogue_metadata),
            'frame_activation': pd.DataFrame(all_frame_activations),
            'strategy_selection': pd.DataFrame(all_strategy_selections),
            'temporal_dynamics': pd.DataFrame(all_temporal_dynamics),
            'negotiation_points': pd.DataFrame(all_negotiation_points),
            'language_features': pd.DataFrame(all_language_features)
        }
        
        # 添加计算字段
        self._add_computed_fields(dataframes)
        
        return dataframes
    
    def _add_computed_fields(self, dataframes: Dict[str, pd.DataFrame]):
        """添加计算字段并合并相关数据"""
        # 合并temporal_dynamics数据到其他表
        if not dataframes['temporal_dynamics'].empty:
            td_df = dataframes['temporal_dynamics']
            
            # 为frame_activation添加时间和阶段信息
            if not dataframes['frame_activation'].empty:
                df = dataframes['frame_activation']
                # 合并temporal_dynamics字段
                df = df.merge(
                    td_df[['dialogue_id', 'turn_id', 'stage', 'relative_position', 'transition_smoothness']],
                    on=['dialogue_id', 'turn_id'], 
                    how='left'
                )
                # 填充缺失值
                if 'relative_position' not in df.columns or df['relative_position'].isna().any():
                    df['relative_position'] = df['relative_position'].fillna(0.5)
                if 'stage' not in df.columns or df['stage'].isna().any():
                    df['stage'] = df.apply(
                        lambda row: row['stage'] if pd.notna(row.get('stage')) else self._get_stage_from_position(row.get('relative_position', 0.5)),
                        axis=1
                    )
                dataframes['frame_activation'] = df
            
            # 为strategy_selection添加时间信息
            if not dataframes['strategy_selection'].empty:
                df = dataframes['strategy_selection']
                df = df.merge(
                    td_df[['dialogue_id', 'turn_id', 'stage', 'relative_position']],
                    on=['dialogue_id', 'turn_id'],
                    how='left'
                )
                dataframes['strategy_selection'] = df
            
            # 为negotiation_points添加时间信息
            if not dataframes['negotiation_points'].empty:
                df = dataframes['negotiation_points']
                df = df.merge(
                    td_df[['dialogue_id', 'turn_id', 'relative_position']],
                    on=['dialogue_id', 'turn_id'],
                    how='left'
                )
                dataframes['negotiation_points'] = df
        
        # 添加角色信息（从XML或推断）
        self._add_speaker_roles(dataframes)
        
        # 添加元数据中的统计信息
        self._merge_metadata_stats(dataframes)
        
        # 创建用于H2分析的字段
        if not dataframes['strategy_selection'].empty:
            df = dataframes['strategy_selection']
            # 从frame_activation获取frame_type和cognitive_load
            if not dataframes['frame_activation'].empty:
                frame_info = dataframes['frame_activation'][[
                    'dialogue_id', 'turn_id', 'utterance_id', 
                    'frame_type', 'frame_category', 'cognitive_load'
                ]].drop_duplicates()
                df = df.merge(frame_info, on=['dialogue_id', 'turn_id', 'utterance_id'], how='left')
            dataframes['strategy_selection'] = df
        
        # 创建用于H3分析的字段
        self._create_h3_fields(dataframes)
        
        # 创建用于H4分析的字段  
        self._create_h4_fields(dataframes)
    
    def _get_stage_from_position(self, position: float) -> str:
        """根据相对位置确定对话阶段"""
        if position <= 0.1:
            return 'opening'
        elif position <= 0.4:
            return 'information_exchange'
        elif position <= 0.8:
            return 'negotiation_verification'
        else:
            return 'closing'
    
    def _extract_language_features_from_xml(self, xml_root: ET.Element, dialogue_id: str) -> List[Dict]:
        """从XML文件中提取语言特征数据"""
        language_features = []
        
        try:
            # 遍历所有turn元素
            for turn in xml_root.findall('.//turn'):
                turn_id = turn.get('id')
                speaker = turn.get('speaker')
                speaker_role = turn.get('speaker_role')
                
                # 遍历utterance_group中的utterance
                for utterance_group in turn.findall('.//utterance_group'):
                    for utterance in utterance_group.findall('.//utterance'):
                        utterance_id = utterance.get('id')
                        sp_act = utterance.get('sp_act', '')
                        polarity = utterance.get('polarity', '')
                        topic = utterance.get('topic', '')
                        mode = utterance.get('mode', '')
                        
                        # 获取话语内容
                        content_elem = utterance.find('content')
                        content = content_elem.text if content_elem is not None else ''
                        
                        # 创建语言特征记录
                        feature_record = {
                            'dialogue_id': dialogue_id,
                            'turn_id': turn_id,
                            'utterance_id': utterance_id,
                            'speaker': speaker,
                            'speaker_role': speaker_role,
                            'speech_act': sp_act,
                            'polarity': polarity,
                            'topic': topic,
                            'mode': mode,
                            'content': content,
                            # 添加一些计算的语言特征
                            'word_count': len(content.split()) if content else 0,
                            'lexical_diversity': self._calculate_lexical_diversity(content),
                            'syntactic_complexity': self._estimate_syntactic_complexity(content)
                        }
                        
                        language_features.append(feature_record)
                        
            logger.info(f"从XML提取了 {len(language_features)} 条语言特征记录 (对话: {dialogue_id})")
            
        except Exception as e:
            logger.error(f"从XML提取语言特征时出错 (对话: {dialogue_id}): {e}")
            
        return language_features
    
    def _calculate_lexical_diversity(self, text: str) -> float:
        """计算词汇多样性（类型-标记比）"""
        if not text:
            return 0.0
        
        words = text.lower().split()
        if not words:
            return 0.0
            
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def _estimate_syntactic_complexity(self, text: str) -> float:
        """估算句法复杂度（基于平均句子长度）"""
        if not text:
            return 0.0
        
        # 简单的句子分割
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            # 如果没有句号，将整个文本作为一个句子
            sentences = [text]
        
        # 计算平均词数
        total_words = sum(len(s.split()) for s in sentences)
        avg_words_per_sentence = total_words / len(sentences) if sentences else 0
        
        # 归一化到1-10的范围
        # 假设平均句子长度1-20词，映射到1-10的复杂度
        complexity = min(10, max(1, avg_words_per_sentence / 2))
        
        return complexity
    
    def _validate_data(self, dataframes: Dict[str, pd.DataFrame]):
        """验证数据完整性"""
        logger.info("验证数据完整性...")
        
        for name, df in dataframes.items():
            if df.empty:
                logger.warning(f"{name} DataFrame为空")
            else:
                logger.info(f"{name}: {len(df)} 条记录")
                
                # 检查关键字段
                if name == 'frame_activation':
                    required_fields = ['dialogue_id', 'turn_id', 'frame_type', 'activation_strength']
                    missing = [f for f in required_fields if f not in df.columns]
                    if missing:
                        logger.warning(f"{name} 缺少字段: {missing}")
                
                elif name == 'strategy_selection':
                    required_fields = ['dialogue_id', 'turn_id', 'strategy_type']
                    missing = [f for f in required_fields if f not in df.columns]
                    if missing:
                        logger.warning(f"{name} 缺少字段: {missing}")
                
                elif name == 'language_features':
                    required_fields = ['dialogue_id', 'turn_id', 'utterance_id', 'speech_act']
                    missing = [f for f in required_fields if f not in df.columns]
                    if missing:
                        logger.warning(f"{name} 缺少字段: {missing}")
                    else:
                        # 显示语言特征的统计信息
                        logger.info(f"  - 唯一言语行为类型: {df['speech_act'].nunique()}")
                        logger.info(f"  - 极性分布: {df['polarity'].value_counts().to_dict()}")
                        logger.info(f"  - 平均词汇多样性: {df['lexical_diversity'].mean():.3f}")
    
    def save_processed_data(self, dataframes: Dict[str, pd.DataFrame]):
        """保存处理后的数据"""
        data_dir = self.output_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV
        for name, df in dataframes.items():
            csv_path = data_dir / f'{name}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"已保存 {name} 到 {csv_path}")
        
        # 保存为JSON
        json_data = {
            name: df.to_dict(orient='records')
            for name, df in dataframes.items()
        }
        
        json_path = data_dir / 'all_data.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存所有数据到 {json_path}")
    
    def _add_speaker_roles(self, dataframes: Dict[str, pd.DataFrame]):
        """添加说话者角色信息"""
        # 从language_features获取角色信息
        if not dataframes['language_features'].empty and 'speaker_role' in dataframes['language_features'].columns:
            role_mapping = dataframes['language_features'][['dialogue_id', 'turn_id', 'speaker_role']].drop_duplicates()
            
            # 添加到各个DataFrame
            for df_name in ['frame_activation', 'strategy_selection', 'temporal_dynamics', 'negotiation_points']:
                if not dataframes[df_name].empty:
                    df = dataframes[df_name]
                    if 'speaker_role' not in df.columns:
                        df = df.merge(role_mapping, on=['dialogue_id', 'turn_id'], how='left')
                        dataframes[df_name] = df
        
        # 如果还没有角色信息，根据turn_id推断
        for df_name in ['frame_activation', 'strategy_selection', 'temporal_dynamics', 'negotiation_points']:
            if not dataframes[df_name].empty:
                df = dataframes[df_name]
                if 'speaker_role' not in df.columns or df['speaker_role'].isna().any():
                    df['speaker_role'] = df.apply(
                        lambda row: row.get('speaker_role') if pd.notna(row.get('speaker_role')) 
                        else ('service_provider' if row['turn_id'].startswith('T') and int(row['turn_id'][1:]) % 2 == 1 else 'customer'),
                        axis=1
                    )
                    dataframes[df_name] = df
    
    def _merge_metadata_stats(self, dataframes: Dict[str, pd.DataFrame]):
        """合并元数据中的统计信息"""
        if not dataframes['dialogue_metadata'].empty:
            meta_df = dataframes['dialogue_metadata']
            
            # 重命名列以匹配预期的字段名
            rename_mapping = {
                'turn_count': 'total_turns',
                'duration_seconds': 'duration'
            }
            
            for old_name, new_name in rename_mapping.items():
                if old_name in meta_df.columns and new_name not in meta_df.columns:
                    meta_df[new_name] = meta_df[old_name]
            
            dataframes['dialogue_metadata'] = meta_df
    
    def _create_h3_fields(self, dataframes: Dict[str, pd.DataFrame]):
        """创建H3分析所需的字段"""
        if not dataframes['temporal_dynamics'].empty:
            df = dataframes['temporal_dynamics']
            
            # 添加time_stamp（使用relative_position作为代理）
            if 'time_stamp' not in df.columns:
                df['time_stamp'] = df['relative_position']
            
            # 计算elapsed_time
            if 'elapsed_time' not in df.columns:
                df['elapsed_time'] = df.groupby('dialogue_id')['time_stamp'].transform(
                    lambda x: (x - x.min()) * 100  # 转换为百分比时间
                )
            
            dataframes['temporal_dynamics'] = df
        
        # 在strategy_selection中添加策略序列字段
        if not dataframes['strategy_selection'].empty:
            df = dataframes['strategy_selection']
            
            # 添加current_strategy和previous_strategy
            if 'current_strategy' not in df.columns:
                df['current_strategy'] = df['strategy_type']
            
            if 'previous_strategy' not in df.columns:
                df['previous_strategy'] = df.groupby('dialogue_id')['current_strategy'].shift(1)
                df['previous_strategy'] = df['previous_strategy'].fillna('none')
            
            # 计算strategy_duration（简化版本）
            if 'strategy_duration' not in df.columns:
                df['strategy_duration'] = df.groupby(['dialogue_id', 'current_strategy']).cumcount() + 1
            
            dataframes['strategy_selection'] = df
    
    def _create_h4_fields(self, dataframes: Dict[str, pd.DataFrame]):
        """创建H4分析所需的字段"""
        if not dataframes['negotiation_points'].empty:
            df = dataframes['negotiation_points']
            
            # 添加marker_type（使用negotiation_type）
            if 'marker_type' not in df.columns:
                df['marker_type'] = df['negotiation_type']
            
            # 添加cognitive_load和emotional_valence（从frame_activation获取或使用默认值）
            if not dataframes['frame_activation'].empty:
                frame_info = dataframes['frame_activation'][[
                    'dialogue_id', 'turn_id', 'cognitive_load'
                ]].drop_duplicates()
                
                if 'cognitive_load' not in df.columns:
                    df = df.merge(frame_info, on=['dialogue_id', 'turn_id'], how='left')
            
            # 填充缺失的cognitive_load
            if 'cognitive_load' not in df.columns or df['cognitive_load'].isna().any():
                df['cognitive_load'] = df['cognitive_load'].fillna(5.0)  # 中等认知负荷
            
            # 添加emotional_valence（简化：基于negotiation_type）
            if 'emotional_valence' not in df.columns:
                valence_mapping = {
                    'challenge': 3.0,
                    'correction': 3.5,
                    'confirmation': 7.0,
                    'satisfaction': 8.0,
                    'compliance': 6.5
                }
                df['emotional_valence'] = df['negotiation_type'].map(valence_mapping).fillna(5.0)
            
            dataframes['negotiation_points'] = df


def main():
    """主函数：测试数据加载"""
    # 创建数据加载器
    loader = SPAADIADataLoader(language='zh')
    
    # 加载数据
    dataframes = loader.load_all_data()
    
    # 显示数据摘要
    print("\n=== 数据加载摘要 ===")
    for name, df in dataframes.items():
        print(f"\n{name}:")
        print(f"  记录数: {len(df)}")
        print(f"  字段: {list(df.columns)}")
        if not df.empty:
            print(f"  示例数据:")
            print(df.head(2))
    
    # 保存数据
    loader.save_processed_data(dataframes)
    
    return dataframes


if __name__ == "__main__":
    dataframes = main()