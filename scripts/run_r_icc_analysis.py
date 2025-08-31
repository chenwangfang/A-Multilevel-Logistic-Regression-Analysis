#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用R的lme4包计算正确的三层嵌套模型ICC
"""

import pandas as pd
import numpy as np
import json
import subprocess
import os
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入数据加载器
from data_loader_enhanced import SPAADIADataLoader

def prepare_data_for_r():
    """准备数据供R脚本使用"""
    logger.info("加载SPAADIA数据...")
    
    # 加载数据
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
    
    # 只保留需要的列
    df_for_r = df[['dialogue_id', 'speaker_id_unique', 'speaker_role', 'turn_id', 'activation_strength']]
    
    # 删除缺失值
    df_for_r = df_for_r.dropna()
    
    logger.info(f"准备的数据: {len(df_for_r)} 行")
    logger.info(f"对话数: {df_for_r['dialogue_id'].nunique()}")
    logger.info(f"说话人数: {df_for_r['speaker_id_unique'].nunique()}")
    
    # 保存为CSV供R读取
    csv_path = Path(__file__).parent / "temp_data_for_r.csv"
    df_for_r.to_csv(csv_path, index=False)
    logger.info(f"数据已保存到: {csv_path}")
    
    return True

def run_r_analysis():
    """运行R脚本进行三层模型分析"""
    logger.info("运行R脚本...")
    
    # 根据操作系统选择R脚本
    import platform
    if platform.system() == "Windows":
        r_script_path = Path(__file__).parent / "three_level_icc_analysis_windows.R"
    else:
        r_script_path = Path(__file__).parent / "three_level_icc_analysis.R"
    
    if not r_script_path.exists():
        logger.error(f"R脚本不存在: {r_script_path}")
        return None
    
    try:
        # 运行R脚本（处理Windows编码问题）
        result = subprocess.run(
            ["Rscript", str(r_script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',  # 指定UTF-8编码
            check=True
        )
        
        # 输出R脚本的日志
        if result.stdout:
            logger.info("R脚本输出:")
            print(result.stdout)
        
        if result.stderr:
            logger.warning("R脚本警告/错误:")
            print(result.stderr)
        
        # 从输出中提取JSON结果
        stdout = result.stdout
        if "===JSON_START===" in stdout and "===JSON_END===" in stdout:
            json_start = stdout.index("===JSON_START===") + len("===JSON_START===")
            json_end = stdout.index("===JSON_END===")
            json_str = stdout[json_start:json_end].strip()
            
            try:
                results = json.loads(json_str)
                logger.info("成功解析R脚本返回的JSON结果")
                return results
            except json.JSONDecodeError as e:
                logger.error(f"解析JSON失败: {e}")
                
                # 尝试从文件读取
                json_file = Path(__file__).parent / "three_level_icc_results.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        results = json.load(f)
                    logger.info("从文件读取JSON结果")
                    return results
        else:
            # 尝试从文件读取
            json_file = Path(__file__).parent / "three_level_icc_results.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    results = json.load(f)
                logger.info("从文件读取JSON结果")
                return results
                
    except subprocess.CalledProcessError as e:
        logger.error(f"R脚本执行失败: {e}")
        if e.stdout:
            print("标准输出:", e.stdout)
        if e.stderr:
            print("错误输出:", e.stderr)
        return None
    except FileNotFoundError:
        logger.error("未找到Rscript。请确保R已安装并在PATH中")
        return None

def display_results(results):
    """显示ICC计算结果"""
    if not results:
        logger.error("没有结果可显示")
        return
    
    print("\n" + "="*60)
    print("三层嵌套模型ICC分析结果")
    print("="*60)
    
    if 'variance_components' in results:
        vc = results['variance_components']
        print("\n方差分解:")
        print(f"  对话层方差: {vc.get('dialogue', 'N/A'):.4f}")
        print(f"  说话人层方差: {vc.get('speaker', 'N/A'):.4f}")
        print(f"  残差方差: {vc.get('residual', 'N/A'):.4f}")
        print(f"  总方差: {vc.get('total', 'N/A'):.4f}")
    
    if 'variance_percentages' in results:
        vp = results['variance_percentages']
        print("\n方差百分比:")
        print(f"  对话层: {vp.get('dialogue_pct', 'N/A'):.1f}%")
        print(f"  说话人层: {vp.get('speaker_pct', 'N/A'):.1f}%")
        print(f"  残差: {vp.get('residual_pct', 'N/A'):.1f}%")
    
    if 'icc' in results:
        icc = results['icc']
        print("\nICC值:")
        print(f"  对话层ICC: {icc.get('dialogue', 'N/A'):.4f}")
        print(f"  说话人层ICC: {icc.get('speaker', 'N/A'):.4f}")
        print(f"  累积ICC: {icc.get('cumulative', 'N/A'):.4f}")
    
    if 'model_info' in results:
        info = results['model_info']
        print("\n模型信息:")
        print(f"  模型类型: {info.get('type', 'N/A')}")
        print(f"  公式: {info.get('formula', 'N/A')}")
        if 'n_observations' in info:
            print(f"  观测数: {info['n_observations']}")
        if 'n_dialogues' in info:
            print(f"  对话数: {info['n_dialogues']}")
        if 'n_speakers' in info:
            print(f"  说话人数: {info['n_speakers']}")
    
    print("="*60)
    
    # 保存到输出目录（处理Windows和Linux路径）
    import platform
    if platform.system() == "Windows":
        output_dir = Path("G:/Project/实证/关联框架/输出/data")
    else:
        output_dir = Path("/mnt/g/Project/实证/关联框架/输出/data")
    
    # 确保目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "three_level_icc_results_from_r.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    """主函数"""
    logger.info("开始三层嵌套模型ICC分析（使用R）")
    
    # 准备数据
    if not prepare_data_for_r():
        logger.error("数据准备失败")
        return
    
    # 运行R分析
    results = run_r_analysis()
    
    # 显示结果
    display_results(results)
    
    # 清理临时文件
    temp_csv = Path(__file__).parent / "temp_data_for_r.csv"
    if temp_csv.exists():
        temp_csv.unlink()
        logger.info("已删除临时CSV文件")

if __name__ == "__main__":
    main()