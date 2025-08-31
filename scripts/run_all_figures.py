#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行所有图表生成脚本
生成用于论文发表的5个核心图表
"""

import sys
import io
import subprocess
from pathlib import Path
import time

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_figure_script(script_name):
    """运行单个图表脚本"""
    print(f"\n{'='*60}")
    print(f"运行 {script_name}...")
    print('='*60)
    
    try:
        # 使用subprocess运行脚本
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=120  # 2分钟超时
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} 运行成功")
            # 打印输出的最后几行（包含保存路径信息）
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print(f"❌ {script_name} 运行失败")
            print(f"错误信息：{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⚠️ {script_name} 运行超时（超过2分钟）")
        return False
    except Exception as e:
        print(f"❌ 运行 {script_name} 时出错：{e}")
        return False

def main():
    """主函数"""
    print("="*80)
    print("批量生成论文图表")
    print("="*80)
    print("\n此脚本将运行以下5个图表生成脚本：")
    print("1. figure1_theoretical_framework.py - 理论框架与研究假设")
    print("2. figure2_dual_mechanism.py - 框架激活的双重机制")
    print("3. figure3_frame_strategy_sankey.py - 框架驱动的策略选择")
    print("4. figure4_markov_evolution.py - 策略演化的马尔可夫模型")
    print("5. figure5_semantic_convergence.py - 协商中的语义收敛")
    
    # 定义要运行的脚本列表
    figure_scripts = [
        "figure1_theoretical_framework.py",
        "figure2_dual_mechanism.py",
        "figure3_frame_strategy_sankey.py",
        "figure4_markov_evolution.py",
        "figure5_semantic_convergence.py"
    ]
    
    # 检查脚本是否存在
    script_dir = Path(__file__).parent
    missing_scripts = []
    for script in figure_scripts:
        script_path = script_dir / script
        if not script_path.exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"\n❌ 缺少以下脚本文件：")
        for script in missing_scripts:
            print(f"   - {script}")
        print("\n请确保所有脚本都在当前目录中")
        return
    
    # 运行所有脚本
    print("\n开始批量运行...")
    start_time = time.time()
    
    success_count = 0
    failed_scripts = []
    
    for i, script in enumerate(figure_scripts, 1):
        print(f"\n[{i}/{len(figure_scripts)}] 处理中...")
        if run_figure_script(script):
            success_count += 1
        else:
            failed_scripts.append(script)
        
        # 稍微延迟，避免资源冲突
        if i < len(figure_scripts):
            time.sleep(1)
    
    # 统计结果
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print("批量运行完成")
    print("="*80)
    print(f"\n运行统计：")
    print(f"  总计：{len(figure_scripts)} 个脚本")
    print(f"  成功：{success_count} 个")
    print(f"  失败：{len(failed_scripts)} 个")
    print(f"  耗时：{elapsed_time:.1f} 秒")
    
    if failed_scripts:
        print(f"\n失败的脚本：")
        for script in failed_scripts:
            print(f"  - {script}")
        print("\n请检查错误信息并重新运行失败的脚本")
    else:
        print("\n✅ 所有图表生成成功！")
        print("\n输出位置：")
        print("  中文版：G:/Project/实证/关联框架/输出/figures/")
        print("  英文版：G:/Project/实证/关联框架/output/figures/")
        print("\n生成的图表文件：")
        print("  - figure_1_theoretical_framework.jpg")
        print("  - figure_2_dual_mechanism.jpg")
        print("  - figure_3_frame_strategy_sankey.jpg")
        print("  - figure_4_markov_evolution.jpg")
        print("  - figure_5_semantic_convergence.jpg")

if __name__ == "__main__":
    main()