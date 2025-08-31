#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行中英文版本的R验证脚本
"""

import subprocess
import os
import sys

def run_r_scripts_in_dir(r_scripts_dir, data_dir, version_name):
    """在指定目录运行R脚本
    
    Args:
        r_scripts_dir: R脚本目录路径
        data_dir: 数据目录路径
        version_name: 版本名称（中文版/English）
    """
    print(f"\n{'='*70}")
    print(f"运行{version_name}R验证脚本")
    print(f"目录: {r_scripts_dir}")
    print('='*70)
    
    # 检查目录是否存在
    if not os.path.exists(r_scripts_dir):
        print(f"⚠️ R脚本目录不存在: {r_scripts_dir}")
        return False
    
    # 检查数据文件
    h1_data_file = os.path.join(data_dir, "h1_data_for_r.json")
    h2_data_file = os.path.join(data_dir, "h2_data_for_r.json")
    
    if not os.path.exists(h1_data_file) or not os.path.exists(h2_data_file):
        print(f"⚠️ {version_name}数据文件缺失")
        return False
    
    # 保存当前目录
    original_dir = os.getcwd()
    
    try:
        # 切换到R脚本目录
        os.chdir(r_scripts_dir)
        print(f"工作目录: {os.getcwd()}")
        
        # R脚本列表
        r_scripts = ["validate_h1.R", "validate_h2.R"]
        success_count = 0
        
        for script in r_scripts:
            if not os.path.exists(script):
                print(f"⚠️ {script} 不存在，跳过")
                continue
                
            print(f"\n{'-'*40}")
            print(f"运行 {script}")
            print('-'*40)
            
            try:
                # 运行R脚本
                result = subprocess.run(
                    ["Rscript", script],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=60  # 60秒超时
                )
                
                # 显示输出
                if result.stdout:
                    print("输出:")
                    # 只显示前20行
                    lines = result.stdout.split('\n')
                    for line in lines[:20]:
                        if line.strip():
                            print(f"  {line}")
                    if len(lines) > 20:
                        print(f"  ... (共{len(lines)}行)")
                
                if result.stderr and "Warning" not in result.stderr:
                    print("错误:")
                    print(result.stderr[:500])  # 只显示前500字符
                    
                if result.returncode == 0:
                    print(f"✓ {script} 运行成功")
                    success_count += 1
                else:
                    print(f"✗ {script} 运行失败 (返回码: {result.returncode})")
                    
            except subprocess.TimeoutExpired:
                print(f"✗ {script} 运行超时（60秒）")
            except Exception as e:
                print(f"✗ 运行 {script} 时出错: {e}")
        
        print(f"\n{version_name}完成: {success_count}/2 个脚本成功")
        return success_count > 0
        
    finally:
        # 恢复原始目录
        os.chdir(original_dir)

def run_r_validation():
    """运行中英文两个版本的R验证脚本"""
    
    print("="*70)
    print("SPAADIA R验证脚本运行器")
    print("="*70)
    
    # 定义中英文版本的路径
    versions = [
        {
            "name": "中文版",
            "r_scripts_dir": r"G:\Project\实证\关联框架\输出\r_scripts",
            "data_dir": r"G:\Project\实证\关联框架\输出\data"
        },
        {
            "name": "English",
            "r_scripts_dir": r"G:\Project\实证\关联框架\output\r_scripts",
            "data_dir": r"G:\Project\实证\关联框架\output\data"
        }
    ]
    
    # 首先检查是否需要准备数据
    data_missing = False
    for ver in versions:
        h1_data = os.path.join(ver["data_dir"], "h1_data_for_r.json")
        h2_data = os.path.join(ver["data_dir"], "h2_data_for_r.json")
        if not os.path.exists(h1_data) or not os.path.exists(h2_data):
            data_missing = True
            break
    
    if data_missing:
        print("\n检测到R验证数据文件缺失，正在准备数据...")
        try:
            import prepare_r_data
            prepare_r_data.prepare_r_validation_data()
            print("✓ 数据准备完成\n")
        except Exception as e:
            print(f"✗ 准备数据时出错: {e}")
            return
    
    # 检查R是否安装
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            print("✓ R环境已就绪")
            # 显示R版本
            version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown"
            print(f"  {version_line}")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("✗ 错误：找不到Rscript命令")
        print("\n请安装R：")
        print("1. 访问 https://www.r-project.org/")
        print("2. 下载并安装R")
        print("3. 将R的bin目录添加到系统PATH")
        print("   例如: C:\\Program Files\\R\\R-4.3.0\\bin")
        return
    
    # 运行各版本的R脚本
    success_versions = []
    for ver in versions:
        if os.path.exists(ver["r_scripts_dir"]):
            success = run_r_scripts_in_dir(
                ver["r_scripts_dir"], 
                ver["data_dir"], 
                ver["name"]
            )
            if success:
                success_versions.append(ver["name"])
        else:
            print(f"\n⚠️ 跳过{ver['name']}：目录不存在")
    
    # 总结
    print("\n" + "="*70)
    print("R验证完成总结")
    print("="*70)
    if success_versions:
        print(f"✓ 成功运行的版本: {', '.join(success_versions)}")
    else:
        print("✗ 没有成功运行的版本")
        print("\n可能的原因：")
        print("1. R脚本目录不存在 - 运行 run_hybrid_analysis.py 生成")
        print("2. R包未安装 - 运行 check_r_packages.py 安装")
        print("3. 数据文件问题 - 运行 prepare_r_data.py 重新生成")
    print("="*70)

if __name__ == "__main__":
    run_r_validation()