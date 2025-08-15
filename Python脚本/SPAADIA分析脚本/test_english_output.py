#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试高级分析脚本的英文输出功能
"""

import os
import sys
from pathlib import Path

def test_output_directories():
    """测试输出目录配置"""
    
    # 测试所有hypothesis_h*_advanced.py文件
    scripts = [
        'hypothesis_h1_advanced.py',
        'hypothesis_h2_advanced.py', 
        'hypothesis_h3_advanced.py',
        'hypothesis_h4_advanced.py'
    ]
    
    print("=" * 60)
    print("测试高级分析脚本的中英文输出配置")
    print("=" * 60)
    
    for script in scripts:
        print(f"\n检查 {script}:")
        script_path = Path(script)
        
        if not script_path.exists():
            print(f"  ❌ 文件不存在")
            continue
            
        # 读取文件内容
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查是否包含双语分析代码
        if 'language=\'en\'' in content and 'language=\'zh\'' in content:
            print(f"  ✅ 包含中英文双语分析")
        else:
            print(f"  ❌ 缺少双语分析配置")
            
        # 检查输出目录配置
        if 'output_dir = Path(f"G:/Project/实证/关联框架/{\'输出\' if language == \'zh\' else \'output\'}")' in content:
            print(f"  ✅ 输出目录配置正确")
        else:
            # 尝试另一种格式
            if '\'输出\' if language == \'zh\' else \'output\'' in content:
                print(f"  ✅ 输出目录配置正确")
            else:
                print(f"  ⚠️  需要检查输出目录配置")
                
        # 检查main函数
        if 'analyzer_en = ' in content and 'analyzer_zh = ' in content:
            print(f"  ✅ main函数包含双语运行")
        else:
            print(f"  ❌ main函数未配置双语运行")
    
    # 检查输出目录是否存在
    print("\n" + "=" * 60)
    print("检查输出目录:")
    print("=" * 60)
    
    zh_dir = Path("G:/Project/实证/关联框架/输出")
    en_dir = Path("G:/Project/实证/关联框架/output")
    
    if zh_dir.exists():
        print(f"✅ 中文输出目录存在: {zh_dir}")
        # 列出子目录
        subdirs = ['data', 'tables', 'figures', 'reports']
        for subdir in subdirs:
            subpath = zh_dir / subdir
            if subpath.exists():
                print(f"   ✅ {subdir}/ 存在")
            else:
                print(f"   ❌ {subdir}/ 不存在")
    else:
        print(f"❌ 中文输出目录不存在: {zh_dir}")
        
    if en_dir.exists():
        print(f"\n✅ 英文输出目录存在: {en_dir}")
        # 列出子目录
        subdirs = ['data', 'tables', 'figures', 'reports']
        for subdir in subdirs:
            subpath = en_dir / subdir
            if subpath.exists():
                print(f"   ✅ {subdir}/ 存在")
            else:
                print(f"   ❌ {subdir}/ 不存在")
    else:
        print(f"❌ 英文输出目录不存在: {en_dir}")
        
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n建议：")
    print("1. 如果看到❌标记，请检查相应的配置")
    print("2. 运行任一脚本: python hypothesis_h1_advanced.py")
    print("3. 检查两个输出目录是否都有新文件生成")

if __name__ == "__main__":
    test_output_directories()