#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复R脚本中的硬编码输出路径，支持中英文双语输出
"""

import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_r_script_paths(file_path: Path):
    """修复单个R脚本中的硬编码路径"""
    
    logger.info(f"处理文件: {file_path}")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 记录修改数量
    changes = 0
    original_content = content
    
    # 1. 修复ggsave路径
    pattern = r'ggsave\("G:/Project/实证/关联框架/输出/(.*?)"\s*,'
    def replace_ggsave(match):
        nonlocal changes
        changes += 1
        file_name = match.group(1)
        return f'ggsave(file.path(base_dir, "{file_name}"),'
    content = re.sub(pattern, replace_ggsave, content)
    
    # 2. 修复write.csv路径
    pattern = r'write\.csv\((.*?),\s*"G:/Project/实证/关联框架/输出/(.*?)"\)'
    def replace_write_csv(match):
        nonlocal changes
        changes += 1
        data_var = match.group(1)
        file_name = match.group(2)
        return f'write.csv({data_var}, file.path(base_dir, "{file_name}"))'
    content = re.sub(pattern, replace_write_csv, content)
    
    # 3. 修复saveRDS路径
    pattern = r'saveRDS\((.*?),\s*"G:/Project/实证/关联框架/输出/(.*?)"\)'
    def replace_saveRDS(match):
        nonlocal changes
        changes += 1
        data_var = match.group(1)
        file_name = match.group(2)
        return f'saveRDS({data_var}, file.path(base_dir, "{file_name}"))'
    content = re.sub(pattern, replace_saveRDS, content)
    
    # 4. 修复readRDS路径
    pattern = r'readRDS\("G:/Project/实证/关联框架/输出/(.*?)"\)'
    def replace_readRDS(match):
        nonlocal changes
        changes += 1
        file_name = match.group(1)
        return f'readRDS(file.path(base_dir, "{file_name}"))'
    content = re.sub(pattern, replace_readRDS, content)
    
    # 5. 修复file.path硬编码
    pattern = r'file\.path\("G:/Project/实证/关联框架/输出",\s*(.*?)\)'
    def replace_file_path(match):
        nonlocal changes
        changes += 1
        sub_path = match.group(1)
        return f'file.path(base_dir, {sub_path})'
    content = re.sub(pattern, replace_file_path, content)
    
    # 6. 添加base_dir变量声明（如果函数中使用了base_dir但没有定义）
    if 'base_dir' in content and 'base_dir <-' not in content:
        # 在每个使用base_dir的函数开头添加声明
        functions_with_base_dir = re.findall(r'(\w+)\s*<-\s*function\([^)]*\)\s*\{[^}]*base_dir[^}]*\}', content, re.DOTALL)
        for func_name in functions_with_base_dir:
            pattern = rf'({func_name}\s*<-\s*function\([^)]*\)\s*\{{)'
            replacement = r'\1\n  # 获取输出目录\n  base_dir <- ifelse(exists("language") && language == "en",\n                     "G:/Project/实证/关联框架/output",\n                     "G:/Project/实证/关联框架/输出")\n'
            content = re.sub(pattern, replacement, content)
            changes += 1
    
    # 只有在有修改时才写入文件
    if changes > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"  ✓ 修复了 {changes} 处路径")
    else:
        logger.info(f"  - 没有需要修复的路径")
    
    return changes

def add_bilingual_execution(file_path: Path):
    """为R脚本的main函数添加双语执行支持"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有需要修改的main调用
    if 'if __name__ == "__main__"' in content or 'main()' in content:
        # 这是Python风格的，跳过
        return 0
    
    # 查找文件末尾的main()调用
    if content.strip().endswith('main()'):
        # 替换为双语调用
        new_ending = '''
# 运行双语分析
cat("\\n运行中文分析...\\n")
results_zh <- main("zh")

cat("\\n运行英文分析...\\n")
results_en <- main("en")

cat("\\n分析完成！结果已保存到:\\n")
cat("中文结果: G:/Project/实证/关联框架/输出/\\n")
cat("英文结果: G:/Project/实证/关联框架/output/\\n")
'''
        content = content.replace('main()', new_ending)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"  ✓ 添加了双语执行支持")
        return 1
    
    return 0

def main():
    """主函数"""
    
    logger.info("="*60)
    logger.info("开始修复R脚本输出路径")
    logger.info("="*60)
    
    # 获取所有R脚本
    script_dir = Path('/mnt/g/Project/实证/关联框架/Python脚本/SPAADIA分析脚本')
    r_scripts = list(script_dir.glob('*.R'))
    
    total_changes = 0
    for script in r_scripts:
        changes = fix_r_script_paths(script)
        total_changes += changes
        
        # 为主要验证脚本添加双语支持
        if 'validation' in script.name or 'comprehensive' in script.name:
            changes += add_bilingual_execution(script)
    
    logger.info("="*60)
    logger.info(f"修复完成！共修改 {total_changes} 处")
    logger.info("="*60)
    
    logger.info("\n建议：")
    logger.info("1. 运行R脚本时，使用: results <- main('zh') 或 main('en')")
    logger.info("2. 检查输出目录确认文件生成正确")

if __name__ == "__main__":
    main()