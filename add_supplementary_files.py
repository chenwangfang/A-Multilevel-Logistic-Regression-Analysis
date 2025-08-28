#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add supplementary materials to GitHub repository
"""

import shutil
from pathlib import Path
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def add_supplementary_files():
    """添加补充材料文件到仓库"""
    
    # 源文件和目标文件映射
    base_path = Path(r"G:\Project\实证\关联框架")
    repo_path = Path(r"G:\Project\实证\关联框架\github-repo-20250828_080438")
    
    # 文件映射列表
    files_to_add = [
        # (源文件, 目标文件)
        ("背景资料/github资料/补充材料_en.md", 
         "documentation/Supplementary_Materials.md"),
        
        ("背景资料/github资料/XML-JSON混合三元架构(英文) .md", 
         "documentation/XML-JSON_Hybrid_Architecture.md"),
        
        ("背景资料/github资料/英文版本编码方案.md", 
         "documentation/Coding_Scheme.md"),
    ]
    
    # 确保documentation目录存在
    doc_dir = repo_path / "documentation"
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Adding supplementary materials to GitHub repository")
    print("="*60)
    print()
    
    # 复制文件
    success_count = 0
    for src_rel, dst_rel in files_to_add:
        src = base_path / src_rel
        dst = repo_path / dst_rel
        
        if src.exists():
            try:
                shutil.copy2(src, dst)
                print(f"[OK] Added: {os.path.basename(dst_rel)}")
                
                # 显示文件大小
                size_kb = os.path.getsize(dst) / 1024
                print(f"  File size: {size_kb:.1f} KB")
                
                success_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to copy: {src_rel}")
                print(f"  Error: {e}")
        else:
            print(f"[ERROR] Source file not found: {src_rel}")
    
    print()
    print("="*60)
    print(f"[SUCCESS] Added {success_count}/{len(files_to_add)} files")
    print("="*60)
    
    # 更新README.md，添加补充材料说明
    readme_path = repo_path / "README.md"
    if readme_path.exists():
        print("\nUpdating README.md...")
        
        # 读取现有内容
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已包含补充材料部分
        if "### Supplementary Materials" not in content:
            # 在documentation部分添加说明
            supplement_section = """

### 📎 Supplementary Materials / 补充材料

The `documentation/` folder contains essential supplementary materials:

1. **Supplementary_Materials.md**: Complete supplementary materials including:
   - Detailed statistical methods
   - Extended results tables
   - Additional analyses
   - Robustness checks

2. **XML-JSON_Hybrid_Architecture.md**: Technical documentation of the XML-JSON hybrid triple architecture:
   - Data structure design
   - Processing pipeline
   - Integration methodology

3. **Coding_Scheme.md**: Complete coding scheme for SPAADIA corpus:
   - Frame type definitions
   - Strategy classifications
   - Annotation guidelines
   - Inter-rater reliability metrics

These documents provide comprehensive technical details supporting the main analysis."""
            
            # 查找合适的插入位置（在## 📁 Repository Structure之前）
            if "## 📁 Repository Structure" in content:
                content = content.replace(
                    "## 📁 Repository Structure",
                    supplement_section + "\n\n## 📁 Repository Structure"
                )
            else:
                # 否则添加到文件末尾
                content += supplement_section
            
            # 写回文件
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("[OK] README.md updated with supplementary materials section")
    
    print("\nNext steps:")
    print("1. Add new files to Git:")
    print(f"   cd {repo_path}")
    print("   git add .")
    print('   git commit -m "Add supplementary materials and technical documentation"')
    print("   git push origin main")

if __name__ == "__main__":
    try:
        add_supplementary_files()
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()