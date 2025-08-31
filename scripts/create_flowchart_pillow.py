#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPAADIA语料库多层次分析流程图生成脚本（Pillow版本）
基于分析流程方案.md的设计要求
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

# 输出路径
BASE_DIR = Path(r"G:\Project\实证\关联框架")
OUTPUT_DIR_ZH = BASE_DIR / "输出" / "figures"
OUTPUT_DIR_EN = BASE_DIR / "output" / "figures"

# 创建输出目录
OUTPUT_DIR_ZH.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_EN.mkdir(parents=True, exist_ok=True)

class SimplifiedFlowchartGenerator:
    """简化版流程图生成器"""
    
    def __init__(self, language='zh'):
        self.language = language
        self.width = 2400
        self.height = 3000
        self.colors = {
            'layer1': '#E8F4F8',
            'layer2': '#D4E6F1',
            'layer3': '#AED6F1',
            'layer4': '#85C1F2',
            'layer5': '#5DADE2',
            'layer6': '#2E86AB',
            'text': '#2C3E50',
            'border': '#34495E'
        }
        
    def create_flowchart(self):
        """创建流程图"""
        # 创建画布
        img = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 尝试加载字体
        try:
            font_large = ImageFont.truetype("arial.ttf", 36)
            font_medium = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            # 如果找不到字体，使用默认字体
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 绘制标题
        title = self._get_text('title')
        title_bbox = draw.textbbox((0, 0), title, font=font_large)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text((self.width//2 - title_width//2, 50), title, 
                 fill=self.colors['text'], font=font_large)
        
        # 绘制六层结构
        layer_height = 400
        y_start = 150
        
        for i in range(1, 7):
            y = y_start + (i-1) * layer_height
            
            # 绘制层框架
            layer_color = self.colors[f'layer{i}']
            draw.rectangle([100, y, self.width-100, y+350], 
                          fill=layer_color, outline=self.colors['border'], width=3)
            
            # 绘制层标题
            layer_info = self._get_text(f'layer{i}')
            draw.text((120, y+20), layer_info['main'], 
                     fill=self.colors['text'], font=font_medium)
            
            # 绘制层内容
            self._draw_layer_content(draw, i, y, layer_info, font_small)
            
            # 绘制连接线
            if i < 6:
                draw.line([self.width//2, y+350, self.width//2, y+400], 
                         fill=self.colors['border'], width=3)
                # 绘制箭头
                draw.polygon([
                    (self.width//2-10, y+390),
                    (self.width//2+10, y+390),
                    (self.width//2, y+400)
                ], fill=self.colors['border'])
        
        # 保存图片
        self._save_figure(img)
        
    def _draw_layer_content(self, draw, layer_num, y_base, layer_info, font):
        """绘制层内容"""
        if layer_num == 1:
            # 第一层：数据源
            y_offset = 80
            for item in layer_info['items']:
                draw.text((150, y_base + y_offset), f"• {item}", 
                         fill=self.colors['text'], font=font)
                y_offset += 30
                
        elif layer_num == 2:
            # 第二层：数据重构
            if 'modules' in layer_info:
                x_positions = [400, 800, 1200]
                for i, module in enumerate(layer_info['modules']):
                    x = x_positions[i]
                    # 绘制模块框
                    draw.rectangle([x-150, y_base+80, x+150, y_base+280],
                                  fill='white', outline=self.colors['border'], width=2)
                    # 模块标题
                    draw.text((x-140, y_base+90), module['title'], 
                             fill=self.colors['text'], font=font)
                    # 模块内容
                    y_offset = 130
                    for detail in module['details']:
                        draw.text((x-140, y_base+y_offset), f"- {detail}", 
                                 fill=self.colors['text'], font=font)
                        y_offset += 25
                        
        elif layer_num == 3:
            # 第三层：变量构建
            if 'modules' in layer_info:
                x_positions = [300, 700, 1100, 1500]
                for i, module in enumerate(layer_info['modules']):
                    x = x_positions[i]
                    draw.rectangle([x-120, y_base+80, x+120, y_base+300],
                                  fill='white', outline=self.colors['border'], width=2)
                    draw.text((x-110, y_base+90), module['title'], 
                             fill=self.colors['text'], font=font)
                    
        elif layer_num == 4:
            # 第四层：预处理
            if 'modules' in layer_info:
                x_positions = [600, 1200]
                for i, module in enumerate(layer_info['modules']):
                    x = x_positions[i]
                    draw.rectangle([x-200, y_base+80, x+200, y_base+250],
                                  fill='white', outline=self.colors['border'], width=2)
                    draw.text((x-190, y_base+90), module['title'], 
                             fill=self.colors['text'], font=font)
                    
        elif layer_num == 5:
            # 第五层：假设检验
            if 'hypotheses' in layer_info:
                x_positions = [400, 800, 1200, 1600]
                for i, hyp in enumerate(layer_info['hypotheses']):
                    x = x_positions[i]
                    draw.rectangle([x-150, y_base+80, x+150, y_base+280],
                                  fill='white', outline=self.colors['border'], width=2)
                    draw.text((x-140, y_base+90), hyp['title'], 
                             fill=self.colors['text'], font=font)
                    draw.text((x-140, y_base+120), hyp['subtitle'], 
                             fill=self.colors['text'], font=font)
                    
        elif layer_num == 6:
            # 第六层：理论整合
            if 'modules' in layer_info:
                x_positions = [500, 1000, 1500]
                for i, module in enumerate(layer_info['modules']):
                    x = x_positions[i]
                    draw.rectangle([x-150, y_base+80, x+150, y_base+250],
                                  fill='white', outline=self.colors['border'], width=2)
                    draw.text((x-140, y_base+90), module['title'], 
                             fill=self.colors['text'], font=font)
    
    def _get_text(self, layer):
        """获取文本内容"""
        texts = {
            'zh': {
                'title': 'SPAADIA语料库多层次分析流程图',
                'layer1': {
                    'main': '第一层：数据源',
                    'items': [
                        'SPAADIA语料库：35个完整服务对话，6,053个话轮',
                        '原始XML格式，含言语行为标注',
                        'Lancaster University开发的公开语料'
                    ]
                },
                'layer2': {
                    'main': '第二层：数据重构与标注系统（XML-JSON混合三元架构）',
                    'modules': [
                        {
                            'title': '结构化主体标注',
                            'details': ['XML格式', '保留对话层级', '嵌套话轮关系']
                        },
                        {
                            'title': '索引数据库',
                            'details': ['JSONL格式', '量化变量访问', '统计分析支持']
                        },
                        {
                            'title': '元数据描述',
                            'details': ['JSON格式', '理论构念映射', '分析参数配置']
                        }
                    ]
                },
                'layer3': {
                    'main': '第三层：多维度变量构建',
                    'modules': [
                        {'title': '框架激活量化'},
                        {'title': '识解操作编码'},
                        {'title': '策略选择分类'},
                        {'title': '协商动态追踪'}
                    ]
                },
                'layer4': {
                    'main': '第四层：数据预处理与质量控制',
                    'modules': [
                        {'title': '变量转换'},
                        {'title': '质量保证'}
                    ]
                },
                'layer5': {
                    'main': '第五层：四个平行假设检验模块',
                    'hypotheses': [
                        {'title': 'H1', 'subtitle': '框架激活的双重机制'},
                        {'title': 'H2', 'subtitle': '框架驱动的策略选择'},
                        {'title': 'H3', 'subtitle': '策略演化的路径依赖'},
                        {'title': 'H4', 'subtitle': '意义协商的语义收敛'}
                    ]
                },
                'layer6': {
                    'main': '第六层：理论整合与模型构建',
                    'modules': [
                        {'title': '扩展路径分析'},
                        {'title': '模型评估'},
                        {'title': '理论贡献'}
                    ]
                }
            },
            'en': {
                'title': 'SPAADIA Corpus Multilevel Analysis Flowchart',
                'layer1': {
                    'main': 'Layer 1: Data Source',
                    'items': [
                        'SPAADIA Corpus: 35 complete service dialogues, 6,053 turns',
                        'Original XML format with speech act annotations',
                        'Public corpus developed by Lancaster University'
                    ]
                },
                'layer2': {
                    'main': 'Layer 2: Data Restructuring & Annotation System',
                    'modules': [
                        {
                            'title': 'Structured Annotation',
                            'details': ['XML format', 'Preserve hierarchy', 'Nested relations']
                        },
                        {
                            'title': 'Index Database',
                            'details': ['JSONL format', 'Quantitative access', 'Statistical support']
                        },
                        {
                            'title': 'Metadata Description',
                            'details': ['JSON format', 'Construct mapping', 'Parameter config']
                        }
                    ]
                },
                'layer3': {
                    'main': 'Layer 3: Multidimensional Variable Construction',
                    'modules': [
                        {'title': 'Frame Activation'},
                        {'title': 'Construal Coding'},
                        {'title': 'Strategy Selection'},
                        {'title': 'Negotiation Tracking'}
                    ]
                },
                'layer4': {
                    'main': 'Layer 4: Data Preprocessing & Quality Control',
                    'modules': [
                        {'title': 'Variable Transform'},
                        {'title': 'Quality Assurance'}
                    ]
                },
                'layer5': {
                    'main': 'Layer 5: Four Parallel Hypothesis Testing Modules',
                    'hypotheses': [
                        {'title': 'H1', 'subtitle': 'Dual Mechanism of Frame'},
                        {'title': 'H2', 'subtitle': 'Frame-driven Strategy'},
                        {'title': 'H3', 'subtitle': 'Path Dependency'},
                        {'title': 'H4', 'subtitle': 'Semantic Convergence'}
                    ]
                },
                'layer6': {
                    'main': 'Layer 6: Theoretical Integration & Model Building',
                    'modules': [
                        {'title': 'Path Analysis'},
                        {'title': 'Model Evaluation'},
                        {'title': 'Contributions'}
                    ]
                }
            }
        }
        
        return texts[self.language].get(layer, {})
        
    def _save_figure(self, img):
        """保存图形"""
        output_dir = OUTPUT_DIR_ZH if self.language == 'zh' else OUTPUT_DIR_EN
        filename = 'spaadia_multilevel_analysis_flowchart.jpg'
        filepath = output_dir / filename
        
        # 保存高分辨率图片
        img.save(filepath, 'JPEG', quality=95, dpi=(1200, 1200))
        print(f"图片已保存: {filepath}")
        
        # 保存元数据
        metadata = {
            'title': self._get_text('title'),
            'layers': 6,
            'hypotheses': 4,
            'modules': 15,
            'language': self.language,
            'resolution': '1200dpi',
            'format': 'jpg'
        }
        
        metadata_path = output_dir.parent / 'data' / f'flowchart_metadata_{self.language}.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"元数据已保存: {metadata_path}")

def main():
    """主函数"""
    print("="*60)
    print("SPAADIA语料库多层次分析流程图生成（简化版）")
    print("="*60)
    
    # 生成中文版本
    print("\n正在生成中文版本...")
    generator_zh = SimplifiedFlowchartGenerator(language='zh')
    generator_zh.create_flowchart()
    
    # 生成英文版本
    print("\n正在生成英文版本...")
    generator_en = SimplifiedFlowchartGenerator(language='en')
    generator_en.create_flowchart()
    
    print("\n流程图生成完成！")
    print(f"\n输出位置:")
    print(f"  中文版: {OUTPUT_DIR_ZH}")
    print(f"  英文版: {OUTPUT_DIR_EN}")
    
if __name__ == "__main__":
    main()