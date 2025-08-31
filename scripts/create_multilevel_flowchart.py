#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPAADIA语料库多层次分析流程图生成脚本
基于分析流程方案.md的设计要求
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出路径
BASE_DIR = Path(r"G:\Project\实证\关联框架")
OUTPUT_DIR_ZH = BASE_DIR / "输出" / "figures"
OUTPUT_DIR_EN = BASE_DIR / "output" / "figures"

# 创建输出目录
OUTPUT_DIR_ZH.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_EN.mkdir(parents=True, exist_ok=True)

class FlowchartGenerator:
    """流程图生成器"""
    
    def __init__(self, language='zh'):
        self.language = language
        self.fig_width = 16
        self.fig_height = 20
        self.colors = {
            'layer1': '#E8F4F8',  # 浅蓝
            'layer2': '#D4E6F1',  # 中蓝
            'layer3': '#AED6F1',  # 深蓝
            'layer4': '#85C1F2',  # 更深蓝
            'layer5': '#5DADE2',  # 假设检验层
            'layer6': '#2E86AB',  # 理论整合层
            'arrow': '#34495E',   # 箭头颜色
            'text': '#2C3E50'     # 文字颜色
        }
        
    def create_flowchart(self):
        """创建完整的六层流程图"""
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 120)
        ax.axis('off')
        
        # 绘制六层架构
        self._draw_layer1(ax)  # 数据源
        self._draw_layer2(ax)  # 数据重构与标注
        self._draw_layer3(ax)  # 多维度变量构建
        self._draw_layer4(ax)  # 数据预处理
        self._draw_layer5(ax)  # 假设检验
        self._draw_layer6(ax)  # 理论整合
        
        # 添加连接线
        self._draw_connections(ax)
        
        # 添加标题
        title = self._get_text('title')
        ax.text(50, 115, title, fontsize=20, weight='bold', 
                ha='center', color=self.colors['text'])
        
        # 保存图形
        self._save_figure(fig)
        
    def _draw_layer1(self, ax):
        """绘制第一层：数据源"""
        # 主框架
        rect = FancyBboxPatch((10, 100), 80, 8, 
                              boxstyle="round,pad=0.1",
                              facecolor=self.colors['layer1'],
                              edgecolor=self.colors['arrow'],
                              linewidth=2)
        ax.add_patch(rect)
        
        # 文字内容
        texts = self._get_text('layer1')
        ax.text(50, 104, texts['main'], fontsize=14, weight='bold', 
                ha='center', va='center')
        
        # 子内容
        y_pos = 102
        for item in texts['items']:
            ax.text(50, y_pos, item, fontsize=10, ha='center', va='center')
            y_pos -= 1.5
            
    def _draw_layer2(self, ax):
        """绘制第二层：数据重构与标注系统"""
        # 主框架
        rect = FancyBboxPatch((10, 85), 80, 12,
                              boxstyle="round,pad=0.1",
                              facecolor=self.colors['layer2'],
                              edgecolor=self.colors['arrow'],
                              linewidth=2)
        ax.add_patch(rect)
        
        texts = self._get_text('layer2')
        ax.text(50, 93, texts['main'], fontsize=14, weight='bold',
                ha='center', va='center')
        
        # 三个子模块
        modules = texts['modules']
        x_positions = [25, 50, 75]
        
        for i, (pos, module) in enumerate(zip(x_positions, modules)):
            # 子框架
            sub_rect = Rectangle((pos-10, 86), 20, 5,
                                facecolor='white',
                                edgecolor=self.colors['arrow'],
                                linewidth=1)
            ax.add_patch(sub_rect)
            
            # 模块标题
            ax.text(pos, 89.5, module['title'], fontsize=11, weight='bold',
                    ha='center', va='center')
            
            # 模块内容
            y_offset = 88
            for detail in module['details']:
                ax.text(pos, y_offset, detail, fontsize=8,
                        ha='center', va='center')
                y_offset -= 0.8
                
    def _draw_layer3(self, ax):
        """绘制第三层：多维度变量构建"""
        rect = FancyBboxPatch((10, 65), 80, 17,
                              boxstyle="round,pad=0.1",
                              facecolor=self.colors['layer3'],
                              edgecolor=self.colors['arrow'],
                              linewidth=2)
        ax.add_patch(rect)
        
        texts = self._get_text('layer3')
        ax.text(50, 79, texts['main'], fontsize=14, weight='bold',
                ha='center', va='center')
        
        # 四个并行模块
        modules = texts['modules']
        x_positions = [20, 40, 60, 80]
        
        for pos, module in zip(x_positions, modules):
            # 子框架
            sub_rect = Rectangle((pos-8, 66), 16, 11,
                                facecolor='white',
                                edgecolor=self.colors['arrow'],
                                linewidth=1)
            ax.add_patch(sub_rect)
            
            # 模块标题
            ax.text(pos, 75, module['title'], fontsize=10, weight='bold',
                    ha='center', va='center')
            
            # 模块内容
            y_offset = 73
            for detail in module['details']:
                ax.text(pos, y_offset, detail, fontsize=7,
                        ha='center', va='center', wrap=True)
                y_offset -= 1.2
                
    def _draw_layer4(self, ax):
        """绘制第四层：数据预处理与质量控制"""
        rect = FancyBboxPatch((10, 50), 80, 12,
                              boxstyle="round,pad=0.1",
                              facecolor=self.colors['layer4'],
                              edgecolor=self.colors['arrow'],
                              linewidth=2)
        ax.add_patch(rect)
        
        texts = self._get_text('layer4')
        ax.text(50, 58, texts['main'], fontsize=14, weight='bold',
                ha='center', va='center')
        
        # 两个子模块
        modules = texts['modules']
        x_positions = [30, 70]
        
        for pos, module in zip(x_positions, modules):
            sub_rect = Rectangle((pos-15, 51), 30, 5,
                                facecolor='white',
                                edgecolor=self.colors['arrow'],
                                linewidth=1)
            ax.add_patch(sub_rect)
            
            ax.text(pos, 54, module['title'], fontsize=11, weight='bold',
                    ha='center', va='center')
            
            y_offset = 52.5
            for detail in module['details']:
                ax.text(pos, y_offset, detail, fontsize=8,
                        ha='center', va='center')
                y_offset -= 0.8
                
    def _draw_layer5(self, ax):
        """绘制第五层：四个平行假设检验模块"""
        rect = FancyBboxPatch((10, 20), 80, 27,
                              boxstyle="round,pad=0.1",
                              facecolor=self.colors['layer5'],
                              edgecolor=self.colors['arrow'],
                              linewidth=2)
        ax.add_patch(rect)
        
        texts = self._get_text('layer5')
        ax.text(50, 44, texts['main'], fontsize=14, weight='bold',
                ha='center', va='center', color='white')
        
        # 四个假设检验模块
        hypotheses = texts['hypotheses']
        positions = [(20, 35), (40, 35), (60, 35), (80, 35)]
        
        for (x, y), hyp in zip(positions, hypotheses):
            # 假设框架
            hyp_rect = Rectangle((x-8, y-12), 16, 10,
                                facecolor='white',
                                edgecolor=self.colors['arrow'],
                                linewidth=1.5)
            ax.add_patch(hyp_rect)
            
            # 假设标题
            ax.text(x, y-2, hyp['title'], fontsize=10, weight='bold',
                    ha='center', va='center')
            
            # 假设内容
            ax.text(x, y-4, hyp['subtitle'], fontsize=8,
                    ha='center', va='center')
            
            # 方法
            ax.text(x, y-6, f"方法: {hyp['method']}", fontsize=7,
                    ha='center', va='center')
            
            # 输出
            ax.text(x, y-8, f"输出: {hyp['output']}", fontsize=7,
                    ha='center', va='center')
            
    def _draw_layer6(self, ax):
        """绘制第六层：理论整合与模型构建"""
        rect = FancyBboxPatch((10, 5), 80, 12,
                              boxstyle="round,pad=0.1",
                              facecolor=self.colors['layer6'],
                              edgecolor=self.colors['arrow'],
                              linewidth=2)
        ax.add_patch(rect)
        
        texts = self._get_text('layer6')
        ax.text(50, 13, texts['main'], fontsize=14, weight='bold',
                ha='center', va='center', color='white')
        
        # 三个子模块
        modules = texts['modules']
        x_positions = [25, 50, 75]
        
        for pos, module in zip(x_positions, modules):
            sub_rect = Rectangle((pos-10, 6), 20, 5,
                                facecolor='white',
                                edgecolor=self.colors['arrow'],
                                linewidth=1)
            ax.add_patch(sub_rect)
            
            ax.text(pos, 9, module['title'], fontsize=10, weight='bold',
                    ha='center', va='center')
            
            y_offset = 7.5
            for detail in module['details']:
                ax.text(pos, y_offset, detail, fontsize=7,
                        ha='center', va='center')
                y_offset -= 0.6
                
    def _draw_connections(self, ax):
        """绘制层级之间的连接线"""
        # 层与层之间的主连接
        connections = [
            (50, 100, 50, 97),    # 层1 -> 层2
            (50, 85, 50, 82),     # 层2 -> 层3
            (50, 65, 50, 62),     # 层3 -> 层4
            (50, 50, 50, 47),     # 层4 -> 层5
            (50, 20, 50, 17)      # 层5 -> 层6
        ]
        
        for x1, y1, x2, y2 in connections:
            arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                                   arrowstyle="->", shrinkA=0, shrinkB=0,
                                   mutation_scale=20, fc=self.colors['arrow'],
                                   linewidth=2)
            ax.add_artist(arrow)
            
        # 层3到层5的并行连接
        layer3_outputs = [20, 40, 60, 80]
        layer5_inputs = [20, 40, 60, 80]
        
        for out, inp in zip(layer3_outputs, layer5_inputs):
            arrow = ConnectionPatch((out, 66), (inp, 47), "data", "data",
                                   arrowstyle="->", shrinkA=0, shrinkB=0,
                                   mutation_scale=15, fc=self.colors['arrow'],
                                   linewidth=1.5, alpha=0.7)
            ax.add_artist(arrow)
            
        # 层5内部的交互连接（虚线）
        # H1->H2, H2->H3, H3->H4
        h_connections = [(20, 25, 40, 25), (40, 25, 60, 25), (60, 25, 80, 25)]
        for x1, y1, x2, y2 in h_connections:
            arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                                   arrowstyle="->", shrinkA=0, shrinkB=0,
                                   mutation_scale=10, fc=self.colors['arrow'],
                                   linewidth=1, linestyle='dashed', alpha=0.5)
            ax.add_artist(arrow)
            
        # 反馈路径 H4->H2
        arrow = ConnectionPatch((80, 30), (40, 30), "data", "data",
                               arrowstyle="->", shrinkA=0, shrinkB=0,
                               mutation_scale=10, fc='red',
                               linewidth=1, linestyle='dashed', alpha=0.5)
        ax.add_artist(arrow)
        
    def _get_text(self, layer):
        """获取不同语言的文本内容"""
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
                        {
                            'title': '框架激活量化',
                            'details': [
                                '18种具体框架',
                                '→4大类',
                                '激活强度(1-7)',
                                '语境依赖度',
                                '机构预设度'
                            ]
                        },
                        {
                            'title': '识解操作编码',
                            'details': [
                                '具体性/图式性',
                                '聚焦(1-5)',
                                '突显(1-5)',
                                '视角(0-1)',
                                'PCA综合指数'
                            ]
                        },
                        {
                            'title': '策略选择分类',
                            'details': [
                                '初始四类',
                                '合并为三类',
                                '策略效能',
                                '适应指数'
                            ]
                        },
                        {
                            'title': '协商动态追踪',
                            'details': [
                                '五类协商点',
                                '贡献率计算',
                                '语义距离',
                                'Word2Vec'
                            ]
                        }
                    ]
                },
                'layer4': {
                    'main': '第四层：数据预处理与质量控制',
                    'modules': [
                        {
                            'title': '变量转换',
                            'details': [
                                '连续变量组均值中心化',
                                '分类变量效应编码(-1,0,1)',
                                '相对位置标准化(0-1)'
                            ]
                        },
                        {
                            'title': '质量保证',
                            'details': [
                                '双编码员独立标注',
                                '信度检验(κ=0.82-0.89)',
                                '缺失数据处理(m=5)',
                                '异常值识别'
                            ]
                        }
                    ]
                },
                'layer5': {
                    'main': '第五层：四个平行假设检验模块',
                    'hypotheses': [
                        {
                            'title': 'H1',
                            'subtitle': '框架激活的双重机制',
                            'method': '三层线性混合模型',
                            'output': '相对影响力动态'
                        },
                        {
                            'title': 'H2',
                            'subtitle': '框架驱动的策略选择',
                            'method': '多层多项逻辑回归',
                            'output': '策略选择预测'
                        },
                        {
                            'title': 'H3',
                            'subtitle': '策略演化的路径依赖',
                            'method': '马尔可夫链+面板',
                            'output': '路径依赖模式'
                        },
                        {
                            'title': 'H4',
                            'subtitle': '意义协商的语义收敛',
                            'method': '分段增长曲线',
                            'output': '收敛轨迹转折点'
                        }
                    ]
                },
                'layer6': {
                    'main': '第六层：理论整合与模型构建',
                    'modules': [
                        {
                            'title': '扩展路径分析',
                            'details': [
                                'H1→H2→H3→H4',
                                '反馈路径H4→H2',
                                '间接效应估计'
                            ]
                        },
                        {
                            'title': '模型评估',
                            'details': [
                                'χ²df3, CFI0.95',
                                'RMSEA0.06',
                                '跨群体不变性'
                            ]
                        },
                        {
                            'title': '理论贡献',
                            'details': [
                                '认知-制度机制',
                                '动态策略理论',
                                '语义收敛模型'
                            ]
                        }
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
                    'main': 'Layer 2: Data Restructuring & Annotation System (XML-JSON Hybrid)',
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
                        {
                            'title': 'Frame Activation',
                            'details': [
                                '18 specific frames',
                                '→4 categories',
                                'Strength (1-7)',
                                'Context depend.',
                                'Institutional preset'
                            ]
                        },
                        {
                            'title': 'Construal Coding',
                            'details': [
                                'Specificity/Schema',
                                'Focusing (1-5)',
                                'Prominence (1-5)',
                                'Perspective (0-1)',
                                'PCA composite'
                            ]
                        },
                        {
                            'title': 'Strategy Selection',
                            'details': [
                                'Initial 4 types',
                                'Merged to 3',
                                'Efficacy metric',
                                'Adaptation index'
                            ]
                        },
                        {
                            'title': 'Negotiation Tracking',
                            'details': [
                                '5 negotiation types',
                                'Contribution ratio',
                                'Semantic distance',
                                'Word2Vec cosine'
                            ]
                        }
                    ]
                },
                'layer4': {
                    'main': 'Layer 4: Data Preprocessing & Quality Control',
                    'modules': [
                        {
                            'title': 'Variable Transform',
                            'details': [
                                'Group-mean centering',
                                'Effect coding (-1,0,1)',
                                'Position standardization'
                            ]
                        },
                        {
                            'title': 'Quality Assurance',
                            'details': [
                                'Dual independent coding',
                                'Reliability (κ=0.82-0.89)',
                                'Multiple imputation (m=5)',
                                'Outlier detection'
                            ]
                        }
                    ]
                },
                'layer5': {
                    'main': 'Layer 5: Four Parallel Hypothesis Testing Modules',
                    'hypotheses': [
                        {
                            'title': 'H1',
                            'subtitle': 'Dual Mechanism of Frame',
                            'method': '3-level LMM',
                            'output': 'Dynamic influence'
                        },
                        {
                            'title': 'H2',
                            'subtitle': 'Frame-driven Strategy',
                            'method': 'Multilevel MLR',
                            'output': 'Strategy prediction'
                        },
                        {
                            'title': 'H3',
                            'subtitle': 'Path Dependency',
                            'method': 'Markov+Panel',
                            'output': 'Path patterns'
                        },
                        {
                            'title': 'H4',
                            'subtitle': 'Semantic Convergence',
                            'method': 'Piecewise growth',
                            'output': 'Convergence trajectory'
                        }
                    ]
                },
                'layer6': {
                    'main': 'Layer 6: Theoretical Integration & Model Building',
                    'modules': [
                        {
                            'title': 'Path Analysis',
                            'details': [
                                'H1→H2→H3→H4',
                                'Feedback H4→H2',
                                'Indirect effects'
                            ]
                        },
                        {
                            'title': 'Model Evaluation',
                            'details': [
                                'χ²df3, CFI0.95',
                                'RMSEA0.06',
                                'Invariance test'
                            ]
                        },
                        {
                            'title': 'Contributions',
                            'details': [
                                'Cognitive-institutional',
                                'Dynamic strategy',
                                'Convergence model'
                            ]
                        }
                    ]
                }
            }
        }
        
        return texts[self.language][layer]
        
    def _save_figure(self, fig):
        """保存图形"""
        # 确定输出路径
        output_dir = OUTPUT_DIR_ZH if self.language == 'zh' else OUTPUT_DIR_EN
        filename = 'spaadia_multilevel_analysis_flowchart.jpg'
        filepath = output_dir / filename
        
        # 保存高分辨率图片
        fig.savefig(filepath, dpi=1200, bbox_inches='tight', 
                   facecolor='white', format='jpg')
        
        print(f"图片已保存: {filepath}")
        
        # 同时保存分析元数据
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
    """主函数：生成中英文两个版本的流程图"""
    print("="*60)
    print("SPAADIA语料库多层次分析流程图生成")
    print("="*60)
    
    # 生成中文版本
    print("\n正在生成中文版本...")
    generator_zh = FlowchartGenerator(language='zh')
    generator_zh.create_flowchart()
    
    # 生成英文版本
    print("\n正在生成英文版本...")
    generator_en = FlowchartGenerator(language='en')
    generator_en.create_flowchart()
    
    print("\n流程图生成完成！")
    print(f"\n输出位置:")
    print(f"  中文版: {OUTPUT_DIR_ZH}")
    print(f"  英文版: {OUTPUT_DIR_EN}")
    
if __name__ == "__main__":
    main()