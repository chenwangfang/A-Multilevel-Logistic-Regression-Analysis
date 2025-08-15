#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPAADIA分析脚本总运行程序（高级版）
包含基础分析和高级统计分析
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# 设置日志
log_dir = Path("G:/Project/实证/关联框架/输出/logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('AdvancedAnalysisRunner')

class AdvancedAnalysisRunner:
    """高级分析运行器"""
    
    def __init__(self, run_basic: bool = True, run_advanced: bool = True):
        """
        初始化运行器
        
        Parameters:
        -----------
        run_basic : bool
            是否运行基础分析
        run_advanced : bool
            是否运行高级分析
        """
        self.run_basic = run_basic
        self.run_advanced = run_advanced
        self.script_dir = Path("G:/Project/实证/关联框架/Python脚本/SPAADIA分析脚本")
        
        # 基础分析脚本列表
        self.basic_scripts = [
            "section_3_1_analysis.py",  # 3.1节分析
            "hypothesis_h1_analysis.py",  # H1假设
            "hypothesis_h2_analysis.py",  # H2假设
            "hypothesis_h3_analysis.py",  # H3假设
            "hypothesis_h4_analysis.py"   # H4假设
        ]
        
        # 高级分析脚本列表
        self.advanced_scripts = [
            "hypothesis_h1_advanced.py",  # H1高级分析（随机斜率）
            "hypothesis_h2_advanced.py",  # H2高级分析（效应编码）
            "hypothesis_h3_advanced.py",  # H3高级分析（增强马尔可夫）
            "hypothesis_h4_advanced.py"   # H4高级分析（五断点+Word2Vec）
        ]
        
        # 验证脚本
        self.validation_scripts = [
            "validation_scripts.R",  # R语言验证
            "integrate_r_validation.py"  # 整合R验证结果
        ]
        
        self.results = {}
        
    def check_dependencies(self):
        """检查依赖项"""
        logger.info("检查Python依赖项...")
        
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'scipy', 'statsmodels',
            'sklearn', 'networkx', 'seaborn', 'lifelines',
            'gensim', 'jieba', 'patsy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package} 已安装")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} 未安装")
        
        if missing_packages:
            logger.error(f"缺少以下依赖包: {', '.join(missing_packages)}")
            logger.info("请运行: pip install " + ' '.join(missing_packages))
            return False
        
        # 检查R环境
        logger.info("检查R环境...")
        try:
            result = subprocess.run(['R', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ R环境已安装")
            else:
                logger.warning("✗ R环境未安装或不在PATH中")
        except:
            logger.warning("✗ 无法检测R环境")
        
        return True
    
    def run_script(self, script_name: str) -> bool:
        """
        运行单个脚本
        
        Parameters:
        -----------
        script_name : str
            脚本名称
            
        Returns:
        --------
        bool
            是否成功运行
        """
        script_path = self.script_dir / script_name
        
        if not script_path.exists():
            logger.error(f"脚本不存在: {script_path}")
            return False
        
        logger.info(f"运行脚本: {script_name}")
        start_time = time.time()
        
        try:
            # 根据文件扩展名选择运行方式
            if script_name.endswith('.py'):
                # Python脚本
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
            elif script_name.endswith('.R'):
                # R脚本
                result = subprocess.run(
                    ['Rscript', str(script_path)],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
            else:
                logger.error(f"不支持的脚本类型: {script_name}")
                return False
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✓ {script_name} 运行成功 (耗时: {elapsed_time:.2f}秒)")
                self.results[script_name] = {
                    'status': 'success',
                    'time': elapsed_time,
                    'output': result.stdout
                }
                return True
            else:
                logger.error(f"✗ {script_name} 运行失败")
                logger.error(f"错误信息: {result.stderr}")
                self.results[script_name] = {
                    'status': 'failed',
                    'time': elapsed_time,
                    'error': result.stderr
                }
                return False
                
        except Exception as e:
            logger.error(f"运行脚本时发生异常: {e}")
            self.results[script_name] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def run_basic_analyses(self):
        """运行基础分析"""
        logger.info("="*50)
        logger.info("开始运行基础分析...")
        logger.info("="*50)
        
        success_count = 0
        
        for script in self.basic_scripts:
            if self.run_script(script):
                success_count += 1
            time.sleep(1)  # 短暂延迟，避免资源冲突
        
        logger.info(f"基础分析完成: {success_count}/{len(self.basic_scripts)} 个脚本成功运行")
        return success_count == len(self.basic_scripts)
    
    def run_advanced_analyses(self):
        """运行高级分析"""
        logger.info("="*50)
        logger.info("开始运行高级统计分析...")
        logger.info("="*50)
        
        success_count = 0
        
        for script in self.advanced_scripts:
            if self.run_script(script):
                success_count += 1
            time.sleep(1)
        
        logger.info(f"高级分析完成: {success_count}/{len(self.advanced_scripts)} 个脚本成功运行")
        return success_count == len(self.advanced_scripts)
    
    def run_validation(self):
        """运行验证脚本"""
        logger.info("="*50)
        logger.info("开始运行验证脚本...")
        logger.info("="*50)
        
        success_count = 0
        
        for script in self.validation_scripts:
            if self.run_script(script):
                success_count += 1
            time.sleep(1)
        
        logger.info(f"验证完成: {success_count}/{len(self.validation_scripts)} 个脚本成功运行")
        return success_count == len(self.validation_scripts)
    
    def generate_summary_report(self):
        """生成汇总报告"""
        logger.info("生成汇总报告...")
        
        report_path = Path("G:/Project/实证/关联框架/输出/reports/advanced_analysis_summary.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# SPAADIA高级分析汇总报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 运行结果汇总
            f.write("## 运行结果汇总\n\n")
            
            # 基础分析结果
            if self.run_basic:
                f.write("### 基础分析\n\n")
                f.write("| 脚本名称 | 状态 | 耗时(秒) |\n")
                f.write("|---------|------|----------|\n")
                
                for script in self.basic_scripts:
                    if script in self.results:
                        result = self.results[script]
                        status = "✓ 成功" if result['status'] == 'success' else "✗ 失败"
                        time_str = f"{result.get('time', 0):.2f}" if 'time' in result else '-'
                        f.write(f"| {script} | {status} | {time_str} |\n")
                
                f.write("\n")
            
            # 高级分析结果
            if self.run_advanced:
                f.write("### 高级分析\n\n")
                f.write("| 脚本名称 | 状态 | 耗时(秒) | 主要改进 |\n")
                f.write("|---------|------|----------|----------|\n")
                
                improvements = {
                    'hypothesis_h1_advanced.py': '随机斜率、多重插补、Bootstrap',
                    'hypothesis_h2_advanced.py': '效应编码、对应分析',
                    'hypothesis_h3_advanced.py': '增强马尔可夫、生存分析、网络分析',
                    'hypothesis_h4_advanced.py': '五断点模型、Word2Vec、CUSUM'
                }
                
                for script in self.advanced_scripts:
                    if script in self.results:
                        result = self.results[script]
                        status = "✓ 成功" if result['status'] == 'success' else "✗ 失败"
                        time_str = f"{result.get('time', 0):.2f}" if 'time' in result else '-'
                        improvement = improvements.get(script, '-')
                        f.write(f"| {script} | {status} | {time_str} | {improvement} |\n")
                
                f.write("\n")
            
            # 验证结果
            f.write("### 验证分析\n\n")
            f.write("| 脚本名称 | 状态 | 说明 |\n")
            f.write("|---------|------|------|\n")
            
            for script in self.validation_scripts:
                if script in self.results:
                    result = self.results[script]
                    status = "✓ 成功" if result['status'] == 'success' else "✗ 失败"
                    desc = "R语言验证" if script.endswith('.R') else "Python-R整合"
                    f.write(f"| {script} | {status} | {desc} |\n")
            
            f.write("\n## 输出文件清单\n\n")
            
            # 列出主要输出
            f.write("### 表格文件\n")
            f.write("- 表1-4: 描述性统计（section_3_1_analysis）\n")
            f.write("- 表5-6: H1假设结果（含高级版）\n")
            f.write("- 表7-8: H2假设结果（含高级版）\n")
            f.write("- 表9-10: H3假设结果（含高级版）\n")
            f.write("- 表11-12: H4假设结果（含高级版）\n\n")
            
            f.write("### 图形文件\n")
            f.write("- 图1: 对话结构分析\n")
            f.write("- 图2: 框架激活双重机制\n")
            f.write("- 图3: 框架-策略关联\n")
            f.write("- 图4: 策略演化动态\n")
            f.write("- 图5: 意义协商过程\n\n")
            
            f.write("### 数据文件\n")
            f.write("- JSON格式: 所有统计结果\n")
            f.write("- CSV格式: 所有表格数据\n\n")
            
            f.write("### 报告文件\n")
            f.write("- 各假设分析报告（基础版+高级版）\n")
            f.write("- R验证整合报告\n")
            f.write("- 本汇总报告\n\n")
            
            # 总结
            total_scripts = len(self.basic_scripts) + len(self.advanced_scripts) + len(self.validation_scripts)
            success_scripts = sum(1 for r in self.results.values() if r['status'] == 'success')
            
            f.write(f"## 总结\n\n")
            f.write(f"- 总计运行脚本: {total_scripts} 个\n")
            f.write(f"- 成功运行: {success_scripts} 个\n")
            f.write(f"- 成功率: {success_scripts/total_scripts*100:.1f}%\n")
            f.write(f"- 日志文件: {log_file}\n")
        
        logger.info(f"汇总报告已保存至: {report_path}")
    
    def run(self):
        """运行所有分析"""
        logger.info("SPAADIA高级分析开始运行...")
        logger.info(f"运行配置: 基础分析={self.run_basic}, 高级分析={self.run_advanced}")
        
        # 检查依赖
        if not self.check_dependencies():
            logger.error("依赖检查失败，请安装缺失的包后重试")
            return False
        
        # 记录开始时间
        total_start_time = time.time()
        
        # 运行基础分析
        if self.run_basic:
            self.run_basic_analyses()
        
        # 运行高级分析
        if self.run_advanced:
            # 确保高级统计工具模块存在
            advanced_stats_path = self.script_dir / "advanced_statistics.py"
            if not advanced_stats_path.exists():
                logger.error("缺少高级统计工具模块: advanced_statistics.py")
                logger.info("请确保该文件存在后再运行高级分析")
                return False
            
            self.run_advanced_analyses()
        
        # 运行验证
        self.run_validation()
        
        # 生成汇总报告
        self.generate_summary_report()
        
        # 计算总耗时
        total_elapsed = time.time() - total_start_time
        logger.info(f"所有分析完成！总耗时: {total_elapsed/60:.2f} 分钟")
        
        return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SPAADIA高级分析运行器')
    parser.add_argument('--skip-basic', action='store_true', 
                       help='跳过基础分析，仅运行高级分析')
    parser.add_argument('--skip-advanced', action='store_true',
                       help='跳过高级分析，仅运行基础分析')
    parser.add_argument('--language', choices=['zh', 'en'], default='zh',
                       help='输出语言（默认：中文）')
    
    args = parser.parse_args()
    
    # 检查参数逻辑
    if args.skip_basic and args.skip_advanced:
        logger.error("不能同时跳过基础分析和高级分析")
        return
    
    # 创建运行器
    runner = AdvancedAnalysisRunner(
        run_basic=not args.skip_basic,
        run_advanced=not args.skip_advanced
    )
    
    # 运行分析
    success = runner.run()
    
    if success:
        print("\n" + "="*50)
        print("✓ SPAADIA高级分析全部完成！")
        print("="*50)
        print(f"输出目录: G:/Project/实证/关联框架/输出/")
        print(f"日志文件: {log_file}")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("✗ 部分分析失败，请查看日志文件")
        print("="*50)

if __name__ == "__main__":
    main()