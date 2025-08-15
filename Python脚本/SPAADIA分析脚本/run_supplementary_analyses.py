#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有补充分析
Run All Supplementary Analyses for SPAADIA Framework

运行2.4小节要求但之前缺失的分析：
1. 统计功效分析
2. FDR多重比较校正
3. 敏感性分析
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupplementaryAnalysesRunner:
    """补充分析运行器"""
    
    def __init__(self):
        """初始化运行器"""
        self.start_time = datetime.now()
        
        # 输出目录
        self.output_dir = Path('/mnt/g/Project/实证/关联框架/输出')
        self.logs_dir = self.output_dir / 'logs'
        self.reports_dir = self.output_dir / 'reports'
        
        # 确保目录存在
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 分析脚本列表
        self.analyses = [
            {
                'name': '统计功效分析',
                'script': 'power_analysis.py',
                'description': '评估各假设检验的统计功效',
                'required': True
            },
            {
                'name': 'FDR多重比较校正',
                'script': 'fdr_correction.py',
                'description': '控制多重比较的错误发现率',
                'required': True
            },
            {
                'name': '敏感性分析',
                'script': 'sensitivity_analysis.py',
                'description': '评估方法选择对结果的影响',
                'required': True
            }
        ]
        
        self.results = {}
        
    def run_analysis(self, analysis_info: dict) -> bool:
        """运行单个分析"""
        name = analysis_info['name']
        script = analysis_info['script']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"运行：{name}")
        logger.info(f"脚本：{script}")
        logger.info(f"描述：{analysis_info['description']}")
        logger.info('='*60)
        
        script_path = Path(__file__).parent / script
        
        if not script_path.exists():
            logger.error(f"脚本不存在：{script_path}")
            return False
        
        try:
            # 运行Python脚本
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                logger.info(f"✓ {name} 完成")
                self.results[name] = {
                    'status': 'success',
                    'script': script,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 保存输出日志
                log_file = self.logs_dir / f"{script.replace('.py', '')}_output.log"
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== {name} 输出 ===\n")
                    f.write(f"时间：{datetime.now()}\n\n")
                    f.write("标准输出：\n")
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n标准错误：\n")
                        f.write(result.stderr)
                
                return True
            else:
                logger.error(f"✗ {name} 失败")
                logger.error(f"错误信息：{result.stderr}")
                self.results[name] = {
                    'status': 'failed',
                    'script': script,
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
                return False
                
        except Exception as e:
            logger.error(f"运行 {name} 时发生错误：{e}")
            self.results[name] = {
                'status': 'error',
                'script': script,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_all_analyses(self):
        """运行所有补充分析"""
        logger.info("\n" + "="*70)
        logger.info("开始运行SPAADIA框架补充分析")
        logger.info("="*70)
        
        success_count = 0
        failed_count = 0
        
        for analysis in self.analyses:
            if self.run_analysis(analysis):
                success_count += 1
            else:
                failed_count += 1
                if analysis.get('required', False):
                    logger.warning(f"关键分析 {analysis['name']} 失败，继续运行其他分析...")
        
        # 生成汇总报告
        self.generate_summary_report(success_count, failed_count)
        
        # 计算总耗时
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info("补充分析运行完成")
        logger.info(f"成功：{success_count} 个")
        logger.info(f"失败：{failed_count} 个")
        logger.info(f"总耗时：{total_time:.2f} 秒")
        logger.info("="*70)
        
        return success_count, failed_count
    
    def generate_summary_report(self, success_count: int, failed_count: int):
        """生成汇总报告"""
        logger.info("\n生成补充分析汇总报告...")
        
        report = ["# SPAADIA框架补充分析汇总报告\n\n"]
        report.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report.append("## 运行概况\n\n")
        report.append(f"- 总分析数：{len(self.analyses)}\n")
        report.append(f"- 成功运行：{success_count}\n")
        report.append(f"- 运行失败：{failed_count}\n")
        report.append(f"- 总耗时：{(datetime.now() - self.start_time).total_seconds():.2f} 秒\n\n")
        
        report.append("## 分析结果\n\n")
        report.append("| 分析名称 | 状态 | 脚本 | 时间 | 备注 |\n")
        report.append("|---------|------|------|------|------|\n")
        
        for name, result in self.results.items():
            status_icon = "✓" if result['status'] == 'success' else "✗"
            report.append(f"| {name} | {status_icon} {result['status']} | ")
            report.append(f"{result['script']} | ")
            report.append(f"{result['timestamp'].split('T')[1].split('.')[0]} | ")
            
            if result['status'] != 'success':
                error_msg = result.get('error', '')[:50]
                report.append(f"{error_msg}... |")
            else:
                report.append("成功完成 |")
            report.append("\n")
        
        report.append("\n## 输出文件\n\n")
        report.append("### 统计功效分析\n")
        report.append("- 报告：`reports/power_analysis_report.md`\n")
        report.append("- 数据：`data/power_analysis_results.json`\n")
        report.append("- 表格：`tables/power_analysis_summary.csv`\n\n")
        
        report.append("### FDR校正\n")
        report.append("- 报告：`reports/fdr_correction_report.md`\n")
        report.append("- 数据：`data/fdr_correction_results.json`\n")
        report.append("- 表格：`tables/fdr_correction_results.csv`\n")
        report.append("- 图形：`figures/fdr_correction_plots.png`\n\n")
        
        report.append("### 敏感性分析\n")
        report.append("- 报告：`reports/sensitivity_analysis_report.md`\n")
        report.append("- 数据：`data/sensitivity_analysis_results.json`\n")
        report.append("- 表格：`tables/sensitivity_analysis_summary.csv`\n\n")
        
        report.append("## 关键发现\n\n")
        
        # 尝试读取各分析的关键结果
        key_findings = self._extract_key_findings()
        for finding in key_findings:
            report.append(f"- {finding}\n")
        
        if not key_findings:
            report.append("- 详见各分析的具体报告\n")
        
        report.append("\n## 建议\n\n")
        
        if failed_count == 0:
            report.append("✓ 所有补充分析成功完成，结果可用于论文撰写\n\n")
        else:
            report.append("⚠ 部分分析失败，建议：\n")
            report.append("1. 检查失败分析的错误日志\n")
            report.append("2. 确保所需数据文件存在\n")
            report.append("3. 验证Python环境和依赖包\n\n")
        
        report.append("### 后续步骤\n\n")
        report.append("1. 查看各分析的详细报告\n")
        report.append("2. 将关键结果整合到论文中\n")
        report.append("3. 更新2.4小节的方法描述\n")
        report.append("4. 在结果部分报告功效分析和FDR校正结果\n")
        report.append("5. 在附录中包含敏感性分析\n")
        
        # 保存报告
        report_content = ''.join(report)
        report_path = self.reports_dir / 'supplementary_analyses_summary.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"汇总报告已保存至：{report_path}")
        
        # 保存JSON格式的运行结果
        json_path = self.output_dir / 'data' / 'supplementary_analyses_status.json'
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'run_time': self.start_time.isoformat(),
                'total_analyses': len(self.analyses),
                'success_count': success_count,
                'failed_count': failed_count,
                'results': self.results
            }, f, ensure_ascii=False, indent=2)
        
        return report_content
    
    def _extract_key_findings(self) -> list:
        """提取关键发现"""
        findings = []
        
        # 尝试从各报告中提取关键信息
        try:
            # 功效分析
            power_json = self.output_dir / 'data' / 'power_analysis_results.json'
            if power_json.exists():
                with open(power_json, 'r', encoding='utf-8') as f:
                    power_data = json.load(f)
                    if 'h1' in power_data:
                        power = power_data['h1'].get('power_results', {}).get('fixed_effects_power', 0)
                        if power:
                            findings.append(f"H1假设统计功效：{power:.3f}")
            
            # FDR校正
            fdr_json = self.output_dir / 'data' / 'fdr_correction_results.json'
            if fdr_json.exists():
                with open(fdr_json, 'r', encoding='utf-8') as f:
                    fdr_data = json.load(f)
                    if isinstance(fdr_data, list) and len(fdr_data) > 0:
                        n_sig = sum(1 for item in fdr_data if item.get('rejected', False))
                        findings.append(f"FDR校正后显著结果数：{n_sig}/{len(fdr_data)}")
            
            # 敏感性分析
            sens_json = self.output_dir / 'data' / 'sensitivity_analysis_results.json'
            if sens_json.exists():
                with open(sens_json, 'r', encoding='utf-8') as f:
                    sens_data = json.load(f)
                    if 'random_effects' in sens_data:
                        robust = sens_data['random_effects'].get('robustness', {}).get('conclusion_stable', False)
                        findings.append(f"随机效应结构稳健性：{'高' if robust else '需注意'}")
        
        except Exception as e:
            logger.warning(f"提取关键发现时出错：{e}")
        
        return findings


def main():
    """主函数"""
    runner = SupplementaryAnalysesRunner()
    success_count, failed_count = runner.run_all_analyses()
    
    # 返回运行状态
    if failed_count == 0:
        logger.info("\n✓ 所有补充分析成功完成！")
        return 0
    else:
        logger.warning(f"\n⚠ {failed_count} 个分析失败，请检查日志")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)