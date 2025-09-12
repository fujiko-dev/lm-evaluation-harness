#!/usr/bin/env python3
"""
Commit Message Generation 评估器

统一的评估脚本，支持所有可配置参数和多种模型
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

# 导入模型以确保注册
import lm_eval.models.openai_completions

from lm_eval.api.registry import get_model
from .config import CommitMessageGenerationConfig, MODEL_CONFIGS, get_model_config
from .lca_commit_message_generation_task import LCACommitMessageGenerationTask


class CommitMessageGenerationEvaluator:
    """Commit Message Generation 评估器"""
    
    def __init__(self, config: CommitMessageGenerationConfig):
        self.config = config
        self.output_dir = None
        
    def setup_output_directory(self) -> Path:
        """设置输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config.output_dir / f"eval_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = output_dir
        return output_dir
    
    def setup_environment(self):
        """设置环境变量"""
        # 检查必需的API密钥
        model_name = self.config.model_name.lower()
        
        if 'openai' in model_name or 'qianfan' in self.config.get('model_params', {}).get('base_url', ''):
            if not os.getenv('OPENAI_API_KEY'):
                raise EnvironmentError(
                    "缺少环境变量 OPENAI_API_KEY。\n"
                    "请设置: export OPENAI_API_KEY=your_api_key"
                )
        elif 'anthropic' in model_name:
            if not os.getenv('ANTHROPIC_API_KEY'):
                raise EnvironmentError(
                    "缺少环境变量 ANTHROPIC_API_KEY。\n"
                    "请设置: export ANTHROPIC_API_KEY=your_api_key"
                )
    
    def create_model(self):
        """创建模型"""
        print(f"🤖 正在初始化模型: {self.config.model_name}")
        
        # 获取模型参数字符串
        model_args_str = self.config.get_model_args_string()
        print(f"   参数: {model_args_str}")
        
        # 创建模型
        model = get_model(self.config.model_name).create_from_arg_string(
            model_args_str,
            {
                "batch_size": self.config.batch_size,
                "device": "auto"
            }
        )
        
        return model
    
    def run_evaluation(self) -> Optional[Dict[str, Any]]:
        """运行评估"""
        print("🚀 开始 Commit Message Generation 评估")
        print("=" * 60)
        print(f"模型: {self.config.model_name}")
        print(f"数据拆分: {self.config.get('split')}")
        print(f"数据集路径: {self.config.get('dataset_path')}")
        print(f"批处理大小: {self.config.get('batch_size')}")
        if self.config.get('limit'):
            print(f"样本限制: {self.config.get('limit')}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
        
        try:
            # 设置环境
            self.setup_environment()
            
            # 创建模型
            model = self.create_model()
            print("✅ 模型初始化成功")
            
            # 直接使用我们的任务类
            print(f"📋 初始化任务...")
            task = LCACommitMessageGenerationTask(self.config.to_dict())
            print("✅ 任务初始化成功")
            
            # 加载测试数据
            print("📊 加载测试数据...")
            test_docs = list(task.test_docs())
            print(f"✅ 加载了 {len(test_docs)} 个测试样本")
            
            # 应用限制
            limit = self.config.get('limit')
            if limit:
                test_docs = test_docs[:limit]
                print(f"📊 限制为 {len(test_docs)} 个样本")
            
            # 开始评估
            print(f"🔍 开始处理 {len(test_docs)} 个样本...")
            start_time = time.time()
            
            results = []
            for i, doc in enumerate(test_docs):
                print(f"\n📝 处理样本 {i+1}/{len(test_docs)}: {doc.get('repo', f'sample_{i}')} ({doc.get('commit_sha', '')[:8]})")
                
                try:
                    # 生成提示
                    text = task.doc_to_text(doc)
                    
                    # 构建请求
                    requests = task.construct_requests(doc, text)
                    
                    if requests:
                        # 调用模型
                        response = model.generate_until(
                            requests=[requests[0]],
                            disable_tqdm=True
                        )
                        
                        # 处理结果
                        result = task.process_results(doc, response)
                        results.append(result)
                        
                        print(f"   ✅ BLEU: {result.get('commit_bleu', 0):.4f}, ROUGE-L: {result.get('commit_rougeL', 0):.4f}")
                    
                except Exception as e:
                    print(f"   ❌ 处理失败: {e}")
                    failed_result = {
                        'idx': doc.get('idx', f'sample_{i}'),
                        'error': str(e),
                        'commit_bleu': 0.0,
                        'commit_rouge1': 0.0,
                        'commit_rouge2': 0.0,
                        'commit_rougeL': 0.0,
                        'commit_bert_f1': 0.0,
                        'length_ratio': 1.0
                    }
                    results.append(failed_result)
            
            # 模拟lm_eval结果格式
            task_name = "lca_commit_message_generation"
            formatted_results = {
                'results': {
                    task_name: self._calculate_aggregated_metrics(results)
                },
                'samples': {
                    task_name: results
                }
            }
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n✅ 评估完成，耗时: {duration:.2f} 秒")
            
            # 保存结果
            self.save_results(formatted_results, duration)
            
            # 打印结果摘要
            self.print_results_summary(formatted_results)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 保存错误信息
            if self.output_dir:
                error_file = self.output_dir / "error_log.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {str(e)}\n\n")
                    f.write(traceback.format_exc())
            
            return None
    
    def _calculate_aggregated_metrics(self, results):
        """计算聚合指标"""
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return {}
        
        total_bleu = sum(r.get('commit_bleu', 0) for r in valid_results)
        total_rouge1 = sum(r.get('commit_rouge1', 0) for r in valid_results)
        total_rouge2 = sum(r.get('commit_rouge2', 0) for r in valid_results)
        total_rougeL = sum(r.get('commit_rougeL', 0) for r in valid_results)
        total_bert_f1 = sum(r.get('commit_bert_f1', 0) for r in valid_results)
        total_length_ratio = sum(r.get('length_ratio', 0) for r in valid_results)
        
        return {
            'commit_bleu': total_bleu / len(valid_results),
            'commit_rouge1': total_rouge1 / len(valid_results),
            'commit_rouge2': total_rouge2 / len(valid_results),
            'commit_rougeL': total_rougeL / len(valid_results),
            'commit_bert_f1': total_bert_f1 / len(valid_results),
            'length_ratio': total_length_ratio / len(valid_results),
            'valid_samples': len(valid_results),
            'total_samples': len(results)
        }
    
    def save_results(self, results: Dict[str, Any], duration: float):
        """保存评估结果"""
        # 保存主要结果
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 结果已保存: {results_file}")
        
        # 保存配置
        config_file = self.output_dir / "evaluation_config.json"
        config_data = self.config.to_dict()
        config_data['evaluation_duration'] = duration
        config_data['timestamp'] = datetime.now().isoformat()
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"💾 配置已保存: {config_file}")
        
        # 保存摘要
        summary = self.extract_summary(results)
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"💾 摘要已保存: {summary_file}")
    
    def extract_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """提取评估摘要"""
        summary = {
            'model': self.config.model_name,
            'split': self.config.get('split'),
            'timestamp': datetime.now().isoformat()
        }
        
        # 提取主要指标
        for task_name, task_results in results.get('results', {}).items():
            summary[f'{task_name}_metrics'] = {
                'bleu': task_results.get('commit_bleu', 0.0),
                'rouge1': task_results.get('commit_rouge1', 0.0),
                'rouge2': task_results.get('commit_rouge2', 0.0),
                'rougeL': task_results.get('commit_rougeL', 0.0),
                'bert_f1': task_results.get('commit_bert_f1', 0.0),
                'length_ratio': task_results.get('length_ratio', 0.0)
            }
        
        # 样本统计
        samples_info = results.get('samples', {})
        total_samples = sum(len(samples) for samples in samples_info.values())
        summary['total_samples'] = total_samples
        
        return summary
    
    def print_results_summary(self, results: Dict[str, Any]):
        """打印结果摘要"""
        print("\n" + "=" * 60)
        print("📊 评估结果摘要")
        print("=" * 60)
        
        for task_name, task_results in results.get('results', {}).items():
            print(f"\n任务: {task_name}")
            print("-" * 40)
            
            # 主要指标
            main_metrics = [
                ('BLEU Score', 'commit_bleu'),
                ('ROUGE-1', 'commit_rouge1'),
                ('ROUGE-2', 'commit_rouge2'),
                ('ROUGE-L', 'commit_rougeL'),
                ('BERT F1', 'commit_bert_f1'),
                ('Length Ratio', 'length_ratio')
            ]
            
            for label, key in main_metrics:
                if key in task_results:
                    value = task_results[key]
                    if isinstance(value, float):
                        print(f"  {label:<15}: {value:.4f}")
                    else:
                        print(f"  {label:<15}: {value}")
        
        # 采样信息
        samples_info = results.get('samples', {})
        total_samples = sum(len(samples) for samples in samples_info.values())
        print(f"\n总样本数: {total_samples}")
        
        print("\n" + "=" * 60)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='Commit Message Generation 评估器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 使用预设模型配置:
   python run_commit_message_generation_eval.py --model gpt-4o

2. 使用千帆模型:
   python run_commit_message_generation_eval.py --model qianfan-qwen

3. 自定义配置:
   python run_commit_message_generation_eval.py --model openai-chat-completions \\
       --model_args "model=gpt-4o,temperature=0.1" \\
       --limit 10

4. 从配置文件运行:
   python run_commit_message_generation_eval.py --config config.json

5. 查看可用模型:
   python run_commit_message_generation_eval.py --list_models
        """
    )
    
    # 模型配置
    parser.add_argument('--model', type=str, 
                       help='模型名称或预设配置键')
    parser.add_argument('--model_args', type=str,
                       help='自定义模型参数字符串 (格式: key1=value1,key2=value2)')
    
    # 任务配置
    parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'],
                       default='test', help='数据拆分 (默认: test)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批处理大小 (默认: 1)')
    parser.add_argument('--limit', type=int,
                       help='限制样本数量（用于测试）')
    
    # 路径配置
    parser.add_argument('--dataset_path', type=str,
                       help='数据集路径 (覆盖配置文件)')
    parser.add_argument('--output_dir', type=str,
                       help='输出目录 (覆盖配置文件)')
    
    # 配置文件
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    parser.add_argument('--save_config', type=str,
                       help='保存当前配置到文件')
    
    # 其他选项
    parser.add_argument('--list_models', action='store_true',
                       help='列出可用的预设模型配置')
    parser.add_argument('--validate_config', action='store_true',
                       help='验证配置有效性')
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list_models:
        print("可用的预设模型配置:")
        print("=" * 40)
        for key, model_config in MODEL_CONFIGS.items():
            print(f"\n{key}:")
            print(f"  模型名称: {model_config['model_name']}")
            params = model_config['model_params']
            for param_key, param_value in params.items():
                if param_value is not None:
                    print(f"  {param_key}: {param_value}")
        return
    
    # 检查必需参数
    if not args.config and not args.model:
        parser.error('必须指定 --model 或 --config')
    
    try:
        # 创建配置
        if args.config:
            print(f"📄 从配置文件加载: {args.config}")
            config = CommitMessageGenerationConfig.from_file(args.config)
        elif args.model in MODEL_CONFIGS:
            print(f"🎯 使用预设配置: {args.model}")
            config = get_model_config(args.model)
        else:
            print(f"🔧 使用自定义配置")
            config = CommitMessageGenerationConfig()
            config.set('model_name', args.model)
        
        # 应用命令行参数覆盖
        overrides = {}
        if args.model_args:
            # 解析模型参数字符串
            model_params = {}
            for param in args.model_args.split(','):
                if '=' in param:
                    key, value = param.split('=', 1)
                    # 尝试转换类型
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)
                    model_params[key.strip()] = value
            overrides['model_params'] = model_params
        
        # 其他参数覆盖
        if args.split:
            overrides['split'] = args.split
        if args.batch_size:
            overrides['batch_size'] = args.batch_size
        if args.limit:
            overrides['limit'] = args.limit
        if args.dataset_path:
            overrides['dataset_path'] = args.dataset_path
        if args.output_dir:
            overrides['output_dir'] = args.output_dir
        
        # 应用覆盖
        if overrides:
            config.update(overrides)
        
        # 验证配置
        if args.validate_config or not config.validate():
            if not config.validate():
                print("❌ 配置验证失败")
                return
            else:
                print("✅ 配置验证通过")
                if args.validate_config:
                    return
        
        # 保存配置
        if args.save_config:
            config.save_to_file(args.save_config)
            print(f"💾 配置已保存到: {args.save_config}")
            return
        
        # 创建评估器并运行
        evaluator_instance = CommitMessageGenerationEvaluator(config)
        evaluator_instance.setup_output_directory()
        
        results = evaluator_instance.run_evaluation()
        
        if results:
            print(f"\n🎉 评估成功完成！")
            print(f"结果保存在: {evaluator_instance.output_dir}")
        else:
            print(f"\n❌ 评估失败")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
