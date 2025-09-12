#!/usr/bin/env python3
"""
Project Level Code Completion 评估器

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
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

# 导入模型以确保注册
import lm_eval.models.openai_completions

from lm_eval.api.registry import get_model
from .config import ProjectLevelCodeCompletionConfig, MODEL_CONFIGS, get_model_config
from .lca_project_level_code_completion_task import LCAProjectLevelCodeCompletionTask


class ProjectLevelCodeCompletionEvaluator:
    """Project Level Code Completion 评估器"""
    
    def __init__(self, config: ProjectLevelCodeCompletionConfig):
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
        
        # 直接返回一个简单的API调用器，避免lm_eval的复杂模型系统
        class SimpleAPICaller:
            def __init__(self, config):
                self.config = config
                self.model_params = config.model_params
                
            def generate_until(self, requests, disable_tqdm=False):
                """模拟lm_eval的generate_until接口"""
                import requests as http_requests
                import json
                
                results = []
                for request in requests:
                    try:
                        # 提取提示
                        prompt = request.arguments[0]
                        
                        # 调用API
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
                        }
                        
                        data = {
                            "model": self.model_params.get('model', 'gpt-4.1'),
                            "messages": [
                                {
                                    "role": "user", 
                                    "content": prompt
                                }
                            ],
                            "max_tokens": self.model_params.get('max_tokens', 200),
                            "temperature": self.model_params.get('temperature', 0.8),
                            "top_p": self.model_params.get('top_p', 0.5),
                            "stream": False
                        }
                        
                        response = http_requests.post(
                            self.model_params.get('base_url', 'http://211.23.3.237:27544/v1/chat/completions'),
                            headers=headers,
                            json=data,
                            timeout=self.model_params.get('timeout', 120)
                        )
                        response.raise_for_status()
                        
                        result = response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            generated_text = result["choices"][0]["message"]["content"].strip()
                            results.append(generated_text)
                        else:
                            results.append("")
                            
                    except Exception as e:
                        print(f"⚠️ API调用失败: {e}")
                        results.append("")
                
                return results
        
        return SimpleAPICaller(self.config)
    
    def run_evaluation(self) -> Optional[Dict[str, Any]]:
        """运行评估"""
        print("🚀 开始 Project Level Code Completion 评估")
        print("=" * 60)
        print(f"模型: {self.config.model_name}")
        print(f"上下文大小: {self.config.context_size}")
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
            task = LCAProjectLevelCodeCompletionTask(self.config.to_dict())
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
                print(f"\n📝 处理样本 {i+1}/{len(test_docs)}: {doc.get('repo', f'sample_{i}')} ({doc.get('file_path', '')[:30]}...)")
                
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
                        
                        # 计算单个样本的汇总指标
                        if 'sample_results' in result and result['sample_results']:
                            sample_exact_match = sum(r['exact_match'] for r in result['sample_results']) / len(result['sample_results'])
                            sample_edit_distance = sum(r['edit_distance'] for r in result['sample_results']) / len(result['sample_results'])
                            print(f"   ✅ Exact Match: {sample_exact_match:.4f}, Edit Distance: {sample_edit_distance:.4f}")
                        else:
                            print(f"   ✅ 处理完成，但无有效结果")
                    
                except Exception as e:
                    print(f"   ❌ 处理失败: {e}")
                    failed_result = {
                        'idx': doc.get('idx', f'sample_{i}'),
                        'error': str(e),
                        'exact_match': 0.0,
                        'edit_distance': 0.0,
                        'bleu_score': 0.0
                    }
                    results.append(failed_result)
            
            # 模拟lm_eval结果格式
            task_name = "lca_project_level_code_completion"
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
        # 展平所有样本结果
        all_sample_results = []
        for sample in results:
            if 'sample_results' in sample:
                for sample_result in sample['sample_results']:
                    all_sample_results.append(sample_result)
        
        if not all_sample_results:
            return {}
        
        total_exact_match = sum(r.get('exact_match', 0) for r in all_sample_results)
        total_edit_distance = sum(r.get('edit_distance', 0) for r in all_sample_results)
        total_bleu_score = sum(r.get('bleu_score', 0) for r in all_sample_results)
        
        return {
            'exact_match': total_exact_match / len(all_sample_results),
            'edit_distance': total_edit_distance / len(all_sample_results),
            'bleu_score': total_bleu_score / len(all_sample_results),
            'valid_samples': len(all_sample_results),
            'total_samples': len(results)
        }
    
    def save_results(self, results: Dict[str, Any], duration: float):
        """保存评估结果"""
        # 保存主要结果
        results_file = self.output_dir / "evaluation_results.json"
        
        # 处理numpy类型，确保JSON可序列化
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # 转换结果
        serializable_results = convert_numpy(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
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
            'context_size': self.config.context_size,
            'split': self.config.get('split'),
            'timestamp': datetime.now().isoformat()
        }
        
        # 提取主要指标
        for task_name, task_results in results.get('results', {}).items():
            summary[f'{task_name}_metrics'] = {
                'exact_match': task_results.get('exact_match', 0.0),
                'edit_distance': task_results.get('edit_distance', 0.0),
                'bleu_score': task_results.get('bleu_score', 0.0)
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
                ('Exact Match', 'exact_match'),
                ('Edit Distance', 'edit_distance'),
                ('BLEU Score', 'bleu_score')
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
        description='Project Level Code Completion 评估器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 使用预设模型配置:
   python run_project_level_code_completion_eval.py --model gpt-4o

2. 使用千帆模型:
   python run_project_level_code_completion_eval.py --model qianfan-qwen

3. 自定义配置:
   python run_project_level_code_completion_eval.py --model openai-chat-completions \\
       --model_args "model=gpt-4o,temperature=0.1" \\
       --context_size small_context \\
       --limit 10

4. 从配置文件运行:
   python run_project_level_code_completion_eval.py --config config.json

5. 查看可用模型:
   python run_project_level_code_completion_eval.py --list_models
        """
    )
    
    # 模型配置
    parser.add_argument('--model', type=str, 
                       help='模型名称或预设配置键')
    parser.add_argument('--model_args', type=str,
                       help='自定义模型参数字符串 (格式: key1=value1,key2=value2)')
    
    # 任务配置
    parser.add_argument('--context_size', type=str, 
                       choices=['small_context', 'medium_context', 'large_context', 'huge_context'],
                       default='small_context', help='上下文大小 (默认: small_context)')
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
            config = ProjectLevelCodeCompletionConfig.from_file(args.config)
        elif args.model in MODEL_CONFIGS:
            print(f"🎯 使用预设配置: {args.model}")
            config = get_model_config(args.model)
        else:
            print(f"🔧 使用自定义配置")
            config = ProjectLevelCodeCompletionConfig()
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
        if args.context_size:
            overrides['context_size'] = args.context_size
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
        evaluator_instance = ProjectLevelCodeCompletionEvaluator(config)
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
