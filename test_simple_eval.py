#!/usr/bin/env python3
"""
简单的测试脚本，验证结果保存功能
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from lm_eval.tasks.long_code_arena.library_based_code_generation.realtime_model import RealtimePrintModel
from lm_eval import evaluator
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import handle_non_serializable
import json
import datetime

def main():
    # 设置API密钥
    api_key = "bce-v3/ALTAK-O0JXIpHCVwyiAn40hEsVe/3cc2daea23929dead2a3e65334415db7f184c57d"
    os.environ['OPENAI_API_KEY'] = api_key
    
    print("🚀 启动简单测试")
    print("=" * 50)
    
    # 创建实时模型 - 启用并发加速
    model_args = {
        'model': 'kimi-k2-instruct',
        'base_url': 'https://qianfan.baidubce.com/v2/chat/completions',
        'num_concurrent': 3,  # 并发请求数量
        'max_retries': 3,
        'tokenized_requests': False
    }
    
    print("🔧 初始化模型...")
    model = RealtimePrintModel(**model_args)
    
    # 设置输出路径
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"./results/long_code_arena/library_based_code_generation/test_save_{timestamp}"
    
    print(f"📁 结果将保存到: {output_path}")
    
    # 创建EvaluationTracker
    evaluation_tracker = EvaluationTracker(output_path=output_path)
    
    try:
        print("📋 开始测试评估...")
        
        # 运行评估（使用较小的数据集进行测试）
        results = evaluator.simple_evaluate(
            model=model,
            model_args=model_args,  # 添加model_args参数
            tasks=['lca_library_based_code_generation_custom'],  # 使用较小的数据集
            batch_size=3,  # 增加批处理大小
            limit=2,  # 只测试2个样本
            log_samples=True,
            verbosity="INFO",
            apply_chat_template=True,
            confirm_run_unsafe_code=True,
            evaluation_tracker=evaluation_tracker
        )
        
        print("\n🎉 评估完成!")
        print("📊 结果汇总:")
        
        if 'results' in results:
            for task, metrics in results['results'].items():
                print(f"任务: {task}")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        
        # 手动保存结果
        print(f"\n💾 保存结果到文件...")
        os.makedirs(output_path, exist_ok=True)
        
        # 保存主要结果文件
        results_file = os.path.join(output_path, "results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=handle_non_serializable, ensure_ascii=False)
        print(f"✅ 结果已保存到: {results_file}")
        
        # 如果有samples，也保存samples文件
        if 'samples' in results:
            samples_file = os.path.join(output_path, "samples.jsonl")
            with open(samples_file, 'w', encoding='utf-8') as f:
                for sample in results['samples']:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"✅ 样本数据已保存到: {samples_file}")
        
        # 检查文件是否被保存
        print(f"\n📁 检查保存的文件...")
        if os.path.exists(output_path):
            files = os.listdir(output_path)
            print(f"✅ 目录存在，包含文件: {files}")
            
            # 检查文件大小
            for file in files:
                file_path = os.path.join(output_path, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   - {file}: {size:,} bytes")
        else:
            print("❌ 输出目录不存在")
        
        print(f"📁 完整路径: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
