#!/usr/bin/env python3
"""
使用实时打印模型的测试脚本
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from lm_eval.tasks.long_code_arena.library_based_code_generation.realtime_model import RealtimePrintModel
from lm_eval import evaluator
from lm_eval.loggers import EvaluationTracker
import json

def main():
    # 设置API密钥
    api_key = "bce-v3/ALTAK-O0JXIpHCVwyiAn40hEsVe/3cc2daea23929dead2a3e65334415db7f184c57d"
    os.environ['OPENAI_API_KEY'] = api_key
    
    print("🚀 启动实时Long Code Arena测试")
    print("=" * 50)
    
    # 创建实时模型 - 启用并发加速
    model_args = {
        'model': 'kimi-k2-instruct',
        'base_url': 'https://qianfan.baidubce.com/v2/chat/completions',
        'num_concurrent': 5,  # 并发请求数量，可以调整到10-20
        'max_retries': 10,
        'tokenized_requests': False
    }
    
    print("🔧 初始化实时打印模型...")
    model = RealtimePrintModel(**model_args)
    
    # 运行评估
    print("📋 开始评估任务...")
    
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = f"./results/long_code_arena/library_based_code_generation/realtime_full_{timestamp}"
        
        print(f"📁 结果将保存到: {output_path}")
        
        # 创建EvaluationTracker来自动保存结果
        evaluation_tracker = EvaluationTracker(output_path=output_path)
        
        results = evaluator.simple_evaluate(
            model=model,
            model_args=model_args,  # 添加model_args参数
            tasks=['lca_library_based_code_generation_full'],  # 使用完整数据集
            batch_size=5,  # 增加批处理大小，可以调整到10-20
            log_samples=True,
            verbosity="INFO",
            apply_chat_template=True,
            confirm_run_unsafe_code=True,
            evaluation_tracker=evaluation_tracker  # 添加evaluation_tracker
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
        
        # EvaluationTracker会自动保存结果，这里只需要手动保存samples（如果需要的话）
        if 'samples' in results:
            print(f"\n💾 保存样本数据...")
            os.makedirs(output_path, exist_ok=True)
            samples_file = os.path.join(output_path, "samples.jsonl")
            with open(samples_file, 'w', encoding='utf-8') as f:
                for sample in results['samples']:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"✅ 样本数据已保存到: {samples_file}")
        
        print(f"📁 结果文件已自动保存到目录: {output_path}")
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
