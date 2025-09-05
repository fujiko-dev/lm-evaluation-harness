#!/usr/bin/env python3
"""
使用实时打印模型的测试脚本
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from lm_eval.tasks.long_code_arena.library_based_code_generation.realtime_model import RealtimePrintModel
from lm_eval import evaluator
import json

def main():
    # 设置API密钥
    api_key = "bce-v3/ALTAK-O0JXIpHCVwyiAn40hEsVe/3cc2daea23929dead2a3e65334415db7f184c57d"
    os.environ['OPENAI_API_KEY'] = api_key
    
    print("🚀 启动实时Long Code Arena测试")
    print("=" * 50)
    
    # 创建实时模型
    model_args = {
        'model': 'kimi-k2-instruct',
        'base_url': 'https://qianfan.baidubce.com/v2/chat/completions',
        'num_concurrent': 1,
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
        
        results = evaluator.simple_evaluate(
            model=model,
            tasks=['lca_library_based_code_generation_full'],  # 使用完整数据集
            batch_size=1,
            log_samples=True,
            verbosity="INFO",
            apply_chat_template=True,
            confirm_run_unsafe_code=True
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
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
