#!/usr/bin/env python3
"""
高性能并发评估脚本 - 大幅提升评估速度
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from lm_eval.tasks.long_code_arena.library_based_code_generation.realtime_model import RealtimePrintModel
from lm_eval import evaluator
from lm_eval.loggers import EvaluationTracker
import json
import datetime

def main():
    # 设置API密钥
    api_key = "bce-v3/ALTAK-O0JXIpHCVwyiAn40hEsVe/3cc2daea23929dead2a3e65334415db7f184c57d"
    os.environ['OPENAI_API_KEY'] = api_key
    
    print("🚀 启动高性能并发评估")
    print("=" * 60)
    
    # 高性能模型配置 - 降低并发避免超时
    model_args = {
        'model': 'kimi-k2-instruct',
        'base_url': 'https://qianfan.baidubce.com/v2/chat/completions',
        'num_concurrent': 5,  # 降低并发数避免超时
        'max_retries': 3,
        'tokenized_requests': False,
        'timeout': 120  # 增加超时时间
    }
    
    print("🔧 初始化高性能模型...")
    print(f"   - 并发请求数: {model_args['num_concurrent']}")
    print(f"   - 最大重试次数: {model_args['max_retries']}")
    print(f"   - 超时时间: {model_args['timeout']}秒")
    
    model = RealtimePrintModel(**model_args)
    
    # 设置输出路径
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"./results/long_code_arena/library_based_code_generation/fast_eval_{timestamp}"
    
    print(f"📁 结果将保存到: {output_path}")
    
    # 创建EvaluationTracker
    evaluation_tracker = EvaluationTracker(output_path=output_path)
    
    try:
        print("📋 开始高性能评估...")
        print("   - 使用完整数据集")
        print("   - 高并发批处理")
        print("   - 自动结果保存")
        
        # 高性能评估配置 - 降低批处理大小
        results = evaluator.simple_evaluate(
            model=model,
            model_args=model_args,  # 添加model_args参数
            tasks=['lca_library_based_code_generation_full'],  # 完整数据集
            batch_size=5,  # 降低批处理大小避免超时
            log_samples=True,
            verbosity="INFO",
            apply_chat_template=True,
            confirm_run_unsafe_code=True,
            evaluation_tracker=evaluation_tracker
        )
        
        print("\n🎉 高性能评估完成!")
        print("📊 结果汇总:")
        
        if 'results' in results:
            for task, metrics in results['results'].items():
                print(f"任务: {task}")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        
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
        
        # 性能统计
        if 'config' in results:
            config = results['config']
            print(f"\n⚡ 性能统计:")
            print(f"   - 批处理大小: {config.get('batch_size', 'N/A')}")
            print(f"   - 设备: {config.get('device', 'N/A')}")
            print(f"   - 限制样本数: {config.get('limit', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
