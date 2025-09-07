#!/usr/bin/env python3
"""
DeepSeek V3模型全量评估 - 百度千帆API
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
    print("🚀 启动qwen3-235b-a22b-instruct-2507模型全量评估")
    print("=" * 60)
    
    # 设置API密钥环境变量
    os.environ['OPENAI_API_KEY'] = 'bce-v3/ALTAK-O0JXIpHCVwyiAn40hEsVe/3cc2daea23929dead2a3e65334415db7f184c57d'
    
    # DeepSeek V3模型配置 - 使用百度千帆API
    model_args = {
        'model': 'qwen3-235b-a22b-instruct-2507',
        'base_url': 'https://qianfan.baidubce.com/v2/chat/completions',
        'num_concurrent': 5,  # 适中的并发数
        'max_retries': 10,
        'tokenized_requests': False,
        'stop': '<|endoftext|>',
        'timeout': 120
    }
    
    print("🔧 初始化qwen3-235b-a22b-instruct-2507模型...")
    print(f"   - 模型: {model_args['model']}")
    print(f"   - API地址: {model_args['base_url']}")
    print(f"   - 并发请求数: {model_args['num_concurrent']}")
    print(f"   - 最大重试次数: {model_args['max_retries']}")
    print(f"   - 停止标记: {model_args['stop']}")
    print(f"   - 超时时间: {model_args['timeout']}秒")
    
    model = RealtimePrintModel(**model_args)
    
    # 设置输出路径
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"./results/long_code_arena/library_based_code_generation/deepseek_v3_full_{timestamp}"
    
    print(f"📁 结果将保存到: {output_path}")
    
    # 创建EvaluationTracker
    evaluation_tracker = EvaluationTracker(output_path=output_path)
    
    try:
        print("📋 开始DeepSeek V3模型全量评估...")
        print("   - 使用完整数据集")
        print("   - 预计需要较长时间")
        
        # 全量评估配置
        results = evaluator.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=['lca_library_based_code_generation_full'],  # 完整数据集
            batch_size=5,  # 适中的批处理大小
            log_samples=True,
            verbosity="INFO",
            apply_chat_template=True,
            confirm_run_unsafe_code=True,
            evaluation_tracker=evaluation_tracker
        )
        
        print("\n🎉 qwen3-235b-a22b-instruct-2507模型全量评估完成!")
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
        
        # 性能统计
        if 'config' in results:
            config = results['config']
            print(f"\n⚡ 性能统计:")
            print(f"   - 批处理大小: {config.get('batch_size', 'N/A')}")
            print(f"   - 设备: {config.get('device', 'N/A')}")
            print(f"   - 限制样本数: {config.get('limit', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 全量评估成功完成！")
    else:
        print("\n❌ 全量评估失败！请检查模型配置。")
