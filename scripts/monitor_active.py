#!/usr/bin/env python3

import os
import time
import json
import subprocess
import glob
from datetime import datetime

def get_lm_eval_processes():
    """获取当前运行的lm_eval进程"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = []
        for line in result.stdout.split('\n'):
            if 'lm_eval' in line and 'grep' not in line:
                processes.append(line)
        return processes
    except:
        return []

def find_latest_output_dir():
    """查找最新的输出目录"""
    pattern = "results/long_code_arena/library_based_code_generation/kimi_full_150_*"
    dirs = glob.glob(pattern)
    if dirs:
        return max(dirs, key=os.path.getctime)
    return None

def count_completed_samples(output_dir):
    """统计已完成的样本数"""
    if not output_dir:
        return 0
    
    pattern = os.path.join(output_dir, "*", "samples_*.jsonl")
    sample_files = glob.glob(pattern)
    
    for sample_file in sample_files:
        try:
            with open(sample_file, 'r') as f:
                lines = f.readlines()
                return len(lines)
        except:
            continue
    return 0

def get_latest_sample_info(output_dir):
    """获取最新样本信息"""
    if not output_dir:
        return None
    
    pattern = os.path.join(output_dir, "*", "samples_*.jsonl")
    sample_files = glob.glob(pattern)
    
    for sample_file in sample_files:
        try:
            with open(sample_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        data = json.loads(last_line)
                        doc = data.get('doc', {})
                        instruction = doc.get('instruction', '未知')[:50] + '...'
                        pass_at_1 = data.get('pass_at_1', '未知')
                        return {
                            'instruction': instruction,
                            'pass_at_1': pass_at_1,
                            'total_samples': len(lines)
                        }
        except:
            continue
    return None

def main():
    print("🔍 Long Code Arena - 实时进度监控")
    print("=" * 50)
    
    total_samples = 150
    start_time = time.time()
    
    while True:
        os.system('clear')
        print("📊 Long Code Arena - 实时进度监控")
        print("=" * 50)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 检查进程状态
        processes = get_lm_eval_processes()
        if processes:
            print("🟢 测试进程运行中:")
            for proc in processes:
                # 提取进程信息
                parts = proc.split()
                if len(parts) > 1:
                    pid = parts[1]
                    print(f"  PID: {pid}")
            print()
        else:
            print("🔴 没有检测到运行中的lm_eval进程")
            print()
        
        # 查找最新输出目录
        output_dir = find_latest_output_dir()
        if output_dir:
            completed = count_completed_samples(output_dir)
            latest_info = get_latest_sample_info(output_dir)
            
            print(f"📁 输出目录: {os.path.basename(output_dir)}")
            print(f"✅ 已完成样本: {completed}/{total_samples}")
            
            if completed > 0:
                progress = (completed / total_samples) * 100
                elapsed = time.time() - start_time
                if completed > 1:  # 避免除零错误
                    estimated_total = elapsed * total_samples / completed
                    remaining = estimated_total - elapsed
                    remaining_mins = remaining / 60
                    print(f"📊 进度: {progress:.1f}%")
                    print(f"⏱️  预计剩余时间: {remaining_mins:.1f} 分钟")
                
                if latest_info:
                    print(f"📝 最新样本: {latest_info['instruction']}")
                    print(f"🎯 Pass@1: {latest_info['pass_at_1']}")
            
            # 检查是否完成
            if completed >= total_samples:
                print()
                print("🎉 测试已完成！")
                
                # 显示最终结果
                result_pattern = os.path.join(output_dir, "*", "results_*.json")
                result_files = glob.glob(result_pattern)
                if result_files:
                    try:
                        with open(result_files[0], 'r') as f:
                            results = json.load(f)
                        
                        print("📈 最终结果:")
                        if 'results' in results:
                            for task, metrics in results['results'].items():
                                print(f"  任务: {task}")
                                for metric, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        print(f"    {metric}: {value:.4f}")
                    except:
                        print("  无法读取结果文件")
                
                break
        else:
            print("⏳ 等待测试开始...")
        
        print()
        print("🔄 每10秒刷新一次 (Ctrl+C 退出)")
        
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n👋 监控已停止")
            break

if __name__ == "__main__":
    main()
