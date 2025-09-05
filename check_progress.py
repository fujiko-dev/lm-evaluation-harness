#!/usr/bin/env python3
"""
Long Code Arena 测试进度检查器
使用方法: python3 check_progress.py
"""

import subprocess
import time
from datetime import datetime, timedelta

def check_lm_eval_process():
    """检查lm_eval进程状态"""
    try:
        result = subprocess.run(['ps', '-o', 'pid,etime,pcpu,command', '-p', '31076'], 
                              capture_output=True, text=True)
        if '31076' in result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split(None, 3)
                etime = parts[1]
                cpu = parts[2]
                return True, etime, cpu
    except:
        pass
    return False, None, None

def estimate_progress():
    """基于运行时间估算进度"""
    # 测试开始时间从输出路径推断: 13:35:55
    start_time = datetime.strptime("13:35:55", "%H:%M:%S")
    start_time = start_time.replace(year=2025, month=9, day=5)
    
    current_time = datetime.now()
    
    # 计算已运行时间
    elapsed = current_time - start_time
    elapsed_seconds = elapsed.total_seconds()
    elapsed_minutes = elapsed_seconds / 60
    
    # 基于观察到的速度: 52秒/样本
    time_per_sample = 52
    estimated_completed = max(0, int(elapsed_seconds / time_per_sample))
    estimated_completed = min(estimated_completed, 150)  # 不超过总数
    
    progress_pct = (estimated_completed / 150) * 100
    
    # 计算剩余时间
    remaining_samples = 150 - estimated_completed
    remaining_seconds = remaining_samples * time_per_sample
    finish_time = current_time + timedelta(seconds=remaining_seconds)
    
    return {
        'elapsed_minutes': elapsed_minutes,
        'estimated_completed': estimated_completed,
        'progress_pct': progress_pct,
        'remaining_samples': remaining_samples,
        'remaining_seconds': remaining_seconds,
        'finish_time': finish_time
    }

def main():
    print("🔍 Long Code Arena 150样本测试 - 进度检查器")
    print("=" * 50)
    print(f"检查时间: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # 检查进程状态
    running, etime, cpu = check_lm_eval_process()
    
    if not running:
        print("❌ 未检测到lm_eval进程 (PID: 31076)")
        print("   测试可能已完成或被中断")
        return
    
    print(f"✅ 进程运行中")
    print(f"   运行时间: {etime}")
    print(f"   CPU使用率: {cpu}%")
    
    if float(cpu) > 0.1:
        print("   状态: 🔄 活跃处理中")
    else:
        print("   状态: ⏳ 等待API响应")
    
    print()
    
    # 估算进度
    progress = estimate_progress()
    
    print("📊 进度估算:")
    print(f"   已运行: {progress['elapsed_minutes']:.1f} 分钟")
    print(f"   预估完成: {progress['estimated_completed']}/150 样本")
    print(f"   进度: {progress['progress_pct']:.1f}%")
    print(f"   剩余样本: {progress['remaining_samples']}")
    
    if progress['remaining_seconds'] > 3600:
        hours = progress['remaining_seconds'] / 3600
        print(f"   预计剩余: {hours:.1f} 小时")
    else:
        minutes = progress['remaining_seconds'] / 60
        print(f"   预计剩余: {minutes:.0f} 分钟")
    
    print(f"   预计完成: {progress['finish_time'].strftime('%H:%M')}")
    
    print()
    print("💡 说明:")
    print("   - 这是基于时间和速度的估算")
    print("   - 实际进度可能因API响应时间波动")
    print("   - 每个样本约需52秒是正常的代码生成速度")
    print("   - 输出文件将在测试完全结束后创建")
    
    print()
    print("🔄 要持续监控，请定期运行: python3 check_progress.py")

if __name__ == "__main__":
    main()
