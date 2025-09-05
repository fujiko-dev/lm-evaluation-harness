#!/bin/bash

# Long Code Arena 进度监控脚本
# 每30秒自动刷新一次

echo "🚀 启动Long Code Arena测试进度监控"
echo "按 Ctrl+C 退出监控"
echo ""

while true; do
    clear
    echo "⏰ $(date '+%H:%M:%S') - Long Code Arena 测试进度"
    echo "=================================================="
    
    # 检查进程是否还在运行
    if ps -p 31076 > /dev/null 2>&1; then
        echo "✅ 测试进程运行中 (PID: 31076)"
        
        # 获取进程运行时间
        ETIME=$(ps -o etime= -p 31076 | tr -d ' ')
        echo "⏱️  运行时间: $ETIME"
        
        # 运行Python进度估算
        python3 -c "
import subprocess
from datetime import datetime, timedelta

# 计算已运行时间（分钟）
result = subprocess.run(['ps', '-o', 'etime=', '-p', '31076'], capture_output=True, text=True)
etime_str = result.stdout.strip()

# 解析运行时间
if ':' in etime_str:
    if etime_str.count(':') == 1:  # MM:SS
        minutes, seconds = map(int, etime_str.split(':'))
        total_minutes = minutes + seconds/60
    else:  # HH:MM:SS
        hours, minutes, seconds = map(int, etime_str.split(':'))
        total_minutes = hours*60 + minutes + seconds/60
else:  # 只有秒
    total_minutes = int(etime_str) / 60

# 估算进度
samples_per_minute = 60 / 52  # 每分钟完成的样本数
estimated_completed = int(total_minutes * samples_per_minute)
estimated_completed = min(estimated_completed, 150)

progress_pct = (estimated_completed / 150) * 100
remaining = 150 - estimated_completed

print(f'📊 估算进度: {estimated_completed}/150 ({progress_pct:.1f}%)')

# 创建进度条
bar_length = 30
filled_length = int(bar_length * estimated_completed // 150)
bar = '█' * filled_length + '░' * (bar_length - filled_length)
print(f'   [{bar}]')

if remaining > 0:
    remaining_minutes = remaining * 52 / 60
    if remaining_minutes > 60:
        remaining_hours = remaining_minutes / 60
        print(f'⏳ 预计剩余: {remaining_hours:.1f} 小时')
    else:
        print(f'⏳ 预计剩余: {remaining_minutes:.0f} 分钟')
    
    finish_time = datetime.now() + timedelta(minutes=remaining_minutes)
    print(f'🎯 预计完成: {finish_time.strftime(\"%H:%M\")}')
"
        
        echo ""
        echo "💡 提示:"
        echo "   - 代码生成需要时间，请耐心等待"
        echo "   - 结果文件将在完成后出现在 results/ 目录"
        echo "   - 当前是第1批完整的150样本测试"
        
    else
        echo "🎉 测试已完成或进程已结束!"
        echo ""
        echo "🔍 检查结果文件..."
        ls -la results/long_code_arena/library_based_code_generation/kimi_full_150_* 2>/dev/null || echo "   结果文件尚未生成"
        break
    fi
    
    echo ""
    echo "🔄 30秒后自动刷新... (Ctrl+C 退出)"
    sleep 30
done

echo ""
echo "👋 监控结束"
