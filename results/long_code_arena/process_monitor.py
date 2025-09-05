import subprocess
import re
from datetime import datetime, timedelta

print('🔍 Long Code Arena 测试进度查看')
print('=' * 40)
print(f'查看时间: {datetime.now().strftime(\"%H:%M:%S\")}')
print()

# 从ps输出中提取进度信息
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'lm_eval' in line and 'kimi-k2-instruct' in line:
        print('✅ 测试进程运行中')
        
        # 尝试从你提供的输出中获取进度
        # 根据终端选择内容，当前是 6/150 (4%)
        current = 6
        total = 150
        progress = (current / total) * 100
        
        print(f'📊 当前进度: {current}/{total} ({progress:.1f}%)')
        print(f'⏱️  平均速度: ~52秒/样本')
        
        # 计算剩余时间
        remaining = total - current
        time_per_sample = 52  # 秒
        remaining_seconds = remaining * time_per_sample
        remaining_minutes = remaining_seconds / 60
        remaining_hours = remaining_minutes / 60
        
        if remaining_hours > 1:
            print(f'⏳ 预计剩余: {remaining_hours:.1f}小时')
        else:
            print(f'⏳ 预计剩余: {remaining_minutes:.0f}分钟')
        
        # 估算完成时间
        now = datetime.now()
        finish_time = now + timedelta(seconds=remaining_seconds)
        print(f'🎯 预计完成: {finish_time.strftime(\"%H:%M\")}')
        
        print()
        print('💡 提示:')
        print('- 可以让测试在后台继续运行')
        print('- 代码生成任务需要较长时间是正常的')
        print('- 每个样本需要生成复杂的Python代码')
        break
else:
    print('❌ 没有检测到运行中的测试进程')