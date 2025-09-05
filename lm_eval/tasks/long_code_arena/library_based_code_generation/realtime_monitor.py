#!/usr/bin/env python3
"""
实时监控脚本 - 在另一个终端运行这个脚本来监控进度
"""

import time
import subprocess
import os
import json
from datetime import datetime

def monitor_lm_eval_progress():
    """监控lm_eval进程的实时进度"""
    print("🔍 Long Code Arena 实时进度监控")
    print("=" * 50)
    print("在另一个终端运行lm_eval，此脚本将监控进度")
    print("按 Ctrl+C 退出监控")
    print()
    
    last_api_count = 0
    start_time = None
    
    while True:
        try:
            # 检查是否有lm_eval进程
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            lm_eval_found = False
            
            for line in result.stdout.split('\n'):
                if 'lm_eval' in line and 'kimi-k2-instruct' in line:
                    lm_eval_found = True
                    break
            
            current_time = datetime.now()
            
            if lm_eval_found:
                if start_time is None:
                    start_time = current_time
                    print(f"✅ 检测到lm_eval进程开始运行 ({current_time.strftime('%H:%M:%S')})")
                
                # 尝试检测网络活动来估算进度
                try:
                    # 检查网络连接数
                    netstat_result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
                    https_connections = 0
                    for line in netstat_result.stdout.split('\n'):
                        if ':443' in line and 'ESTABLISHED' in line:
                            https_connections += 1
                    
                    elapsed_time = (current_time - start_time).total_seconds()
                    elapsed_minutes = elapsed_time / 60
                    
                    print(f"\\r🔄 运行中... {elapsed_minutes:.1f}分钟 | HTTPS连接: {https_connections}", end='', flush=True)
                    
                except Exception:
                    elapsed_time = (current_time - start_time).total_seconds()
                    elapsed_minutes = elapsed_time / 60
                    print(f"\\r🔄 运行中... {elapsed_minutes:.1f}分钟", end='', flush=True)
            
            else:
                if start_time is not None:
                    print(f"\\n🎉 lm_eval进程已结束")
                    
                    # 检查是否有新的结果文件
                    print("\\n🔍 检查结果文件...")
                    result_dirs = []
                    base_path = "./results/long_code_arena/library_based_code_generation"
                    
                    if os.path.exists(base_path):
                        for item in os.listdir(base_path):
                            item_path = os.path.join(base_path, item)
                            if os.path.isdir(item_path):
                                # 检查修改时间是否在最近10分钟内
                                mtime = os.path.getmtime(item_path)
                                if (time.time() - mtime) < 600:  # 10分钟内
                                    result_dirs.append(item)
                    
                    if result_dirs:
                        print(f"📁 发现新结果目录: {result_dirs}")
                        for dir_name in result_dirs:
                            dir_path = os.path.join(base_path, dir_name)
                            print(f"   查看结果: ls -la {dir_path}")
                    else:
                        print("❌ 没有发现新的结果文件")
                    
                    break
                else:
                    print(f"\\r⏳ 等待lm_eval进程启动... ({current_time.strftime('%H:%M:%S')})", end='', flush=True)
            
            time.sleep(2)  # 每2秒检查一次
            
        except KeyboardInterrupt:
            print("\\n👋 监控已停止")
            break
        except Exception as e:
            print(f"\\n❌ 监控出错: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_lm_eval_progress()
