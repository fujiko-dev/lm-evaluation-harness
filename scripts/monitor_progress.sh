#!/bin/bash

# Long Code Arena 测试进度监控脚本
# 使用方法: ./scripts/monitor_progress.sh [results_directory]

RESULTS_DIR=${1:-"./results/long_code_arena/library_based_code_generation"}

echo "🔍 监控Long Code Arena测试进度"
echo "================================"
echo "结果目录: $RESULTS_DIR"
echo ""

while true; do
    clear
    echo "📊 Long Code Arena - 测试进度监控"
    echo "================================"
    echo "时间: $(date)"
    echo ""
    
    # 查找最新的结果目录
    LATEST_DIR=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "*$(date +%Y-%m-%d)*" | sort | tail -1)
    
    if [ -n "$LATEST_DIR" ]; then
        echo "📁 最新测试目录: $(basename "$LATEST_DIR")"
        
        # 检查是否有结果文件
        RESULT_FILE=$(find "$LATEST_DIR" -name "results_*.json" | head -1)
        SAMPLE_FILE=$(find "$LATEST_DIR" -name "samples_*.jsonl" | head -1)
        
        if [ -f "$SAMPLE_FILE" ]; then
            COMPLETED_COUNT=$(wc -l < "$SAMPLE_FILE" 2>/dev/null || echo "0")
            echo "✅ 已完成样本数: $COMPLETED_COUNT"
            
            if [ "$COMPLETED_COUNT" -gt 0 ]; then
                echo "📈 最近处理的样本:"
                tail -1 "$SAMPLE_FILE" 2>/dev/null | python3 -c "
import json
import sys
try:
    line = sys.stdin.read().strip()
    if line:
        data = json.loads(line)
        doc = data.get('doc', {})
        print(f'  指令: {doc.get(\"instruction\", \"未知\")[:50]}...')
        if 'pass_at_1' in data:
            print(f'  Pass@1: {data[\"pass_at_1\"]}')
except:
    print('  解析样本数据时出错')
"
            fi
            
            # 估算剩余时间
            if [ "$COMPLETED_COUNT" -gt 0 ]; then
                TOTAL_SAMPLES=150
                REMAINING=$((TOTAL_SAMPLES - COMPLETED_COUNT))
                PROGRESS=$((COMPLETED_COUNT * 100 / TOTAL_SAMPLES))
                echo "📊 进度: $PROGRESS% ($COMPLETED_COUNT/$TOTAL_SAMPLES)"
                echo "⏳ 剩余样本: $REMAINING"
            fi
        else
            echo "⏳ 等待测试开始..."
        fi
        
        if [ -f "$RESULT_FILE" ]; then
            echo ""
            echo "🎯 最终结果:"
            python3 -c "
import json
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)
    
    if 'results' in data:
        for task, metrics in data['results'].items():
            print(f'任务: {task}')
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f'  {metric}: {value:.4f}')
                else:
                    print(f'  {metric}: {value}')
    else:
        print('  结果文件格式不正确')
except Exception as e:
    print(f'  读取结果文件时出错: {e}')
"
            echo ""
            echo "✅ 测试已完成！"
            break
        fi
    else
        echo "⏳ 等待测试开始..."
    fi
    
    echo ""
    echo "🔄 5秒后刷新 (Ctrl+C 退出)..."
    sleep 5
done

echo ""
echo "🎉 监控结束！"
