#!/bin/bash

# Long Code Arena 实时进度测试脚本
# 使用方法: ./scripts/run_lca_realtime.sh [model_name] [base_url] [samples_count]

set -e

MODEL_NAME=${1:-"kimi-k2-instruct"}
BASE_URL=${2:-"https://qianfan.baidubce.com/v2/chat/completions"}
SAMPLES=${3:-5}
DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%H-%M-%S)

# 结果目录
RESULTS_DIR="./results/long_code_arena/library_based_code_generation"
OUTPUT_DIR="${RESULTS_DIR}/${MODEL_NAME}_realtime_${DATE}_${TIMESTAMP}"

# 确保目录存在
mkdir -p "$RESULTS_DIR"

echo "🚀 启动实时进度Long Code Arena测试"
echo "=========================================="
echo "模型: $MODEL_NAME"
echo "API: $BASE_URL"  
echo "样本数: $SAMPLES"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 启动后台监控进程
echo "📡 启动实时监控..."
python3 lm_eval/tasks/long_code_arena/library_based_code_generation/realtime_monitor.py &
MONITOR_PID=$!

# 等待1秒让监控启动
sleep 1

echo "🔄 开始评估测试..."

# 执行lm_eval，并捕获输出
python3 -m lm_eval \
    --model local-chat-completions \
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL},num_concurrent=1,max_retries=10,tokenized_requests=False" \
    --apply_chat_template \
    --tasks lca_library_based_code_generation_verbose \
    --batch_size 1 \
    --limit "$SAMPLES" \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --confirm_run_unsafe_code \
    2>&1 | while IFS= read -r line; do
        echo "$line"
        
        # 检测特定的进度模式并添加时间戳
        if [[ "$line" =~ "Requesting API:" ]]; then
            echo "⏰ $(date '+%H:%M:%S') - API请求进行中..."
        elif [[ "$line" =~ "🔍 开始评估" ]]; then
            echo "⏰ $(date '+%H:%M:%S') - 开始评估阶段"
        elif [[ "$line" =~ "samples_.*\.jsonl" ]]; then
            echo "⏰ $(date '+%H:%M:%S') - 结果文件已生成"
        fi
    done

# 停止监控进程
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "🎉 测试完成!"

# 显示结果汇总
if [ -d "$OUTPUT_DIR" ]; then
    echo "📁 结果目录: $OUTPUT_DIR"
    
    # 查找结果文件
    RESULT_FILE=$(find "$OUTPUT_DIR" -name "results_*.json" | head -1)
    if [ -f "$RESULT_FILE" ]; then
        echo "📊 测试结果:"
        python3 -c "
import json
with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)

if 'results' in data:
    for task, metrics in data['results'].items():
        print(f'任务: {task}')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'  {metric}: {value:.4f}')
"
    fi
    
    # 显示样本文件信息
    SAMPLE_FILE=$(find "$OUTPUT_DIR" -name "samples_*.jsonl" | head -1)
    if [ -f "$SAMPLE_FILE" ]; then
        SAMPLE_COUNT=$(wc -l < "$SAMPLE_FILE")
        echo "📄 样本文件: $SAMPLE_COUNT 个样本已保存"
        echo "   查看详情: cat $SAMPLE_FILE | jq ."
    fi
else
    echo "❌ 没有找到结果目录"
fi

echo ""
echo "🎯 下一步:"
echo "1. 查看详细结果: ls -la $OUTPUT_DIR"
echo "2. 分析样本输出: 查看 samples_*.jsonl 文件"  
echo "3. 运行更大规模测试: 增加样本数量"
