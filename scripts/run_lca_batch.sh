#!/bin/bash

# Long Code Arena - Library-Based Code Generation 批量测试脚本
# 使用方法: ./scripts/run_lca_batch.sh [model_name] [base_url] [config_type]

set -e

# 参数设置
MODEL_NAME=${1:-"test_model"}
BASE_URL=${2:-"http://localhost:8000/v1/chat/completions"}
CONFIG_TYPE=${3:-"custom"}  # custom, fallback, enhanced, full
DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%H-%M-%S)

# 结果目录
RESULTS_DIR="./results/long_code_arena/library_based_code_generation"
OUTPUT_DIR="${RESULTS_DIR}/${MODEL_NAME}_${CONFIG_TYPE}_${DATE}_${TIMESTAMP}"

# 确保目录存在
mkdir -p "$RESULTS_DIR"

echo "🚀 开始Long Code Arena评估测试"
echo "================================"
echo "模型: $MODEL_NAME"
echo "API: $BASE_URL"
echo "配置: $CONFIG_TYPE"
echo "输出目录: $OUTPUT_DIR"
echo "================================"

# 根据配置类型选择任务
case $CONFIG_TYPE in
    "fallback")
        TASK="lca_library_based_code_generation_fallback"
        LIMIT="--limit 5"
        echo "📝 使用fallback配置 (5个样本，快速测试)"
        ;;
    "custom")
        TASK="lca_library_based_code_generation_custom"
        LIMIT="--limit 10"
        echo "⚙️ 使用优化配置 (10个样本，推荐测试)"
        ;;
    "enhanced")
        TASK="lca_library_based_code_generation_enhanced"
        LIMIT="--limit 20"
        echo "📊 使用增强配置 (20个样本，多指标评估)"
        ;;
    "full")
        TASK="lca_library_based_code_generation_full"
        LIMIT=""
        echo "🔬 使用完整配置 (150个样本，完整评估)"
        ;;
    *)
        echo "❌ 未知配置类型: $CONFIG_TYPE"
        echo "支持的类型: fallback, custom, enhanced, full"
        exit 1
        ;;
esac

# 执行评估
echo "🔄 开始评估..."
python3 -m lm_eval \
    --model local-chat-completions \
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL},num_concurrent=1,max_retries=10,tokenized_requests=False" \
    --apply_chat_template \
    --tasks "$TASK" \
    --batch_size 1 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    $LIMIT \
    --confirm_run_unsafe_code

# 检查结果
if [ $? -eq 0 ]; then
    echo "✅ 评估完成！"
    echo "📁 结果保存在: $OUTPUT_DIR"
    
    # 显示结果汇总
    if [ -f "$OUTPUT_DIR"/*/results_*.json ]; then
        echo "📊 结果汇总:"
        python3 -c "
import json
import glob
import os

result_files = glob.glob('$OUTPUT_DIR/*/results_*.json')
if result_files:
    with open(result_files[0], 'r') as f:
        data = json.load(f)
    
    print('模型: $MODEL_NAME')
    print('配置: $CONFIG_TYPE')
    print('任务: $TASK')
    
    if 'results' in data:
        for task, metrics in data['results'].items():
            print(f'任务: {task}')
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f'  {metric}: {value:.4f}')
                else:
                    print(f'  {metric}: {value}')
"
    fi
    
    echo ""
    echo "🎯 下一步建议:"
    echo "1. 查看详细结果: ls -la $OUTPUT_DIR"
    echo "2. 分析样本输出: 查看 samples_*.jsonl 文件"
    echo "3. 对比其他模型: 运行不同模型的测试"
    
else
    echo "❌ 评估失败！"
    exit 1
fi

echo "🎉 测试完成！"
