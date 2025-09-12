# Bug Localization Task - Long Code Arena

Bug Localization（缺陷定位）任务是 Long Code Arena 基准测试中的一个重要任务，用于评估大语言模型在给定bug描述和代码仓库的情况下，识别需要修改以修复bug的文件的能力。

## 任务定义

给定一个包含bug描述的GitHub issue和一个处于bug可重现状态的代码仓库快照，模型需要识别仓库中需要修改以解决报告的bug的文件。

## 数据集信息

- **数据来源**: JetBrains-Research/lca-bug-localization
- **支持语言**: Python (.py), Java (.java), Kotlin (.kt)
- **数据拆分**: train (训练集), dev (开发集), test (测试集)

### 数据集统计

| 语言类别 | 数据点数量 |
|:--------:|:----------:|
| Python   | 4,339      |
| Java     | 2,522      | 
| Kotlin   | 618        |

## 文件结构

```
bug_localization/
├── __init__.py                          # 包初始化
├── config.py                            # 统一配置管理
├── lca_bug_localization_task.py        # 主任务实现
├── data_loader.py                       # 数据加载器
├── metrics.py                           # 评估指标
├── run_bug_localization_eval.py        # 统一评估脚本
├── test_case.py                         # 测试用例
├── lca_bug_localization*.yaml          # YAML配置文件
└── README.md                            # 本文档
```

## 评估指标

### 主要指标
- **Precision**: 预测正确的文件数 / 总预测文件数
- **Recall**: 预测正确的文件数 / 总目标文件数
- **F1 Score**: Precision和Recall的调和平均数
- **False Positive Rate**: 误判率

### 布尔指标
- **All Correct**: 是否所有文件都被正确识别
- **At Least One Correct**: 是否至少有一个文件被正确识别
- **All Incorrect**: 是否所有识别的文件都是错误的

## 快速开始

### 1. 环境配置

首先设置环境变量来配置数据集和输出路径：

```bash
# 设置数据集路径（可选，有默认值）
export LCA_DATASET_PATH="/path/to/your/lca_bug_localization_dataset"

# 设置输出路径（可选，有默认值）
export LCA_OUTPUT_PATH="/path/to/your/output_directory"

# 设置API密钥
export OPENAI_API_KEY="your-openai-api-key"  # 用于OpenAI模型或千帆模型
export ANTHROPIC_API_KEY="your-anthropic-key"  # 用于Claude模型
```

### 2. 运行测试

首先运行测试用例验证环境配置：

```bash
cd /path/to/lm-evaluation-harness
python -m lm_eval.tasks.long_code_arena.bug_localization.test_case
```

### 3. 快速评估示例

```bash
# 使用GPT-4o评估Python项目（默认配置）
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model gpt-4o \
    --language py \
    --limit 5

# 使用千帆模型评估
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model qianfan-qwen \
    --language py \
    --limit 5

# 查看所有可用的预设模型配置
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval --list_models
```

## 配置系统

新的配置系统支持灵活的参数配置，无需硬编码路径。

### 环境变量配置

可以通过环境变量设置默认路径：

```bash
# 数据集路径
export LCA_DATASET_PATH="/your/dataset/path"
export LCA_OUTPUT_PATH="/your/output/path"
```

### 预设模型配置

支持以下预设模型配置：

| 配置名称 | 模型 | 说明 |
|---------|------|------|
| `gpt-4o` | GPT-4o | OpenAI最新模型 |
| `gpt-4o-mini` | GPT-4o Mini | OpenAI小型模型 |
| `claude-3.5-sonnet` | Claude 3.5 Sonnet | Anthropic模型 |
| `qianfan-qwen` | 千帆 Qwen | 百度千帆平台 |

### 配置文件

可以创建JSON配置文件：

```json
{
  "dataset_path": "/your/dataset/path",
  "repos_path": "/your/dataset/path/repos", 
  "output_dir": "/your/output/path",
  "language": "py",
  "split": "test",
  "batch_size": 1,
  "limit": 10,
  "model_name": "openai-chat-completions",
  "model_params": {
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 1024
  }
}
```

然后使用配置文件：

```bash
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --config config.json
```

## 详细使用说明

### 1. 基本评估

```bash
# 评估Python项目，使用GPT-4o
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model gpt-4o \
    --language py

# 评估Java项目，使用Claude
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model claude-3.5-sonnet \
    --language java

# 评估Kotlin项目，限制5个样本用于测试
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model gpt-4o-mini \
    --language kt \
    --limit 5
```

### 2. 自定义模型配置

```bash
# 自定义OpenAI模型参数
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model openai-chat-completions \
    --model_args "model=gpt-4o,temperature=0.2,max_tokens=2048" \
    --language py

# 使用千帆模型（自定义参数）
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model openai-chat-completions \
    --model_args "model=qwen3-235b-a22b-instruct-2507,base_url=https://qianfan.baidubce.com/v2/chat/completions,temperature=0.1" \
    --language py
```

### 3. 路径配置

```bash
# 指定自定义数据集路径
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model gpt-4o \
    --dataset_path "/custom/dataset/path" \
    --repos_path "/custom/repos/path" \
    --output_dir "/custom/output/path" \
    --language py
```

### 4. 使用lm_eval命令行

也可以直接使用lm_eval命令行工具：

```bash
lm_eval --model openai-chat-completions \
        --model_args model=gpt-4o,temperature=0.1 \
        --tasks lca_bug_localization_py \
        --batch_size 1 \
        --limit 10
```

## Demo示例

### 示例1：快速测试（5个样本）

```bash
# 设置API密钥
export OPENAI_API_KEY="your-api-key"

# 运行快速测试
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model gpt-4o \
    --language py \
    --limit 5

# 预期输出：
# 🚀 开始 Bug Localization 评估
# 模型: openai-chat-completions
# 语言: py
# 样本限制: 5
# ✅ 模型初始化成功
# 📋 开始评估任务: ['lca_bug_localization_py']
# ✅ 评估完成
# 📊 评估结果摘要
# Precision: 0.xxxx
# Recall: 0.xxxx
# F1 Score: 0.xxxx
```

### 示例2：千帆模型评估

```bash
# 设置千帆API密钥（使用OPENAI_API_KEY格式）
export OPENAI_API_KEY="bce-v3/your-qianfan-key"

# 运行千帆模型评估
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model qianfan-qwen \
    --language py \
    --limit 3

# 输出包含千帆特定配置：
# 模型: openai-chat-completions
# 参数: model=qwen3-235b-a22b-instruct-2507,base_url=https://qianfan.baidubce.com/v2/chat/completions...
```

### 示例3：多语言评估

```bash
# 批量评估多种语言
for lang in py java kt; do
    echo "评估 $lang 项目..."
    python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
        --model gpt-4o-mini \
        --language $lang \
        --limit 3
done
```

### 示例4：配置文件使用

创建配置文件 `my_config.json`：

```json
{
  "language": "py",
  "split": "test", 
  "limit": 10,
  "batch_size": 1,
  "model_name": "openai-chat-completions",
  "model_params": {
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 1024
  }
}
```

使用配置文件：

```bash
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --config my_config.json
```

## 结果分析

评估完成后，结果将保存在输出目录中：

```
output_dir/eval_YYYYMMDD_HHMMSS/
├── evaluation_results.json      # 完整评估结果
├── evaluation_config.json       # 评估配置
├── evaluation_summary.json      # 结果摘要
└── samples/                     # 详细样本结果（如果启用）
```

主要指标说明：

- **F1 Score**: 最重要的综合指标，平衡精确率和召回率
- **Precision**: 预测文件的准确性
- **Recall**: 对目标文件的覆盖率
- **All Correct**: 完全正确预测的比例
- **At Least One Correct**: 至少预测对一个文件的比例

## 性能建议

1. **API模型**: 使用 `batch_size=1` 避免速率限制
2. **上下文控制**: 通过配置 `max_context_files` 控制输入文件数量
3. **快速测试**: 使用 `--limit 5` 进行快速验证
4. **并行处理**: 本地模型可使用更大的批处理大小

## 故障排除

### 常见问题

1. **数据集路径错误**
   ```
   FileNotFoundError: Dataset file not found
   ```
   - 检查环境变量 `LCA_DATASET_PATH` 或使用 `--dataset_path` 参数
   - 确保数据集文件存在且格式正确

2. **API密钥问题**
   ```
   ❌ 缺少环境变量: OPENAI_API_KEY
   ```
   - 设置正确的API密钥：`export OPENAI_API_KEY="your-key"`
   - 千帆模型也使用 `OPENAI_API_KEY` 环境变量

3. **模型配置错误**
   ```
   未知的模型配置: xxx
   ```
   - 使用 `--list_models` 查看可用配置
   - 或使用完整模型名称如 `openai-chat-completions`

4. **JSON解析错误**
   ```
   JSON解析错误 for text_id: ...
   ```
   - 模型输出格式不正确，可能需要调整温度参数
   - 检查模型是否支持JSON输出格式

### 调试方法

1. **运行测试**: `python -m ...test_case` 验证基本功能
2. **配置验证**: 使用 `--validate_config` 验证配置
3. **小规模测试**: 使用 `--limit 5` 进行调试
4. **详细日志**: 查看评估过程中的输出信息

## 本地使用方式

根据您的需求，推荐以下本地配置方式：

### 1. 环境变量配置（推荐）

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加：
export LCA_DATASET_PATH="/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/datasets/lca_bug_localization"
export LCA_OUTPUT_PATH="/Users/xiaoyunting/Documents/ModelDev/working_path/projects/lm-evaluation-harness/results/long_code_arena/bug_localization"
export OPENAI_API_KEY="your-api-key"
```

### 2. 创建本地配置文件

```bash
# 创建配置文件
cat > ~/bug_localization_config.json << EOF
{
  "dataset_path": "/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/datasets/lca_bug_localization",
  "repos_path": "/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/datasets/lca_bug_localization/repos",
  "output_dir": "/Users/xiaoyunting/Documents/ModelDev/working_path/projects/lm-evaluation-harness/results/long_code_arena/bug_localization",
  "language": "py",
  "limit": 10
}
EOF

# 使用配置文件
python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --config ~/bug_localization_config.json \
    --model gpt-4o
```

### 3. 快速启动脚本

创建一个快速启动脚本 `quick_eval.sh`：

```bash
#!/bin/bash
# Bug Localization 快速评估脚本

# 设置路径
export LCA_DATASET_PATH="/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/datasets/lca_bug_localization"
export LCA_OUTPUT_PATH="/Users/xiaoyunting/Documents/ModelDev/working_path/projects/lm-evaluation-harness/results/long_code_arena/bug_localization"

# 检查API密钥
if [ -z "$OPENAI_API_KEY" ]; then
    echo "请先设置 OPENAI_API_KEY 环境变量"
    exit 1
fi

# 运行评估
cd /Users/xiaoyunting/Documents/ModelDev/working_path/projects/lm-evaluation-harness

python -m lm_eval.tasks.long_code_arena.bug_localization.run_bug_localization_eval \
    --model ${1:-gpt-4o} \
    --language ${2:-py} \
    --limit ${3:-5}
```

使用方式：
```bash
chmod +x quick_eval.sh
./quick_eval.sh gpt-4o py 5
```

## 开发和扩展

### 添加新的模型配置

在 `config.py` 中的 `MODEL_CONFIGS` 字典中添加新配置：

```python
MODEL_CONFIGS['my-custom-model'] = {
    'model_name': 'your-model-type',
    'model_params': {
        'model': 'your-model-name',
        'base_url': 'your-api-url',
        # 其他参数...
    }
}
```

### 自定义评估指标

在 `metrics.py` 中添加新的指标函数：

```python
def your_custom_metric(expected_files, predicted_files):
    # 实现自定义指标
    return metric_value
```

### 修改提示词

在 `lca_bug_localization_task.py` 中修改 `doc_to_text` 方法来自定义提示词格式。

## 引用

如果您使用此实现，请引用 Long Code Arena 论文：

```bibtex
@article{bogomolov2024long,
  title={Long Code Arena: a Set of Benchmarks for Long-Context Code Models},
  author={Bogomolov, Egor and Eliseeva, Aleksandra and Galimzyanov, Timur and Glukhov, Evgeniy and Shapkin, Anton and Tigina, Maria and Golubev, Yaroslav and Kovrigin, Alexander and van Deursen, Arie and Izadi, Maliheh and others},
  journal={arXiv preprint arXiv:2406.11612},
  year={2024}
}
```

论文链接：https://arxiv.org/abs/2406.11612