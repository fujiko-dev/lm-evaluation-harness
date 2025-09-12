# LCA Project Level Code Completion 任务

## 概述

Project Level Code Completion（项目级代码补全）是 Long Code Arena 基准测试中的一个重要任务，旨在评估模型在项目上下文中进行代码补全的能力。

## 任务定义

**输入**: 项目上下文、代码文件、目标位置
**输出**: 补全的代码行
**评估**: Exact Match、Edit Distance、BLEU Score等多维度指标

该任务考验模型对项目结构的理解能力以及代码生成的准确性。

## 数据集信息

- **来源**: LCA Project Level Code Completion 数据集
- **语言**: Python
- **样本数**: 根据上下文大小变化
- **格式**: Parquet 文件
- **上下文大小**: small_context, medium_context, large_context, huge_context

### 数据集示例
```json
{
  "repo": "example/project",
  "file_path": "src/main.py",
  "completion_lines": "def hello():\n    print('Hello World')",
  "context": "import os\nimport sys\n",
  "target_lines": "def hello():\n    print('Hello World')"
}
```

## 文件结构

```
project_level_code_completion/
├── config.py                                    # ① 配置管理
├── data_loader.py                               # ② 数据加载器  
├── lca_project_level_code_completion_task.py   # ③ 任务实现
├── project_level_code_completion.yaml           # ④ YAML配置
├── metrics.py                                   # ⑤ 评估指标
├── run_project_level_code_completion_eval.py   # ⑥ 评估脚本
├── test_case.py                                 # ⑦ 测试用例
├── api_models.py                                # ⑧ API模型支持
└── README.md                                    # ⑨ 使用手册
```

## 评估指标

| 指标名称 | 描述 | 范围 | 越高越好 |
|---------|------|------|----------|
| **Exact Match** | 完全匹配的补全 | 0.0-1.0 | ✅ |
| **Edit Distance** | 编辑距离 | 0.0+ | ❌ |
| **BLEU Score** | 基于n-gram的文本相似度 | 0.0-1.0 | ✅ |

## 快速开始

### 环境设置

```bash
# 1. 设置API密钥（根据使用的模型）
export OPENAI_API_KEY="your_openai_api_key"          # OpenAI模型
export ANTHROPIC_API_KEY="your_anthropic_api_key"    # Anthropic模型

# 2. 设置数据集路径（可选，有默认值）
export LCA_PLCC_DATASET_PATH="/path/to/project-level-code-completion/dataset"

# 3. 设置输出路径（可选，有默认值）
export LCA_PLCC_OUTPUT_PATH="/path/to/output/directory"
```

### 基础使用

#### 1. 使用预设模型配置

```bash
# GPT-4o 评估
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model gpt-4o \
    --context_size small_context \
    --limit 10

# 千帆模型评估
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model qianfan-qwen \
    --context_size small_context \
    --limit 10

# Claude 3.5 Sonnet 评估
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model claude-3.5-sonnet \
    --context_size small_context \
    --limit 10
```

#### 2. 自定义模型参数

```bash
# 自定义 GPT 模型
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model openai-chat-completions \
    --model_args "model=gpt-4o,temperature=0.1,max_tokens=200" \
    --context_size medium_context \
    --limit 20

# 自定义千帆模型
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model openai-chat-completions \
    --model_args "model=qwen3-235b-a22b-instruct-2507,base_url=https://qianfan.baidubce.com/v2/chat/completions,temperature=0.1,apply_chat_template=true" \
    --context_size large_context \
    --limit 20
```

#### 3. 完整参数示例

```bash
# 使用自定义API的GPT模型
export OPENAI_API_KEY="sk-your-custom-api-key"

python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model openai-chat-completions \
    --model_args "model=gpt-4.1,base_url=http://custom-api-url:8080/v1/chat/completions,temperature=0.8,top_p=0.5,max_tokens=200,max_retries=10,timeout=120,apply_chat_template=true" \
    --context_size small_context \
    --limit 22 \
    --dataset_path "/custom/path/to/dataset" \
    --output_dir "/custom/path/to/output"
```

### 高级使用

#### 配置文件模式

1. **创建配置文件**:
```json
{
  "model_name": "openai-chat-completions",
  "model_params": {
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 200
  },
  "context_size": "small_context",
  "dataset_path": "/path/to/dataset",
  "output_dir": "/path/to/output",
  "batch_size": 1,
  "limit": 50
}
```

2. **使用配置文件运行**:
```bash
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --config my_config.json
```

#### 查看可用模型

```bash
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --list_models
```

#### 验证配置

```bash
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model gpt-4o \
    --context_size small_context \
    --validate_config
```

## 示例场景

### 场景1: 快速原型测试
```bash
# 只测试5个样本，快速验证模型
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model gpt-4o-mini \
    --context_size small_context \
    --limit 5
```

### 场景2: 完整评估
```bash
# 完整数据集评估，保存详细结果
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model gpt-4o \
    --context_size small_context \
    --output_dir "./results/gpt4o_full_eval"
```

### 场景3: 不同上下文大小对比
```bash
# 评估不同上下文大小的影响
for context in small_context medium_context large_context; do
    python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
        --model gpt-4o \
        --context_size $context \
        --limit 50 \
        --output_dir "./results/gpt4o_${context}_comparison"
done
```

## 评估结果

### 输出文件
- `evaluation_results.json`: 完整评估结果
- `evaluation_config.json`: 评估配置
- `evaluation_summary.json`: 结果摘要
- `error_log.txt`: 错误日志（如有）

### 结果示例
```json
{
  "results": {
    "lca_project_level_code_completion": {
      "exact_match": 0.2847,
      "edit_distance": 2.1345,
      "bleu_score": 0.7234
    }
  }
}
```

## 性能基准

### 参考性能（10%采样）

| 模型 | 上下文大小 | Exact Match | Edit Distance | BLEU Score | 平均耗时 |
|------|------------|-------------|---------------|------------|----------|
| GPT-4o | small_context | 0.285 | 2.13 | 0.723 | ~3分钟 |
| GPT-4o-mini | small_context | 0.241 | 2.45 | 0.689 | ~2分钟 |
| Claude-3.5-Sonnet | small_context | 0.267 | 2.23 | 0.701 | ~4分钟 |

## 故障排除

### 常见问题

1. **数据集未找到**
   ```
   错误: FileNotFoundError: 数据集路径不存在
   解决: 检查数据集路径，设置正确的 dataset_path
   ```

2. **API密钥错误**
   ```
   错误: 缺少环境变量 OPENAI_API_KEY
   解决: 正确设置API密钥环境变量
   ```

3. **上下文大小不支持**
   ```
   错误: 不支持的上下文大小
   解决: 使用支持的值: small_context, medium_context, large_context, huge_context
   ```

4. **内存不足**
   ```
   解决: 减少 batch_size 或使用 limit 参数限制样本数量
   ```

### 调试模式

启用详细日志:
```bash
python3 -m lm_eval.tasks.long_code_arena.project_level_code_completion.run_project_level_code_completion_eval \
    --model gpt-4o \
    --context_size small_context \
    --limit 5 \
    --verbose
```

## 自定义扩展

### 添加新的评估指标

在 `metrics.py` 中添加新函数:
```python
def custom_metric(predictions, references):
    # 实现自定义指标
    return score
```

在任务类中注册:
```python
def aggregation(self):
    result = super().aggregation()
    result["custom_metric"] = ["custom_metric", "mean"]
    return result
```

### 修改提示模板

在 `lca_project_level_code_completion_task.py` 的 `doc_to_text` 方法中修改提示生成逻辑。

## 最佳实践

1. **上下文大小选择**: 根据模型能力和任务需求选择合适的上下文大小
2. **样本选择**: 对于快速测试使用 `--limit 10`，正式评估使用完整数据集
3. **批处理**: 保持 `batch_size=1` 以获得最佳结果质量
4. **温度设置**: 代码补全建议使用较低温度 (0.1-0.3)
5. **结果保存**: 始终指定 `output_dir` 以保存详细结果

## 技术细节

### 上下文大小说明
- **small_context**: 最小上下文，适合快速测试
- **medium_context**: 中等上下文，平衡性能和准确性
- **large_context**: 大上下文，提供更多项目信息
- **huge_context**: 最大上下文，包含完整项目结构

### 提示构建策略
- 项目上下文信息
- 目标文件路径和位置
- 代码补全任务说明
- 生成指导和要求

### 后处理逻辑
- 代码格式化和清理
- 长度控制和截断
- 语法检查和验证

### 指标计算
- Exact Match: 完全匹配的补全比例
- Edit Distance: 基于编辑距离的相似度
- BLEU Score: 基于n-gram的文本相似度

---

如有问题或建议，请查看项目文档或提交Issue。