# LCA Commit Message Generation 任务

## 概述

Commit Message Generation（提交消息生成）是 Long Code Arena 基准测试中的一个重要任务，旨在评估模型根据代码变更自动生成高质量提交消息的能力。

## 任务定义

**输入**: 代码差异（diff）、仓库信息、文件变更统计
**输出**: 简洁准确的提交消息
**评估**: BLEU、ROUGE、BERT Score等多维度指标

该任务考验模型对代码变更的理解能力以及技术写作能力。

## 数据集信息

- **来源**: CommitChronicle-Py-Long 数据集
- **语言**: Python
- **样本数**: ~220个测试样本
- **格式**: Parquet 文件
- **字段**: repo, commit_sha, message, diff, mods, adds, dels

### 数据集示例
```json
{
  "repo": "example/project",
  "commit_sha": "abc123def456",
  "message": "Fix authentication bug in user login",
  "diff": "@@ -15,7 +15,7 @@ def authenticate(username, password):\n-    if validate_user(username, password):\n+    if validate_user_secure(username, password):",
  "mods": ["auth/login.py"],
  "adds": [],
  "dels": []
}
```

## 文件结构

```
commit_message_generation/
├── config.py                              # ① 配置管理
├── data_loader.py                         # ② 数据加载器  
├── lca_commit_message_generation_task.py  # ③ 任务实现
├── lca_commit_message_generation.yaml     # ④ YAML配置
├── metrics.py                             # ⑤ 评估指标
├── run_commit_message_generation_eval.py  # ⑥ 评估脚本
├── test_case.py                           # ⑦ 测试用例
└── README.md                              # ⑧ 使用手册
```

## 评估指标

| 指标名称 | 描述 | 范围 | 越高越好 |
|---------|------|------|----------|
| **BLEU Score** | 基于n-gram的文本相似度 | 0.0-1.0 | ✅ |
| **ROUGE-1** | 基于unigram的召回率 | 0.0-1.0 | ✅ |
| **ROUGE-2** | 基于bigram的召回率 | 0.0-1.0 | ✅ |
| **ROUGE-L** | 基于最长公共子序列 | 0.0-1.0 | ✅ |
| **BERT F1** | 基于语义的相似度 | 0.0-1.0 | ✅ |
| **Length Ratio** | 长度比例 | 0.0+ | 接近1.0更好 |

## 快速开始

### 环境设置

```bash
# 1. 设置API密钥（根据使用的模型）
export OPENAI_API_KEY="your_openai_api_key"          # OpenAI模型
export ANTHROPIC_API_KEY="your_anthropic_api_key"    # Anthropic模型

# 2. 设置数据集路径（可选，有默认值）
export LCA_CMG_DATASET_PATH="/path/to/commit-message-generation/dataset"

# 3. 设置输出路径（可选，有默认值）
export LCA_CMG_OUTPUT_PATH="/path/to/output/directory"
```

### 基础使用

#### 1. 使用预设模型配置

```bash
# GPT-4o 评估
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model gpt-4o \
    --limit 10

# 千帆模型评估
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model qianfan-qwen \
    --limit 10

# Claude 3.5 Sonnet 评估
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model claude-3.5-sonnet \
    --limit 10
```

#### 2. 自定义模型参数

```bash
# 自定义 GPT 模型
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model openai-chat-completions \
    --model_args "model=gpt-4o,temperature=0.1,max_tokens=200" \
    --limit 20

# 自定义千帆模型
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model openai-chat-completions \
    --model_args "model=qwen3-235b-a22b-instruct-2507,base_url=https://qianfan.baidubce.com/v2/chat/completions,temperature=0.1,apply_chat_template=true" \
    --limit 20
```

#### 3. 完整参数示例

```bash
# 使用自定义API的GPT模型
export OPENAI_API_KEY="sk-your-custom-api-key"

python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model openai-chat-completions \
    --model_args "model=gpt-4.1,base_url=http://custom-api-url:8080/v1/chat/completions,temperature=0.8,top_p=0.5,max_tokens=200,max_retries=10,timeout=120,apply_chat_template=true" \
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
  "dataset_path": "/path/to/dataset",
  "output_dir": "/path/to/output",
  "batch_size": 1,
  "limit": 50
}
```

2. **使用配置文件运行**:
```bash
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --config my_config.json
```

#### 查看可用模型

```bash
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --list_models
```

#### 验证配置

```bash
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model gpt-4o \
    --validate_config
```

## 示例场景

### 场景1: 快速原型测试
```bash
# 只测试5个样本，快速验证模型
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model gpt-4o-mini \
    --limit 5
```

### 场景2: 完整评估
```bash
# 完整数据集评估，保存详细结果
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model gpt-4o \
    --output_dir "./results/gpt4o_full_eval"
```

### 场景3: 模型对比
```bash
# 评估多个模型进行对比
for model in gpt-4o gpt-4o-mini claude-3.5-sonnet; do
    python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
        --model $model \
        --limit 50 \
        --output_dir "./results/${model}_comparison"
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
    "lca_commit_message_generation": {
      "commit_bleu": 0.2847,
      "commit_rouge1": 0.4521,
      "commit_rouge2": 0.2134,
      "commit_rougeL": 0.3986,
      "commit_bert_f1": 0.7234,
      "length_ratio": 1.0231
    }
  }
}
```

## 性能基准

### 参考性能（10%采样，~22个样本）

| 模型 | BLEU | ROUGE-1 | ROUGE-L | BERT F1 | 平均耗时 |
|------|------|---------|---------|---------|----------|
| GPT-4o | 0.285 | 0.452 | 0.399 | 0.723 | ~3分钟 |
| GPT-4o-mini | 0.241 | 0.398 | 0.356 | 0.689 | ~2分钟 |
| Claude-3.5-Sonnet | 0.267 | 0.421 | 0.378 | 0.701 | ~4分钟 |

## 故障排除

### 常见问题

1. **数据集未找到**
   ```
   错误: FileNotFoundError: 本地数据集路径不存在
   解决: 检查数据集路径，设置正确的 dataset_path
   ```

2. **API密钥错误**
   ```
   错误: 缺少环境变量 OPENAI_API_KEY
   解决: 正确设置API密钥环境变量
   ```

3. **内存不足**
   ```
   解决: 减少 batch_size 或使用 limit 参数限制样本数量
   ```

4. **网络超时**
   ```
   解决: 增加 timeout 参数值，或检查网络连接
   ```

### 调试模式

启用详细日志:
```bash
python3 -m lm_eval.tasks.long_code_arena.commit_message_generation.run_commit_message_generation_eval \
    --model gpt-4o \
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

在 `lca_commit_message_generation_task.py` 的 `doc_to_text` 方法中修改提示生成逻辑。

## 最佳实践

1. **样本选择**: 对于快速测试使用 `--limit 10`，正式评估使用完整数据集
2. **批处理**: 保持 `batch_size=1` 以获得最佳结果质量
3. **温度设置**: 提交消息生成建议使用较低温度 (0.1-0.3)
4. **上下文长度**: 确保 `max_context_length` 足够容纳完整的diff
5. **结果保存**: 始终指定 `output_dir` 以保存详细结果

## 技术细节

### 提示构建策略
- 系统消息：明确任务角色和要求
- 上下文信息：仓库名、文件变更统计
- 代码差异：格式化的diff内容
- 生成指导：提交消息最佳实践

### 后处理逻辑
- 移除无关前缀
- 提取第一行作为提交消息
- 长度控制和格式清理

### 指标计算
- BLEU：使用 sacrebleu 库
- ROUGE：使用 rouge_score 库  
- BERT Score：使用 bert_score 库

---

如有问题或建议，请查看项目文档或提交Issue。