# CI Builds Repair Task - 使用手册

## 任务简介

CI Builds Repair 是 Long Code Arena 基准测试的一部分，用于评估大语言模型在修复持续集成（CI）构建失败方面的能力。

**任务定义**: 给定一个失败的 GitHub Actions 工作流日志和相应的仓库快照，修复仓库内容以使工作流能够通过。

## 文件结构

经过整理后的文件结构如下：

```
ci_builds_repair/
├── __init__.py                    # 包初始化
├── config.py                     # ① 配置管理
├── data_loader.py                # ② 数据集加载
├── metrics.py                    # ③ 评估指标
├── lca_ci_builds_repair_task.py  # ④ 任务注册和实现
├── run_ci_builds_repair_eval.py  # ⑤ 统筹评估脚本
├── test_case.py                  # ⑥ 测试用例
├── ci_builds_repair.yaml         # ⑦ YAML配置文件
├── README.md                     # ⑧ 使用手册
└── test_ci_builds_repair_old.py  # 旧测试文件（参考用）
```

### 文件功能说明

- **config.py**: 统一配置管理，包含路径、模型参数等所有可配置项
- **data_loader.py**: 负责加载和处理CI构建修复数据集
- **metrics.py**: 实现各种评估指标（成功率、BLEU、ROUGE等）
- **lca_ci_builds_repair_task.py**: 集成到lm-evaluation-harness的任务类
- **run_ci_builds_repair_eval.py**: 主评估脚本，支持命令行参数配置
- **test_case.py**: 完整的测试套件，验证所有组件功能
- **ci_builds_repair.yaml**: lm-eval框架的YAML配置文件

## 环境配置

### 1. 数据集配置

设置数据集路径环境变量（可选，有默认值）：
```bash
export LCA_CI_DATASET_PATH="/path/to/lca_ci_builds_repair"
export LCA_CI_OUTPUT_PATH="/path/to/output"
```

### 2. API密钥配置

根据使用的模型设置相应的API密钥：

```bash
# OpenAI模型
export OPENAI_API_KEY="your_openai_api_key"

# Anthropic模型  
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# 千帆模型（使用OPENAI_API_KEY格式）
export OPENAI_API_KEY="your_qianfan_api_key"
```

## 使用方法

### 1. 快速测试

```bash
# 运行测试套件验证安装
cd lm_eval/tasks/long_code_arena/ci_builds_repair
python test_case.py
```

### 2. 预设模型评估

系统提供以下预设模型配置：

```bash
# 查看可用预设模型
python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval --list_models

# 使用GPT-4o（推荐）
python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --model gpt-4o \
    --limit 5

# 使用GPT-4o Mini（经济型）
python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --model gpt-4o-mini \
    --limit 10

# 使用Claude 3.5 Sonnet
python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --model claude-3.5-sonnet \
    --limit 5

# 使用千帆模型
python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --model qianfan-qwen \
    --limit 5
```

### 3. 自定义模型评估

#### GPT-4.1 自定义API示例（如您的需求）

```bash
export OPENAI_API_KEY="sk-9RcZoJ0dFdcRVmjMqUTx5c5mVqlr8CFcAA7NgXqw86Tt88bi"

python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --model openai-chat-completions \
    --model_args "model=gpt-4.1,base_url=http://211.23.3.237:27544/v1/chat/completions,temperature=0.8,top_p=0.5,max_tokens=8192,max_retries=10,timeout=120,apply_chat_template=true" \
    --limit 5 \
    --dataset_path "/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/datasets/lca_ci_builds_repair" \
    --output_dir "/Users/xiaoyunting/Documents/ModelDev/working_path/projects/lm-evaluation-harness/results/gpt41_ci_builds_repair"
```

### 4. 配置文件方式

#### 创建配置文件

```bash
# 保存当前配置
python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --model gpt-4o \
    --limit 10 \
    --save_config my_config.json

# 从配置文件运行
python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --config my_config.json
```

#### 配置文件示例 (my_config.json)

```json
{
  "dataset_path": "/path/to/lca_ci_builds_repair",
  "output_dir": "/path/to/results",
  "split": "test",
  "batch_size": 1,
  "limit": 10,
  "max_context_length": 16000,
  "include_logs": true,
  "model_name": "openai-chat-completions",
  "model_params": {
    "model": "gpt-4o",
    "max_tokens": 2048,
    "temperature": 0.1,
    "num_concurrent": 5,
    "max_retries": 3,
    "timeout": 120,
    "apply_chat_template": true
  }
}
```

## 可配置参数

### 主要参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--model` | 模型名称或预设配置 | 必需 | `gpt-4o`, `openai-chat-completions` |
| `--model_args` | 模型参数字符串 | - | `model=gpt-4o,temperature=0.1` |
| `--limit` | 样本数量限制 | None(全部) | `5`, `10`, `50` |
| `--dataset_path` | 数据集路径 | 环境变量/默认路径 | `/path/to/dataset` |
| `--output_dir` | 输出目录 | 环境变量/默认路径 | `/path/to/results` |
| `--split` | 数据拆分 | `test` | `train`, `dev`, `test` |
| `--batch_size` | 批处理大小 | `1` | `1`, `2`, `4` |

### 模型参数

model_args字符串支持的参数：

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `model` | str | 模型名称 | `gpt-4o`, `claude-3-5-sonnet-20241022` |
| `base_url` | str | API基础URL | `https://api.openai.com/v1` |
| `api_key` | str | API密钥（推荐用环境变量） | `sk-xxx` |
| `max_tokens` | int | 最大生成token数 | `2048`, `4096` |
| `temperature` | float | 生成温度 | `0.1`, `0.8` |
| `top_p` | float | 核采样参数 | `0.5`, `0.9` |
| `max_retries` | int | 最大重试次数 | `3`, `10` |
| `timeout` | int | 请求超时（秒） | `60`, `120` |
| `num_concurrent` | int | 并发请求数 | `1`, `5` |
| `apply_chat_template` | bool | 应用对话模板 | `true`, `false` |

## 评估指标

系统自动计算以下指标：

1. **CI Repair Score**: 综合修复质量评分 (0-1)
2. **Success Rate**: 修复成功率 (0-1)  
3. **BLEU Score**: 与目标修复的BLEU相似度 (0-1)
4. **ROUGE-L**: 与目标修复的ROUGE-L相似度 (0-1)
5. **Exact Match**: 精确匹配率 (0-1)

## 结果输出

评估完成后，结果保存在指定的输出目录中：

```
output_dir/
├── eval_YYYYMMDD_HHMMSS/
│   ├── evaluation_results.json    # 完整评估结果
│   ├── evaluation_config.json     # 评估配置
│   ├── evaluation_summary.json    # 结果摘要
│   └── error_log.txt              # 错误日志（如有）
```

## 数据集信息

- **任务类型**: 代码修复
- **编程语言**: 主要为Python
- **数据格式**: Parquet文件
- **样本包含**: 
  - 仓库信息 (repo_name)
  - 构建失败原因 (build_failure_reason)
  - 构建日志 (logs)
  - 相关文件 (related_files)
  - 目标修复 (target_fix)

## Demo示例

### 示例1: 快速评估（10%采样）

```bash
export OPENAI_API_KEY="your_api_key"
cd /path/to/lm-evaluation-harness

python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --model gpt-4o \
    --limit 5 \
    --output_dir "./results/ci_repair_demo"
```

### 示例2: 千帆模型评估

```bash
export OPENAI_API_KEY="your_qianfan_key"

python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --model openai-chat-completions \
    --model_args "model=qwen3-235b-a22b-instruct-2507,base_url=https://qianfan.baidubce.com/v2/chat/completions,temperature=0.1,max_tokens=2048,max_retries=10,timeout=120,stop=<|endoftext|>,apply_chat_template=true" \
    --limit 10
```

### 示例3: 自定义参数评估

```bash
python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval \
    --model openai-chat-completions \
    --model_args "model=gpt-4.1,base_url=http://custom-api.com/v1/chat/completions,temperature=0.8,top_p=0.5,max_tokens=8192" \
    --limit 5 \
    --dataset_path "/custom/dataset/path" \
    --output_dir "/custom/output/path"
```

## 故障排除

### 常见问题

1. **配置验证失败**
   ```bash
   # 验证配置
   python -m lm_eval.tasks.long_code_arena.ci_builds_repair.run_ci_builds_repair_eval --validate_config --model gpt-4o
   ```

2. **数据集路径错误**
   - 检查环境变量: `echo $LCA_CI_DATASET_PATH`
   - 检查文件存在: `ls /path/to/lca_ci_builds_repair/data/python/`

3. **API密钥问题**
   - 检查环境变量: `echo $OPENAI_API_KEY`
   - 验证密钥有效性

4. **模型加载失败**
   - 查看可用模型: `--list_models`
   - 检查model_args格式

### 调试模式

```bash
# 运行测试验证环境
python -c "
import sys
sys.path.append('.')
from lm_eval.tasks.long_code_arena.ci_builds_repair.test_case import main
main()
"
```

## 性能建议

1. **采样策略**: 
   - 开发/调试: `--limit 5`
   - 快速评估: `--limit 25` (约50%样本)  
   - 完整评估: 不使用limit参数

2. **并发设置**: 
   - 调整`num_concurrent`参数平衡速度和稳定性
   - 推荐值: 1-5

3. **超时设置**:
   - 复杂任务建议`timeout=120`或更高
   - 网络不稳定时增加`max_retries`

## 技术支持

如遇到问题，请：

1. 运行`test_case.py`检查环境
2. 使用`--validate_config`验证配置  
3. 查看输出目录中的`error_log.txt`
4. 检查API密钥和网络连接

---

**版本**: 1.0  
**更新时间**: 2025-09-11  
**兼容性**: lm-evaluation-harness v0.4+