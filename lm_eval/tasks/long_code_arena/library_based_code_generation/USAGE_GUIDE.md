# Long Code Arena - Library-Based Code Generation - 使用指南

## 概述

我们已经成功整合了 Long Code Arena 的 library-based code generation 任务到 lm-evaluation-harness 框架中，并集成了官方的评估函数。

## 可用任务配置

### 1. 基础版本（推荐用于快速测试）
```bash
lm_eval --model dummy \
        --tasks lca_library_based_code_generation_fallback \
        --limit 2 \
        --confirm_run_unsafe_code
```

### 2. 完整版本（使用真实数据集）
```bash
lm_eval --model dummy \
        --tasks lca_library_based_code_generation \
        --limit 2 \
        --confirm_run_unsafe_code
```

### 3. 增强版本（包含官方评估指标）
```bash
lm_eval --model dummy \
        --tasks lca_library_based_code_generation_enhanced \
        --limit 2 \
        --confirm_run_unsafe_code
```

## 数据集配置

### 真实数据集路径
- **数据集位置**: `/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/datasets/library_based_code_generation/`
- **数据文件**: `data/test-00000-of-00001-518ed46ecbe35ff9.parquet`
- **样本数量**: 150个测试样本

### 数据集字段说明
- `instruction`: 任务描述（自然语言）
- `reference`: 参考代码实现
- `clean_reference`: 去除注释的参考代码
- `unique_apis`: 目标API列表
- `repo_full_name`: 源代码仓库名称
- 其他字段: 包含项目元数据和API分析信息

## 评估指标

### 1. Pass@1 (所有版本)
- **描述**: 衡量生成代码的功能正确性
- **评估内容**: 语法正确性、库使用、语义模式匹配
- **范围**: 0-1，越高越好

### 2. ChrF Score (增强版本)
- **描述**: 字符级别的相似度评分
- **评估内容**: 生成代码与参考代码的字符级重叠
- **范围**: 0-1，越高越好

### 3. API Recall (增强版本)
- **描述**: API使用召回率
- **评估内容**: 生成代码中正确使用的目标API比例
- **范围**: 0-1，越高越好

## 评估函数集成

### 官方评估函数来源
- **路径**: `/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/eval_func/library_based_code_generation/`
- **主要组件**:
  - ChrF评分（基于sacrebleu）
  - API召回率（基于tree-sitter解析）
  - 代码提取和清理

### 依赖库说明
```bash
# 完整功能需要安装（可选）
pip install sacrebleu tree-sitter tree-sitter-python

# 不安装也可以运行，会使用fallback实现
```

## 文件结构

```
library_based_code_generation/
├── library_based_code_generation.yaml              # 主任务配置
├── library_based_code_generation_fallback.yaml     # 离线版本
├── library_based_code_generation_enhanced.yaml     # 增强版本
├── utils.py                                         # 评估函数（已增强）
├── data_loader.py                                   # 数据加载器
├── sample_data.json                                 # 样本数据
├── real_sample_data.json                           # 真实数据样本
├── lca_custom_task.py                              # 自定义任务类
├── download_dataset.py                             # 数据下载脚本
└── README.md                                        # 任务文档
```

## 测试结果示例

```
dummy (), gen_kwargs: (None), limit: 2.0, num_fewshot: None, batch_size: 1
|                  Tasks                   |Version|  Filter   |n-shot|     Metric      |   |Value |   |Stderr|
|------------------------------------------|------:|-----------|-----:|-----------------|---|-----:|---|-----:|
|lca_library_based_code_generation_enhanced|    1.1|create_test|     0|api_recall_metric|↑  |0.0000|±  |0     |
|                                          |       |create_test|     0|chrf_metric      |↑  |0.0005|±  |0     |
|                                          |       |create_test|     0|pass_at_1        |↑  |0.0000|±  |0     |
```

## 实际使用建议

### 1. 快速验证
```bash
# 使用fallback版本进行快速测试
lm_eval --model dummy --tasks lca_library_based_code_generation_fallback --limit 5
```

### 2. 完整评估
```bash
# 使用真实模型和完整数据集
lm_eval --model hf \
        --model_args pretrained=codellama/CodeLlama-7b-Python-hf \
        --tasks lca_library_based_code_generation_enhanced \
        --confirm_run_unsafe_code
```

### 3. 自定义评估
- 修改 `utils.py` 中的评估函数
- 调整 YAML 配置中的 generation_kwargs
- 添加新的评估指标

## 故障排除

### 常见问题
1. **ImportError for tree-sitter**: 使用fallback AST解析
2. **ImportError for sacrebleu**: 使用简单字符相似度
3. **unsafe code warning**: 添加 `--confirm_run_unsafe_code` 参数

### 调试模式
```bash
# 增加详细日志
lm_eval --model dummy --tasks lca_library_based_code_generation_enhanced --limit 1 --log_samples
```

## 扩展功能

### 添加新评估指标
1. 在 `utils.py` 中定义新函数
2. 在 YAML 配置的 `metric_list` 中添加
3. 确保函数签名符合框架要求

### 集成其他LCA任务
- 使用相同的模式创建其他任务目录
- 参考 `_default_template_yaml` 模板
- 更新 `long_code_arena.yaml` 组配置

## 总结

✅ **已完成功能**:
- 数据集加载（本地parquet文件）
- 官方评估指标集成（ChrF + API Recall）
- 多版本任务配置（基础/完整/增强）
- 完整的端到端测试
- 详细的文档和使用指南

这个实现现在可以用于评估各种代码生成模型在library-based代码生成任务上的性能。
