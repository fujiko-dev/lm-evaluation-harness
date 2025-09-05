# Long Code Arena

This directory contains implementations of tasks from the Long Code Arena benchmark suite, designed to evaluate long-context code models.

## Overview

Long Code Arena is a comprehensive benchmark suite that evaluates language models' ability to understand and generate code that requires project-wide context. The benchmark includes six main tasks:

1. **Library-based Code Generation** ✅ (Implemented)
2. CI Builds Repair 
3. Project-Level Code Completion
4. Commit Message Generation  
5. Bug Localization
6. Module Summarization

## Current Implementation

### Library-Based Code Generation (`library_based_code_generation/`)

This task evaluates models' ability to generate Python code that uses specific libraries based on natural language instructions.

**Usage:**
```bash
# Offline version with sample data
lm_eval --model dummy \
        --tasks lca_library_based_code_generation_fallback \
        --num_fewshot 0

# Full version (requires dataset download)
lm_eval --model hf \
        --model_args pretrained=codellama/CodeLlama-7b-Python-hf \
        --tasks lca_library_based_code_generation \
        --num_fewshot 0
```

## Directory Structure

```
long_code_arena/
├── __init__.py
├── README.md                           # This file
├── IMPLEMENTATION_SUMMARY.md           # Implementation details
├── _default_template_yaml              # Template for future tasks
├── long_code_arena.yaml               # Group configuration
└── library_based_code_generation/     # Library-based code generation task
    ├── __init__.py
    ├── README.md
    ├── library_based_code_generation.yaml
    ├── library_based_code_generation_fallback.yaml
    ├── utils.py
    ├── data_loader.py
    ├── sample_data.json
    ├── download_dataset.py
    └── lca_custom_task.py
```

## Paper Reference

This benchmark suite is described in:
"Long Code Arena: a Set of Benchmarks for Long-Context Code Models" (https://arxiv.org/abs/2406.11612)

## Future Tasks

The remaining five Long Code Arena tasks will be implemented in separate subdirectories following the same structure pattern.
