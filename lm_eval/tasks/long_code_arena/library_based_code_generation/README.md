# Long Code Arena - Library-Based Code Generation

This directory contains the implementation of the library-based code generation task from the Long Code Arena benchmark suite.

## Task Description

The library-based code generation task requires models to generate Python code that uses specific libraries based on natural language instructions. This tests the model's ability to:

1. Understand library-specific APIs and functions
2. Generate syntactically correct Python code
3. Use appropriate library functions for the given task
4. Follow programming best practices

## Files

- `library_based_code_generation.yaml` - Main task configuration (requires HuggingFace dataset)
- `library_based_code_generation_fallback.yaml` - Offline-capable task configuration with sample data
- `utils.py` - Evaluation metrics and utility functions
- `data_loader.py` - Dataset loading utilities with fallback support
- `sample_data.json` - Sample data for offline testing
- `download_dataset.py` - Script to download the real dataset from HuggingFace
- `lca_custom_task.py` - Custom task class implementation

## Usage

### Basic Usage (Works Offline)
```bash
lm_eval --model dummy \
        --tasks lca_library_based_code_generation_fallback \
        --num_fewshot 0
```

### With Real Dataset
```bash
# First download the dataset
python lm_eval/tasks/long_code_arena/library_based_code_generation/download_dataset.py

# Then run evaluation
lm_eval --model hf \
        --model_args pretrained=codellama/CodeLlama-7b-Python-hf \
        --tasks lca_library_based_code_generation \
        --num_fewshot 0
```

## Dataset

The dataset comes from HuggingFace: `JetBrains-Research/lca-library-based-code-generation`

Each example contains:
- `instruction`: Natural language description of what code to generate
- `reference`: Reference implementation
- `repo_full_name`: Source repository information
- `libraries`: List of required libraries
- `difficulty`: Task difficulty level

## Evaluation

The task uses pass@1 evaluation, measuring whether the generated code is functionally correct. The evaluation includes:
- Syntax validation
- Library usage validation
- Semantic pattern matching
- Functional correctness checks

## Paper Reference

This task is part of the Long Code Arena benchmark suite described in:
"Long Code Arena: a Set of Benchmarks for Long-Context Code Models" (https://arxiv.org/abs/2406.11612)
