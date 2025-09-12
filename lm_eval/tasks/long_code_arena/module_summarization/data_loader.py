"""
Data loader for LCA Module Summarization dataset from local files
"""

import os
import json
import pandas as pd
from datasets import load_dataset, Dataset
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from .config import ModuleSummarizationConfig
except ImportError:
    from lm_eval.tasks.long_code_arena.module_summarization.config import ModuleSummarizationConfig


def load_lca_module_summarization_dataset(config: Union[ModuleSummarizationConfig, Dict[str, Any], str, None] = None) -> List[Dict[str, Any]]:
    """
    加载本地LCA Module Summarization数据集
    
    Args:
        config: 配置对象、配置字典、路径字符串或None（使用默认配置）
        
    Returns:
        List[Dict]: 包含任务数据的字典列表
    """
    # 处理配置
    if config is None:
        config = ModuleSummarizationConfig()
    elif isinstance(config, dict):
        config = ModuleSummarizationConfig(config)
    elif isinstance(config, str):
        # 如果是字符串，当作数据集路径处理
        config = ModuleSummarizationConfig({'dataset_path': config})
    elif not isinstance(config, ModuleSummarizationConfig):
        raise ValueError(f"不支持的配置类型: {type(config)}")
    
    dataset_path = config.dataset_path
    split = config.get('split', 'test')
    
    try:
        print(f"🔄 正在从本地路径加载Module Summarization数据集: {dataset_path}")
        
        # 检查本地数据集是否存在
        if not dataset_path.exists():
            raise FileNotFoundError(f"本地数据集路径不存在: {dataset_path}")
        
        # 尝试使用datasets库加载本地数据集
        dataset = load_dataset(str(dataset_path))
        test_dataset = dataset[split]
        
        print(f"✅ 成功加载数据集，包含 {len(test_dataset)} 个样本")
        
        # 转换为字典列表格式
        data_list = []
        for i, item in enumerate(test_dataset):
            data_dict = {
                'idx': i,
                'repo': item['repo'],
                'docfile_name': item['docfile_name'],
                'doc_type': item['doc_type'],
                'intent': item['intent'],
                'license': item['license'],
                'path_to_docfile': item['path_to_docfile'],
                'relevant_code_files': item['relevant_code_files'],
                'relevant_code_dir': item['relevant_code_dir'],
                'target_text': item['target_text'],
                'relevant_code_context': item['relevant_code_context']
            }
            data_list.append(data_dict)
        
        print(f"📊 数据集基本信息:")
        print(f"   - 总样本数: {len(data_list)}")
        print(f"   - 示例字段: {list(data_list[0].keys())}")
        
        # 显示第一个样本的基本信息
        if data_list:
            sample = data_list[0]
            print(f"   - 示例样本: repo={sample['repo']}, intent={sample['intent'][:50]}...")
        
        return data_list
        
    except Exception as e:
        logger.error(f"加载本地数据集失败: {e}")
        print(f"❌ 加载本地数据集失败: {e}")
        # 返回空列表而不是抛出异常，这样评估可以优雅地处理
        return []


def load_fallback_dataset() -> List[Dict[str, Any]]:
    """
    加载备用数据集（用于测试）
    
    Returns:
        List[Dict]: 包含少量测试样本的字典列表
    """
    print("🔄 使用备用测试数据集")
    return [
        {
            'idx': 0,
            'repo': 'test_repo',
            'docfile_name': 'README.md',
            'doc_type': 'README',
            'intent': 'Provide documentation for a Python utility module',
            'license': 'MIT',
            'path_to_docfile': 'README.md',
            'relevant_code_files': ['utils.py'],
            'relevant_code_dir': 'src/',
            'target_text': 'This is a test documentation for the utility module.',
            'relevant_code_context': '''def hello_world():
    """A simple hello world function."""
    return "Hello, World!"

def add_numbers(a, b):
    """Add two numbers together."""
    return a + b'''
        }
    ]


def validate_dataset_format(dataset: List[Dict[str, Any]]) -> bool:
    """
    验证数据集格式是否正确
    
    Args:
        dataset: 要验证的数据集
        
    Returns:
        bool: 如果格式正确返回True
    """
    if not dataset:
        print("⚠️ 数据集为空")
        return False
    
    required_fields = [
        'idx', 'repo', 'docfile_name', 'doc_type', 'intent', 
        'license', 'path_to_docfile', 'relevant_code_files',
        'relevant_code_dir', 'target_text', 'relevant_code_context'
    ]
    
    sample = dataset[0]
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        print(f"⚠️ 数据集格式验证失败，缺少字段: {missing_fields}")
        return False
    
    print("✅ 数据集格式验证通过")
    return True


def get_sample_data(num_samples: int = 5) -> List[Dict[str, Any]]:
    """
    获取指定数量的样本数据用于测试
    
    Args:
        num_samples: 要获取的样本数量
        
    Returns:
        List[Dict]: 样本数据列表
    """
    full_dataset = load_lca_module_summarization_dataset()
    if not full_dataset:
        return load_fallback_dataset()
    
    return full_dataset[:num_samples]


if __name__ == "__main__":
    # 测试数据加载
    print("🧪 测试模块摘要数据加载...")
    dataset = load_lca_module_summarization_dataset()
    if dataset:
        validate_dataset_format(dataset)
        print(f"数据集加载成功，包含 {len(dataset)} 个样本")
    else:
        print("数据集加载失败，使用备用数据集")
        backup_dataset = load_fallback_dataset()
        print(f"备用数据集包含 {len(backup_dataset)} 个样本")
