"""
Data loader for LCA Commit Message Generation dataset from local files
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
    from .config import CommitMessageGenerationConfig
except ImportError:
    from lm_eval.tasks.long_code_arena.commit_message_generation.config import CommitMessageGenerationConfig


def load_lca_commit_message_generation_dataset(config: Union[CommitMessageGenerationConfig, Dict[str, Any], str, None] = None) -> List[Dict[str, Any]]:
    """
    加载本地LCA Commit Message Generation数据集
    
    Args:
        config: 配置对象、配置字典、路径字符串或None（使用默认配置）
        
    Returns:
        List[Dict]: 包含任务数据的字典列表
    """
    # 处理配置
    if config is None:
        config = CommitMessageGenerationConfig()
    elif isinstance(config, dict):
        config = CommitMessageGenerationConfig(config)
    elif isinstance(config, str):
        # 如果是字符串，当作数据集路径处理
        config = CommitMessageGenerationConfig({'dataset_path': config})
    elif not isinstance(config, CommitMessageGenerationConfig):
        raise ValueError(f"不支持的配置类型: {type(config)}")
    
    dataset_path = config.dataset_path
    split = config.get('split', 'test')
    
    try:
        print(f"🔄 正在从本地路径加载Commit Message Generation数据集: {dataset_path}")
        
        # 检查本地数据集是否存在
        if not dataset_path.exists():
            raise FileNotFoundError(f"本地数据集路径不存在: {dataset_path}")
        
        # 查找parquet文件
        parquet_files = list(dataset_path.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"在数据集路径中未找到parquet文件: {dataset_path}")
        
        # 选择第一个parquet文件（通常只有一个测试文件）
        parquet_file = parquet_files[0]
        print(f"📁 使用数据文件: {parquet_file}")
        
        # 使用pandas读取parquet文件
        df = pd.read_parquet(parquet_file)
        
        print(f"✅ 成功加载数据集，包含 {len(df)} 个样本")
        
        # 转换为字典列表格式
        data_list = []
        for i, row in df.iterrows():
            # 创建标准化的数据格式
            data_dict = {
                'idx': i,
                'repo': row.get('repo', ''),
                'commit_sha': row.get('commit_sha', ''),
                'message': row.get('message', ''),
                'diff': row.get('diff', ''),
                'mods': row.get('mods', []),  # 修改的文件列表
                'adds': row.get('adds', []),  # 添加的文件列表
                'dels': row.get('dels', []),  # 删除的文件列表
            }
            
            # 添加可选字段
            for optional_field in ['author', 'date', 'subject', 'body']:
                if optional_field in row:
                    data_dict[optional_field] = row[optional_field]
            
            data_list.append(data_dict)
        
        print(f"📊 数据集基本信息:")
        print(f"   - 总样本数: {len(data_list)}")
        print(f"   - 示例字段: {list(data_list[0].keys()) if data_list else []}")
        if data_list:
            print(f"   - 示例样本: repo={data_list[0].get('repo', 'N/A')}, message={data_list[0].get('message', 'N/A')[:50]}...")
        
        return data_list
        
    except FileNotFoundError as e:
        logger.error(f"数据集文件未找到: {e}")
        raise
    except Exception as e:
        logger.error(f"加载数据集时发生错误: {e}")
        raise


def validate_dataset_format(data: List[Dict[str, Any]]) -> bool:
    """
    验证数据集格式
    
    Args:
        data: 数据列表
        
    Returns:
        bool: 是否有效
    """
    if not data:
        logger.warning("数据集为空")
        return False
    
    required_fields = ['idx', 'repo', 'message', 'diff']
    
    for i, item in enumerate(data[:5]):  # 只检查前5个样本
        for field in required_fields:
            if field not in item:
                logger.error(f"样本 {i} 缺少必需字段: {field}")
                return False
    
    logger.info("数据集格式验证通过")
    return True


def preprocess_commit_data(data: List[Dict[str, Any]], config: CommitMessageGenerationConfig) -> List[Dict[str, Any]]:
    """
    预处理提交数据
    
    Args:
        data: 原始数据
        config: 配置对象
        
    Returns:
        预处理后的数据
    """
    processed_data = []
    max_context_length = config.get('max_context_length', 8000)
    
    for item in data:
        processed_item = item.copy()
        
        # 截断diff以适应上下文长度限制
        diff = item.get('diff', '')
        if len(diff) > max_context_length:
            diff = diff[:max_context_length] + "\n... (diff truncated due to length)"
            processed_item['diff'] = diff
            processed_item['diff_truncated'] = True
        else:
            processed_item['diff_truncated'] = False
        
        # 清理提交消息
        message = item.get('message', '').strip()
        processed_item['message'] = message
        
        # 计算统计信息
        processed_item['diff_length'] = len(item.get('diff', ''))
        processed_item['message_length'] = len(message)
        processed_item['num_files_modified'] = len(item.get('mods', []))
        processed_item['num_files_added'] = len(item.get('adds', []))
        processed_item['num_files_deleted'] = len(item.get('dels', []))
        
        processed_data.append(processed_item)
    
    return processed_data


def get_dataset_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    获取数据集统计信息
    
    Args:
        data: 数据列表
        
    Returns:
        统计信息字典
    """
    if not data:
        return {}
    
    stats = {
        'total_samples': len(data),
        'avg_diff_length': sum(len(item.get('diff', '')) for item in data) / len(data),
        'avg_message_length': sum(len(item.get('message', '')) for item in data) / len(data),
        'unique_repos': len(set(item.get('repo', '') for item in data)),
        'avg_files_modified': sum(len(item.get('mods', [])) for item in data) / len(data),
    }
    
    # 统计消息长度分布
    message_lengths = [len(item.get('message', '')) for item in data]
    stats['message_length_percentiles'] = {
        'p50': sorted(message_lengths)[len(message_lengths) // 2],
        'p90': sorted(message_lengths)[int(len(message_lengths) * 0.9)],
        'p95': sorted(message_lengths)[int(len(message_lengths) * 0.95)]
    }
    
    return stats
