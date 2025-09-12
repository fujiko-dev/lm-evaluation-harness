"""
Data loader for LCA CI Builds Repair dataset from local files
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
    from .config import CIBuildsRepairConfig
except ImportError:
    from lm_eval.tasks.long_code_arena.ci_builds_repair.config import CIBuildsRepairConfig


def load_lca_ci_builds_repair_dataset(config: Union[CIBuildsRepairConfig, Dict[str, Any], str, None] = None) -> List[Dict[str, Any]]:
    """
    加载本地LCA CI Builds Repair数据集
    
    Args:
        config: 配置对象、配置字典、路径字符串或None（使用默认配置）
        
    Returns:
        List[Dict]: 包含任务数据的字典列表
    """
    # 处理配置
    if config is None:
        config = CIBuildsRepairConfig()
    elif isinstance(config, dict):
        config = CIBuildsRepairConfig(config)
    elif isinstance(config, str):
        # 如果是字符串，当作数据集路径处理
        config = CIBuildsRepairConfig({'dataset_path': config})
    elif not isinstance(config, CIBuildsRepairConfig):
        raise ValueError(f"不支持的配置类型: {type(config)}")
    
    dataset_path = config.dataset_path
    split = config.get('split', 'test')
    
    try:
        print(f"🔄 正在从本地路径加载CI Builds Repair数据集: {dataset_path}")
        
        # 检查本地数据集是否存在
        if not dataset_path.exists():
            raise FileNotFoundError(f"本地数据集路径不存在: {dataset_path}")
        
        # 直接读取parquet文件
        parquet_path = dataset_path / f"data/python/{split}-00000-of-00001.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {parquet_path}")
        
        # 使用pandas读取parquet文件
        df = pd.read_parquet(parquet_path)
        
        print(f"✅ 成功加载数据集，包含 {len(df)} 个样本")
        
        # 转换为字典列表格式
        data_list = []
        for i, row in df.iterrows():
            # 处理logs字段（它是一个包含字典的数组）
            logs_processed = []
            if 'logs' in row and row['logs'] is not None:
                for log_entry in row['logs']:
                    logs_processed.append({
                        'step_name': log_entry.get('step_name', ''),
                        'log': log_entry.get('log', '')
                    })
            
            # 处理changed_files字段
            changed_files = []
            if 'changed_files' in row and row['changed_files'] is not None:
                changed_files = list(row['changed_files'])
            
            data_dict = {
                'idx': i,
                'id': row['id'],
                'language': row['language'],
                'repo_owner': row['repo_owner'],
                'repo_name': row['repo_name'],
                'head_branch': row['head_branch'],
                'workflow_name': row['workflow_name'],
                'workflow_filename': row['workflow_filename'],
                'workflow_path': row['workflow_path'],
                'contributor': row['contributor'],
                'sha_fail': row['sha_fail'],
                'sha_success': row['sha_success'],
                'workflow': row['workflow'],
                'logs': logs_processed,
                'diff': row['diff'],
                'difficulty': row['difficulty'],
                'changed_files': changed_files,
                'commit_link': row['commit_link'],
                'commit_date': row['commit_date']
            }
            data_list.append(data_dict)
        
        return data_list
        
    except Exception as e:
        logger.error(f"加载数据集时出错: {str(e)}")
        raise


def validate_dataset_format(data_list: List[Dict[str, Any]]) -> bool:
    """
    验证数据集格式是否正确
    
    Args:
        data_list: 数据集列表
        
    Returns:
        bool: 格式是否正确
    """
    required_fields = [
        'id', 'language', 'repo_owner', 'repo_name', 'workflow_name',
        'sha_fail', 'sha_success', 'workflow', 'logs', 'diff', 'difficulty'
    ]
    
    if not data_list:
        logger.error("数据集为空")
        return False
    
    for i, item in enumerate(data_list):
        for field in required_fields:
            if field not in item:
                logger.error(f"第 {i} 个样本缺少必需字段: {field}")
                return False
    
    logger.info(f"数据集格式验证通过，包含 {len(data_list)} 个样本")
    return True


def create_ci_repair_prompt(datapoint: Dict[str, Any]) -> str:
    """
    为CI修复任务创建提示词
    
    Args:
        datapoint: 单个数据点
        
    Returns:
        str: 格式化的提示词
    """
    # 构建失败日志信息
    logs_text = ""
    for log_entry in datapoint['logs']:
        logs_text += f"Step: {log_entry['step_name']}\n"
        logs_text += f"Log:\n{log_entry['log']}\n\n"
    
    # 构建工作流信息
    workflow_info = f"Workflow Name: {datapoint['workflow_name']}\n"
    workflow_info += f"Workflow File: {datapoint['workflow_path']}\n"
    workflow_info += f"Repository: {datapoint['repo_owner']}/{datapoint['repo_name']}\n"
    workflow_info += f"Failed Commit: {datapoint['sha_fail']}\n"
    workflow_info += f"Success Commit: {datapoint['sha_success']}\n\n"
    
    # 工作流内容
    workflow_content = f"Workflow Definition:\n```yaml\n{datapoint['workflow']}\n```\n\n"
    
    # 失败日志
    logs_section = f"Failed CI Logs:\n{logs_text}\n"
    
    # 构建最终提示词
    prompt = f"""You are tasked with fixing a failing GitHub Actions CI workflow. Below is the information about the failed workflow:

{workflow_info}

{workflow_content}

{logs_section}

Based on the workflow definition and the failure logs, please provide the necessary changes to fix the CI workflow. Focus on identifying the root cause of the failure and provide specific file modifications needed.

Your response should include:
1. Analysis of the failure cause
2. Specific changes needed to fix the issue
3. The modified code/configuration

Please provide your solution:"""

    return prompt


def get_target_diff(datapoint: Dict[str, Any]) -> str:
    """
    获取目标diff作为参考答案
    
    Args:
        datapoint: 单个数据点
        
    Returns:
        str: diff内容
    """
    return datapoint.get('diff', '')


def extract_file_changes_from_diff(diff_content: str) -> Dict[str, str]:
    """
    从diff内容中提取文件变更
    
    Args:
        diff_content: diff文本
        
    Returns:
        Dict[str, str]: 文件路径到变更内容的映射
    """
    changes = {}
    current_file = None
    current_content = []
    
    for line in diff_content.split('\n'):
        if line.startswith('diff --git'):
            # 保存之前文件的内容
            if current_file and current_content:
                changes[current_file] = '\n'.join(current_content)
            
            # 解析新文件路径
            parts = line.split()
            if len(parts) >= 4:
                # 从 "a/path/file.py" 中提取 "path/file.py"
                current_file = parts[3][2:] if parts[3].startswith('b/') else parts[3]
                current_content = []
        elif current_file:
            current_content.append(line)
    
    # 保存最后一个文件的内容
    if current_file and current_content:
        changes[current_file] = '\n'.join(current_content)
    
    return changes


def prepare_evaluation_data(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    准备评估数据，为每个样本添加提示词和参考答案
    
    Args:
        data_list: 原始数据列表
        
    Returns:
        List[Dict[str, Any]]: 包含提示词和参考答案的数据列表
    """
    evaluation_data = []
    
    for datapoint in data_list:
        prompt = create_ci_repair_prompt(datapoint)
        target_diff = get_target_diff(datapoint)
        file_changes = extract_file_changes_from_diff(target_diff)
        
        eval_item = {
            **datapoint,  # 保留原始数据
            'prompt': prompt,
            'target_diff': target_diff,
            'file_changes': file_changes
        }
        evaluation_data.append(eval_item)
    
    return evaluation_data


def get_dataset_statistics(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    获取数据集统计信息
    
    Args:
        data_list: 数据集列表
        
    Returns:
        Dict[str, Any]: 统计信息
    """
    if not data_list:
        return {}
    
    # 难度分布
    difficulty_counts = {}
    for item in data_list:
        difficulty = item.get('difficulty', 'unknown')
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    # 语言分布
    language_counts = {}
    for item in data_list:
        language = item.get('language', 'unknown')
        language_counts[language] = language_counts.get(language, 0) + 1
    
    # 仓库分布
    repo_counts = {}
    for item in data_list:
        repo = f"{item.get('repo_owner', '')}/{item.get('repo_name', '')}"
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
    
    stats = {
        'total_samples': len(data_list),
        'difficulty_distribution': difficulty_counts,
        'language_distribution': language_counts,
        'repository_distribution': repo_counts,
        'avg_changed_files': sum(len(item.get('changed_files', [])) for item in data_list) / len(data_list)
    }
    
    return stats
