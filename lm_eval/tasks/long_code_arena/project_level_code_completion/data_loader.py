"""
Project-level Code Completion数据加载器
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def load_project_level_code_completion_dataset(
    dataset_path: str,
    context_size: str = "small_context"
) -> List[Dict[str, Any]]:
    """
    加载Project-level Code Completion数据集
    
    Args:
        dataset_path: 数据集路径
        context_size: 上下文大小，可选值：small_context, medium_context, large_context, huge_context
        
    Returns:
        包含数据样本的列表
    """
    try:
        # 构建parquet文件路径
        context_dir = os.path.join(dataset_path, "data", context_size)
        
        if not os.path.exists(context_dir):
            raise FileNotFoundError(f"数据集目录不存在: {context_dir}")
        
        # 获取所有parquet文件
        parquet_files = [f for f in os.listdir(context_dir) if f.endswith('.parquet')]
        parquet_files.sort()  # 确保文件顺序一致
        
        if not parquet_files:
            raise FileNotFoundError(f"在 {context_dir} 中未找到parquet文件")
        
        logger.info(f"发现 {len(parquet_files)} 个parquet文件，正在加载...")
        
        # 加载所有parquet文件
        all_data = []
        for parquet_file in parquet_files:
            file_path = os.path.join(context_dir, parquet_file)
            logger.info(f"正在加载: {parquet_file}")
            
            df = pd.read_parquet(file_path)
            # 将DataFrame转换为字典列表
            data_batch = df.to_dict('records')
            all_data.extend(data_batch)
            
            logger.info(f"从 {parquet_file} 加载了 {len(data_batch)} 个样本")
        
        logger.info(f"总共加载了 {len(all_data)} 个数据样本")
        
        # 验证数据格式
        if all_data:
            validate_dataset_format(all_data[0])
            
        return all_data
        
    except Exception as e:
        logger.error(f"加载数据集时出错: {str(e)}")
        raise


def validate_dataset_format(sample: Dict[str, Any]) -> bool:
    """
    验证数据样本格式是否正确
    
    Args:
        sample: 数据样本
        
    Returns:
        验证是否通过
    """
    required_fields = [
        'repo', 'commit_hash', 'completion_file', 
        'completion_lines', 'repo_snapshot'
    ]
    
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"数据样本缺少必需字段: {field}")
    
    # 验证completion_file结构
    completion_file = sample['completion_file']
    if not isinstance(completion_file, dict):
        raise ValueError("completion_file应该是字典格式")
    
    if 'filename' not in completion_file or 'content' not in completion_file:
        raise ValueError("completion_file缺少filename或content字段")
    
    # 验证completion_lines结构
    completion_lines = sample['completion_lines']
    if not isinstance(completion_lines, dict):
        raise ValueError("completion_lines应该是字典格式")
    
    # 注意：数据中使用的是 'commited' 而不是 'committed'
    expected_line_types = ['commited', 'inproject', 'infile', 'common', 'non_informative', 'random']
    for line_type in expected_line_types:
        if line_type not in completion_lines:
            logger.warning(f"completion_lines缺少 {line_type} 字段")
    
    # 验证repo_snapshot结构 - 实际上是字典格式，包含filename和content数组
    repo_snapshot = sample['repo_snapshot']
    if not isinstance(repo_snapshot, dict):
        raise ValueError("repo_snapshot应该是字典格式")
    
    if 'filename' not in repo_snapshot or 'content' not in repo_snapshot:
        raise ValueError("repo_snapshot缺少filename或content字段")
    
    # 验证filename和content数组长度一致
    if len(repo_snapshot['filename']) != len(repo_snapshot['content']):
        raise ValueError("repo_snapshot中filename和content数组长度不一致")
    
    logger.debug("数据样本格式验证通过")
    return True


def prepare_completion_context(
    sample: Dict[str, Any],
    max_context_chars: int = 50000
) -> Tuple[str, Dict[str, Any]]:
    """
    为代码补全任务准备上下文
    
    Args:
        sample: 数据样本
        max_context_chars: 最大上下文字符数
        
    Returns:
        (上下文字符串, 补全信息)
    """
    # 构建仓库上下文
    repo_context_parts = []
    repo_snapshot = sample['repo_snapshot']
    
    # 获取文件名和内容数组
    filenames = repo_snapshot['filename']
    contents = repo_snapshot['content']
    
    # 创建文件列表并按文件名排序
    files = list(zip(filenames, contents))
    files.sort(key=lambda x: x[0])
    
    current_chars = 0
    for filename, content in files:
        # 跳过非Python文件（可以根据需要调整）
        if not filename.endswith('.py'):
            continue
            
        file_block = f"# File: {filename}\n{content}\n\n"
        
        # 检查是否会超过字符限制
        if current_chars + len(file_block) > max_context_chars:
            break
            
        repo_context_parts.append(file_block)
        current_chars += len(file_block)
    
    repo_context = "".join(repo_context_parts)
    
    # 获取补全文件信息
    completion_file = sample['completion_file']
    completion_info = {
        'filename': completion_file['filename'],
        'content': completion_file['content'],
        'completion_lines': sample['completion_lines'],
        'repo': sample['repo'],
        'commit_hash': sample['commit_hash']
    }
    
    return repo_context, completion_info


def extract_completion_lines(
    completion_info: Dict[str, Any],
    line_types: Optional[List[str]] = None
) -> List[Tuple[int, str, str]]:
    """
    提取需要补全的代码行
    
    Args:
        completion_info: 补全信息
        line_types: 要提取的行类型，如果为None则提取所有类型
        
    Returns:
        (行号, 行内容, 行类型) 的列表
    """
    if line_types is None:
        line_types = ['commited', 'inproject', 'infile', 'common', 'non_informative', 'random']
    
    completion_lines = completion_info['completion_lines']
    file_content = completion_info['content']
    file_lines = file_content.split('\n')
    
    extracted_lines = []
    
    for line_type in line_types:
        if line_type in completion_lines:
            line_numbers = completion_lines[line_type]
            for line_num in line_numbers:
                # 确保行号在有效范围内
                if 0 <= line_num < len(file_lines):
                    line_content = file_lines[line_num]
                    extracted_lines.append((line_num, line_content, line_type))
    
    # 按行号排序
    extracted_lines.sort(key=lambda x: x[0])
    
    return extracted_lines


def get_available_context_sizes(dataset_path: str) -> List[str]:
    """
    获取数据集中可用的上下文大小
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        可用的上下文大小列表
    """
    data_dir = os.path.join(dataset_path, "data")
    if not os.path.exists(data_dir):
        return []
    
    context_sizes = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.endswith('_context'):
            context_sizes.append(item)
    
    return sorted(context_sizes)


def create_line_completion_prompt(
    repo_context: str,
    completion_file_path: str,
    file_prefix: str,
    max_prompt_chars: int = 80000
) -> str:
    """
    创建行级代码补全的提示
    
    Args:
        repo_context: 仓库上下文
        completion_file_path: 补全文件路径
        file_prefix: 文件前缀（到当前行之前的内容）
        max_prompt_chars: 最大提示字符数
        
    Returns:
        格式化的提示字符串
    """
    # 基础提示模板
    prompt_template = """You are a code completion assistant. Complete the next line of code based on the repository context and current file content.

Repository Context:
{repo_context}

File: {completion_file_path}

Current code (complete the next line):
{file_prefix}

Complete the next line of code. Provide ONLY the single line of code without any explanation, comments, or markdown formatting:"""

    # 如果上下文太长，需要截断
    total_estimated = len(prompt_template) + len(repo_context) + len(completion_file_path) + len(file_prefix)
    
    if total_estimated > max_prompt_chars:
        # 优先保留文件前缀，然后尽可能多地保留仓库上下文
        available_for_context = max_prompt_chars - len(prompt_template) - len(completion_file_path) - len(file_prefix) - 100
        if available_for_context > 0:
            repo_context = repo_context[:available_for_context] + "\n# ... (context truncated) ..."
        else:
            repo_context = "# (context too large, truncated)"
    
    prompt = prompt_template.format(
        repo_context=repo_context,
        completion_file_path=completion_file_path,
        file_prefix=file_prefix
    )
    
    return prompt
