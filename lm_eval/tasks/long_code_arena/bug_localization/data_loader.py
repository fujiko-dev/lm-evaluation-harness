"""
Bug Localization Data Loader

This module handles loading and processing of bug localization data including:
- Reading parquet files
- Loading repository content
- Extracting file lists from repositories
"""

import json
import os
import zipfile
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import hashlib
import tempfile

from .config import BugLocalizationConfig


class BugLocalizationDataLoader:
    """Data loader for bug localization task"""
    
    def __init__(self, config: Union[BugLocalizationConfig, Dict[str, Any], None] = None, 
                 dataset_path: Optional[str] = None, 
                 repos_path: Optional[str] = None, 
                 language: str = 'py'):
        """
        初始化数据加载器
        
        Args:
            config: 配置对象或配置字典
            dataset_path: 数据集路径（向后兼容）
            repos_path: 仓库路径（向后兼容）
            language: 编程语言
        """
        # 处理配置
        if config is None:
            from .config import BugLocalizationConfig
            config = BugLocalizationConfig()
        elif isinstance(config, dict):
            from .config import BugLocalizationConfig
            config = BugLocalizationConfig(config)
        
        self.config = config
        
        # 向后兼容：如果提供了参数，则覆盖配置
        if dataset_path is not None:
            self.config.set('dataset_path', dataset_path)
        if repos_path is not None:
            self.config.set('repos_path', repos_path)
        if language != 'py':
            self.config.set('language', language)
        
        self.dataset_path = Path(self.config.get('dataset_path'))
        self.repos_path = Path(self.config.get('repos_path'))
        self.language = self.config.get('language')
        self.supported_languages = {'py', 'java', 'kt'}
        
        if self.language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {self.language}. Supported: {self.supported_languages}")
        
        # 临时目录用于解压仓库
        self.temp_repos_dir = tempfile.mkdtemp(prefix=f"bug_loc_repos_{self.language}_")
        
    def load_data(self, split: str = 'test') -> List[Dict[str, Any]]:
        """Load data for the specified split"""
        parquet_file = self.dataset_path / f"{self.language}/{split}-00000-of-00001.parquet"
        
        if not parquet_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {parquet_file}")
        
        print(f"加载 {self.language} 语言的 {split} 数据集...")
        df = pd.read_parquet(parquet_file)
        
        data = []
        for idx, row in df.iterrows():
            try:
                doc = self._process_row(row, idx)
                if doc:
                    data.append(doc)
                    if len(data) % 10 == 0:
                        print(f"已处理 {len(data)} 个样本...")
            except Exception as e:
                print(f"处理样本 {idx} 时出错: {e}")
                continue
        
        print(f"成功加载 {len(data)} 个 {self.language} 语言的样本")
        return data
    
    def _process_row(self, row: pd.Series, idx: int) -> Optional[Dict[str, Any]]:
        """Process a single row from the dataset"""
        # 基本信息
        doc = {
            'text_id': row['text_id'],
            'repo_owner': row['repo_owner'],
            'repo_name': row['repo_name'],
            'issue_title': row['issue_title'],
            'issue_body': row['issue_body'],
            'base_sha': row['base_sha'],
            'head_sha': row['head_sha'],
            'changed_files': self._parse_changed_files(row['changed_files']),
            'issue_url': row['issue_url'],
            'pull_url': row['pull_url']
        }
        
        # 获取仓库文件列表
        repo_files = self._get_repo_files(row['repo_owner'], row['repo_name'])
        if not repo_files:
            print(f"无法获取仓库文件列表: {row['repo_owner']}/{row['repo_name']}")
            return None
        
        # 按相关性排序文件
        sorted_files = self._sort_files_by_relevance(
            repo_files, 
            f"{row['issue_title']}\n{row['issue_body']}"
        )
        
        doc['repo_files'] = sorted_files
        doc['total_repo_files'] = len(repo_files)
        
        return doc
    
    def _parse_changed_files(self, changed_files) -> List[str]:
        """Parse changed files from various formats"""
        if isinstance(changed_files, str):
            try:
                # 首先尝试直接使用 ast.literal_eval 解析
                import ast
                parsed = ast.literal_eval(changed_files)
                if isinstance(parsed, list):
                    return parsed
                else:
                    return [str(parsed)]
            except:
                try:
                    # 如果失败，尝试JSON解析
                    parsed = json.loads(changed_files)
                    if isinstance(parsed, list):
                        return parsed
                    else:
                        return [changed_files]
                except:
                    # 如果都失败，当作单个文件处理
                    return [changed_files]
        elif isinstance(changed_files, list):
            return changed_files
        else:
            return []
    
    def _get_repo_files(self, repo_owner: str, repo_name: str) -> List[str]:
        """Get list of files from repository"""
        repo_identifier = f"{repo_owner}__{repo_name}"
        repo_zip_path = self.repos_path / f"{self.language}/{repo_identifier}.zip"
        
        if not repo_zip_path.exists():
            print(f"仓库ZIP文件不存在: {repo_zip_path}")
            return []
        
        # 解压到临时目录
        temp_repo_path = Path(self.temp_repos_dir) / repo_identifier
        if not temp_repo_path.exists():
            try:
                with zipfile.ZipFile(repo_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_repo_path)
            except Exception as e:
                print(f"解压仓库失败 {repo_zip_path}: {e}")
                return []
        
        # 收集指定语言的文件
        files = []
        extensions = self._get_file_extensions()
        
        for root, dirs, filenames in os.walk(temp_repo_path):
            # 跳过测试目录
            dirs[:] = [d for d in dirs if not self._is_test_directory(d)]
            
            for filename in filenames:
                if any(filename.endswith(ext) for ext in extensions):
                    # 获取相对于仓库根目录的路径
                    file_path = Path(root) / filename
                    relative_path = file_path.relative_to(temp_repo_path)
                    
                    # 跳过测试文件
                    if not self._is_test_file(str(relative_path)):
                        files.append(str(relative_path))
        
        return sorted(files)
    
    def _get_file_extensions(self) -> List[str]:
        """Get file extensions for the current language"""
        extension_map = {
            'py': ['.py'],
            'java': ['.java'],
            'kt': ['.kt']
        }
        return extension_map.get(self.language, [])
    
    def _is_test_directory(self, dirname: str) -> bool:
        """Check if directory is a test directory"""
        test_patterns = ['test', 'tests', '__pycache__', '.git', '.idea', 'build', 'target']
        dirname_lower = dirname.lower()
        return any(pattern in dirname_lower for pattern in test_patterns)
    
    def _is_test_file(self, filepath: str) -> bool:
        """Check if file is a test file"""
        filepath_lower = filepath.lower()
        test_patterns = ['test_', '_test', 'test.', '/test/', '/tests/', '__pycache__']
        return any(pattern in filepath_lower for pattern in test_patterns)
    
    def _sort_files_by_relevance(self, files: List[str], issue_text: str) -> List[str]:
        """Sort files by relevance to the issue"""
        # 简单的相关性排序算法
        issue_words = set(issue_text.lower().split())
        
        def calculate_relevance(filepath: str) -> float:
            score = 0.0
            filepath_lower = filepath.lower()
            
            # 文件名匹配
            filename = os.path.basename(filepath_lower)
            for word in issue_words:
                if len(word) > 3 and word in filename:
                    score += 2.0
                elif len(word) > 3 and word in filepath_lower:
                    score += 1.0
            
            # 路径深度（较浅的文件可能更重要）
            depth = filepath.count('/')
            score -= depth * 0.1
            
            # 特定文件类型的权重
            if 'main' in filename or 'index' in filename:
                score += 1.0
            if 'util' in filename or 'helper' in filename:
                score -= 0.5
            
            return score
        
        # 按相关性排序
        scored_files = [(f, calculate_relevance(f)) for f in files]
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        return [f for f, _ in scored_files]
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_repos_dir):
            shutil.rmtree(self.temp_repos_dir)
    
    def __del__(self):
        """Destructor to cleanup temporary files"""
        self.cleanup()


def get_repo_content_on_commit(repo_path: str, commit_sha: str, 
                              extensions: List[str] = None, 
                              ignore_tests: bool = True) -> Dict[str, str]:
    """Get repository content at specific commit (placeholder function)"""
    # This is a simplified version - in real implementation would use git
    # For now, just return empty dict as we're using the zip files directly
    return {}


def get_changed_files_between_commits(repo_path: str, base_sha: str, head_sha: str,
                                    extensions: List[str] = None,
                                    ignore_tests: bool = True) -> List[str]:
    """Get changed files between commits (placeholder function)"""
    # This is a simplified version - in real implementation would use git
    # For now, just return empty list as we're using the changed_files from dataset
    return []
