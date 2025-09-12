"""
Long Code Arena Bug Localization Task

This task evaluates models on identifying files that need to be modified to fix bugs.
It uses the bug localization dataset from Long Code Arena benchmark.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task

# Import custom modules
from . import metrics
from . import data_loader
from .config import BugLocalizationConfig


@register_task("lca_bug_localization")
class LCABugLocalizationTask(Task):
    """Long Code Arena Bug Localization Task"""
    
    VERSION = 0.1
    
    def __init__(self, config: Union[BugLocalizationConfig, Dict[str, Any], None] = None):
        """
        初始化Bug Localization任务
        
        Args:
            config: 配置对象、配置字典或None（使用默认配置）
        """
        # 处理配置
        if config is None:
            self._task_config = BugLocalizationConfig()
        elif isinstance(config, dict):
            self._task_config = BugLocalizationConfig(config)
        elif isinstance(config, BugLocalizationConfig):
            self._task_config = config
        else:
            raise ValueError(f"不支持的配置类型: {type(config)}")
        
        # 初始化数据加载器
        self.data_loader = data_loader.BugLocalizationDataLoader(self._task_config)
    
    @property
    def config(self) -> BugLocalizationConfig:
        """获取任务配置"""
        return self._task_config
        
    def has_training_docs(self) -> bool:
        return self.config.get('split') == 'train'

    def has_validation_docs(self) -> bool:
        return self.config.get('split') == 'dev'

    def has_test_docs(self) -> bool:
        return self.config.get('split') == 'test'
        
    def training_docs(self):
        """Load training documents"""
        if self.config.get('split') == 'train':
            return self.data_loader.load_data('train')
        return []
        
    def validation_docs(self):
        """Load validation documents"""
        if self.config.get('split') == 'dev':
            return self.data_loader.load_data('dev')
        return []
        
    def test_docs(self):
        """Load test documents"""
        if self.config.get('split') == 'test':
            return self.data_loader.load_data('test')
        return []
    
    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to input text for the model"""
        # 组合issue信息
        issue_text = f"{doc['issue_title']}\n{doc['issue_body']}"
        
        # 获取仓库文件列表（按相关性排序）
        repo_files = doc.get('repo_files', [])
        max_files = self.config.get('max_context_files', 50)
        if len(repo_files) > max_files:
            repo_files = repo_files[:max_files]
        
        filepath_list = "\n".join(repo_files)
        
        # 构建用户提示（system prompt会在construct_requests中处理）
        user_prompt = f"""GitHub repo name:
{doc['repo_owner']}/{doc['repo_name']}
Issue description:
{issue_text}
File paths from the repo:
{filepath_list}"""
        
        return user_prompt
    
    def doc_to_target(self, doc: Dict[str, Any]) -> List[str]:
        """Extract the target files that should be modified"""
        changed_files = doc.get('changed_files', [])
        if isinstance(changed_files, str):
            # 如果是字符串格式，尝试解析为列表
            try:
                import ast
                changed_files = ast.literal_eval(changed_files)
            except:
                try:
                    changed_files = json.loads(changed_files)
                except:
                    changed_files = [changed_files]
        return changed_files if isinstance(changed_files, list) else [changed_files]
    
    def construct_requests(
        self, doc: Dict[str, Any], ctx: str, **kwargs
    ) -> List[Instance]:
        """Construct requests for generation"""
        # 为chat模型构建消息格式
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant specialized in software bug localization. "
                          "Your task is to identify the most likely files to be modified to fix a given bug. "
                          "Provide the output in JSON format with the list of file paths under the key \"files\". "
                          "Provide JSON ONLY without any additional comments."
            },
            {
                "role": "user", 
                "content": ctx
            }
        ]
        
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(messages, {
                    "until": ["\n\n"],  # 移除 "```" 以避免截断JSON输出
                    "max_gen_toks": 1024, 
                    "temperature": 0.1,  # 较低温度以获得更稳定的JSON输出
                    "do_sample": False
                }),
                idx=0,
                metadata={
                    "text_id": doc['text_id'],
                    "repo_owner": doc['repo_owner'],
                    "repo_name": doc['repo_name']
                }
            )
        ]
    
    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        """Process the model's generated results"""
        pred_text = results[0].strip() if results else ""
        target_files = self.doc_to_target(doc)
        
        # 尝试解析JSON输出
        predicted_files = []
        try:
            # 清理输出中可能的多余文本
            pred_text = pred_text.strip()
            if pred_text.startswith('```json'):
                pred_text = pred_text[7:]
            if pred_text.endswith('```'):
                pred_text = pred_text[:-3]
            
            pred_json = json.loads(pred_text)
            predicted_files = pred_json.get('files', [])
            if not isinstance(predicted_files, list):
                predicted_files = []
        except json.JSONDecodeError as e:
            print(f"JSON解析错误 for {doc['text_id']}: {e}")
            print(f"模型输出: {pred_text}")
            predicted_files = []
        
        # 获取所有仓库文件用于计算指标
        all_files = doc.get('repo_files', [])
        
        # 计算质量指标
        quality_metrics = metrics.compute_quality_metrics(
            all_files=all_files,
            expected_files=target_files,
            predicted_files=predicted_files
        )
        
        result = {
            "text_id": doc['text_id'],
            "repo_owner": doc['repo_owner'],
            "repo_name": doc['repo_name'],
            "prediction_text": pred_text,
            "predicted_files": predicted_files,
            "target_files": target_files,
            "all_files_count": len(all_files),
            "target_files_count": len(target_files),
            "predicted_files_count": len(predicted_files)
        }
        
        # 添加质量指标
        result.update(quality_metrics)
        
        return result
    
    def aggregation(self) -> Dict[str, Any]:
        """Define how to aggregate results across all examples"""
        return {
            "bug_loc_precision": ["exact_match", "mean"],
            "bug_loc_recall": ["exact_match", "mean"],
            "bug_loc_f1": ["exact_match", "mean"],
            "bug_loc_fpr": ["exact_match", "mean"],
            "bug_loc_all_correct": ["exact_match", "mean"],
            "bug_loc_at_least_one_correct": ["exact_match", "mean"],
            "bug_loc_all_incorrect": ["exact_match", "mean"],
            "bug_loc_target_files_rate": ["exact_match", "mean"]
        }
    
    def higher_is_better(self) -> Dict[str, bool]:
        """Define which metrics should be maximized"""
        return {
            "bug_loc_precision": True,
            "bug_loc_recall": True,
            "bug_loc_f1": True,
            "bug_loc_fpr": False,  # Lower is better for false positive rate
            "bug_loc_all_correct": True,
            "bug_loc_at_least_one_correct": True,
            "bug_loc_all_incorrect": False,  # Lower is better
            "bug_loc_target_files_rate": True
        }


@register_task("lca_bug_localization_py")
class LCABugLocalizationPyTask(LCABugLocalizationTask):
    """Bug Localization task for Python projects"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['language'] = 'py'
        super().__init__(config)


@register_task("lca_bug_localization_java")
class LCABugLocalizationJavaTask(LCABugLocalizationTask):
    """Bug Localization task for Java projects"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['language'] = 'java'
        super().__init__(config)


@register_task("lca_bug_localization_kt")
class LCABugLocalizationKtTask(LCABugLocalizationTask):
    """Bug Localization task for Kotlin projects"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['language'] = 'kt'
        super().__init__(config)
