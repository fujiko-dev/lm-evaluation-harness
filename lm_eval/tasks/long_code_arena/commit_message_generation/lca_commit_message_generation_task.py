"""
LCA Commit Message Generation 任务实现

用于评估模型在生成提交消息任务上的表现
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task

try:
    from .config import CommitMessageGenerationConfig
    from .data_loader import load_lca_commit_message_generation_dataset, preprocess_commit_data
    from .metrics import compute_bleu_score, compute_rouge_scores, compute_bertscore
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from lm_eval.tasks.long_code_arena.commit_message_generation.config import CommitMessageGenerationConfig
    from lm_eval.tasks.long_code_arena.commit_message_generation.data_loader import load_lca_commit_message_generation_dataset, preprocess_commit_data
    from lm_eval.tasks.long_code_arena.commit_message_generation.metrics import compute_bleu_score, compute_rouge_scores, compute_bertscore

logger = logging.getLogger(__name__)


@register_task("lca_commit_message_generation")
class LCACommitMessageGenerationTask(Task):
    """
    LCA Commit Message Generation 任务实现
    评估模型在生成提交消息任务上的表现
    """
    
    VERSION = 0.1
    
    def __init__(self, config: Union[CommitMessageGenerationConfig, Dict[str, Any], None] = None):
        """
        初始化Commit Message Generation任务
        
        Args:
            config: 配置对象、配置字典或None（使用默认配置）
        """
        # 处理配置
        if config is None:
            self._task_config = CommitMessageGenerationConfig()
        elif isinstance(config, dict):
            self._task_config = CommitMessageGenerationConfig(config)
        elif isinstance(config, CommitMessageGenerationConfig):
            self._task_config = config
        else:
            raise ValueError(f"不支持的配置类型: {type(config)}")
        
        # 初始化数据集
        self._dataset = None
    
    @property
    def config(self) -> CommitMessageGenerationConfig:
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
            return self._load_data('train')
        return []
        
    def validation_docs(self):
        """Load validation documents"""
        if self.config.get('split') == 'dev':
            return self._load_data('dev')
        return []
        
    def test_docs(self):
        """Load test documents"""
        if self.config.get('split') == 'test':
            return self._load_data('test')
        return []
    
    def _load_data(self, split: str):
        """加载指定拆分的数据"""
        if self._dataset is None:
            # 创建临时配置用于加载数据
            load_config = self.config.to_dict()
            load_config['split'] = split
            raw_data = load_lca_commit_message_generation_dataset(load_config)
            
            # 预处理数据
            self._dataset = preprocess_commit_data(raw_data, self.config)
            
            # 应用样本限制
            limit = self.config.get('limit')
            if limit:
                self._dataset = self._dataset[:limit]
                
        return self._dataset
    
    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to input text for the model"""
        
        # 构建提交消息生成提示
        prompt_parts = []
        
        # 添加系统说明
        prompt_parts.append("You are an expert software developer tasked with writing a concise and informative commit message.")
        prompt_parts.append("Based on the following code changes (diff), generate an appropriate commit message.")
        prompt_parts.append("")
        
        # 添加仓库信息
        if 'repo' in doc:
            prompt_parts.append(f"Repository: {doc['repo']}")
        
        # 添加文件变更统计
        stats = []
        if doc.get('num_files_modified', 0) > 0:
            stats.append(f"{doc['num_files_modified']} modified")
        if doc.get('num_files_added', 0) > 0:
            stats.append(f"{doc['num_files_added']} added")
        if doc.get('num_files_deleted', 0) > 0:
            stats.append(f"{doc['num_files_deleted']} deleted")
        
        if stats:
            prompt_parts.append(f"Files changed: {', '.join(stats)}")
        
        # 添加代码差异（如果配置中启用）
        if self.config.get('include_diff', True) and 'diff' in doc:
            diff = doc['diff']
            if diff:
                prompt_parts.append("\nCode changes (diff):")
                prompt_parts.append("```diff")
                prompt_parts.append(diff)
                prompt_parts.append("```")
        
        prompt_parts.append("")
        prompt_parts.append("Generate a concise commit message that:")
        prompt_parts.append("1. Summarizes what was changed")
        prompt_parts.append("2. Uses imperative mood (e.g., 'Add feature' not 'Added feature')")
        prompt_parts.append("3. Is under 72 characters for the first line")
        prompt_parts.append("4. Provides clear context about the change")
        prompt_parts.append("")
        prompt_parts.append("Commit message:")
        
        return "\n".join(prompt_parts)
    
    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        """Extract the target commit message from the document"""
        return doc.get('message', '').strip()
    
    def construct_requests(self, doc: Dict[str, Any], ctx: str, **kwargs) -> List[Instance]:
        """Construct requests for generation"""
        # 为chat模型构建消息格式
        messages = [
            {
                "role": "system",
                "content": "You are an expert software developer specialized in writing clear, concise commit messages that accurately describe code changes."
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
                    "until": ["\n\n"],
                    "max_gen_toks": self.config.get('max_commit_length', 200),
                    "temperature": self.config.get('temperature', 0.1),
                    "do_sample": False
                }),
                idx=0,
                metadata=("lca_commit_message_generation", doc.get('idx', 0), 1)
            )
        ]
    
    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        """Process the model's generated results"""
        pred_text = results[0].strip() if results else ""
        target_message = self.doc_to_target(doc)
        
        # 清理生成的提交消息（移除可能的前缀）
        pred_message = pred_text
        if pred_message.lower().startswith("commit message:"):
            pred_message = pred_message[15:].strip()
        elif pred_message.lower().startswith("message:"):
            pred_message = pred_message[8:].strip()
        
        # 只取第一行作为提交消息（符合Git惯例）
        pred_message = pred_message.split('\n')[0].strip()
        
        # 计算评估指标
        bleu_score = compute_bleu_score([pred_message], [target_message]) if target_message else 0.0
        rouge_scores = compute_rouge_scores([pred_message], [target_message]) if target_message else {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        # BERT Score需要特殊处理
        try:
            bert_score_val = compute_bertscore([pred_message], [target_message]) if target_message else 0.0
        except Exception as e:
            logger.warning(f"BERT Score计算失败: {e}")
            bert_score_val = 0.0
        
        result = {
            "idx": doc.get('idx', ''),
            "repo": doc.get('repo', ''),
            "commit_sha": doc.get('commit_sha', ''),
            "prediction_text": pred_text,
            "predicted_message": pred_message,
            "target_message": target_message,
            
            # 评估指标
            "commit_bleu": bleu_score,
            "commit_rouge1": rouge_scores.get('rouge1', 0.0),
            "commit_rouge2": rouge_scores.get('rouge2', 0.0),
            "commit_rougeL": rouge_scores.get('rougeL', 0.0),
            "commit_bert_f1": bert_score_val,
            
            # 长度指标
            "predicted_length": len(pred_message),
            "target_length": len(target_message),
            "length_ratio": len(pred_message) / max(len(target_message), 1)
        }
        
        return result
    
    def aggregation(self) -> Dict[str, Any]:
        """Define how to aggregate results across all examples"""
        return {
            "commit_bleu": ["commit_bleu", "mean"],
            "commit_rouge1": ["commit_rouge1", "mean"],
            "commit_rouge2": ["commit_rouge2", "mean"],
            "commit_rougeL": ["commit_rougeL", "mean"],
            "commit_bert_f1": ["commit_bert_f1", "mean"],
            "length_ratio": ["length_ratio", "mean"]
        }
    
    def higher_is_better(self) -> Dict[str, bool]:
        """Define which metrics should be maximized"""
        return {
            "commit_bleu": True,
            "commit_rouge1": True,
            "commit_rouge2": True,
            "commit_rougeL": True,
            "commit_bert_f1": True,
            "length_ratio": False  # 接近1.0更好
        }


@register_task("lca_commit_message_generation_verbose")
class LCACommitMessageGenerationVerboseTask(LCACommitMessageGenerationTask):
    """Commit Message Generation任务的详细版本，包含更多调试信息"""
    
    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        """Process results with additional verbose information"""
        result = super().process_results(doc, results)
        
        # 添加详细信息
        result.update({
            "verbose_info": {
                "diff_length": doc.get('diff_length', 0),
                "diff_truncated": doc.get('diff_truncated', False),
                "num_files_total": doc.get('num_files_modified', 0) + doc.get('num_files_added', 0) + doc.get('num_files_deleted', 0),
                "commit_sha": doc.get('commit_sha', ''),
                "author": doc.get('author', ''),
                "date": doc.get('date', '')
            }
        })
        
        return result