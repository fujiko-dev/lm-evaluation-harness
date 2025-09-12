"""
LCA Module Summarization 任务实现

用于评估模型在模块摘要生成任务上的表现
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task

try:
    from .config import ModuleSummarizationConfig
    from .data_loader import load_lca_module_summarization_dataset
    from .metrics import ModuleSummarizationEvaluator, evaluate_predictions_batch
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from lm_eval.tasks.long_code_arena.module_summarization.config import ModuleSummarizationConfig
    from lm_eval.tasks.long_code_arena.module_summarization.data_loader import load_lca_module_summarization_dataset
    from lm_eval.tasks.long_code_arena.module_summarization.metrics import ModuleSummarizationEvaluator, evaluate_predictions_batch

logger = logging.getLogger(__name__)


@register_task("lca_module_summarization")
class LCAModuleSummarizationTask(Task):
    """
    LCA Module Summarization 任务实现
    评估模型在模块摘要生成任务上的表现
    """
    
    VERSION = 0.1
    
    def __init__(self, config: Union[ModuleSummarizationConfig, Dict[str, Any], None] = None):
        """
        初始化Module Summarization任务
        
        Args:
            config: 配置对象、配置字典或None（使用默认配置）
        """
        # 处理配置
        if config is None:
            self._task_config = ModuleSummarizationConfig()
        elif isinstance(config, dict):
            self._task_config = ModuleSummarizationConfig(config)
        elif isinstance(config, ModuleSummarizationConfig):
            self._task_config = config
        else:
            raise ValueError(f"不支持的配置类型: {type(config)}")
        
        # 初始化数据集
        self._dataset = None
        self._evaluator = ModuleSummarizationEvaluator()
    
    @property
    def config(self) -> ModuleSummarizationConfig:
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
            self._dataset = load_lca_module_summarization_dataset(load_config)
            
            # 应用样本限制
            limit = self.config.get('limit')
            if limit:
                self._dataset = self._dataset[:limit]
                
        return self._dataset
    
    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to input text for the model"""
        
        # 构建模块摘要提示
        prompt_parts = []
        
        # 添加系统说明
        prompt_parts.append("You are an expert software engineer and technical writer.")
        prompt_parts.append("Your task is to write a clear, concise, and informative summary for a software module.")
        prompt_parts.append("")
        
        # 添加仓库信息
        if 'repo' in doc:
            prompt_parts.append(f"Repository: {doc['repo']}")
        
        # 添加模块信息
        if 'docfile_name' in doc:
            prompt_parts.append(f"Module file: {doc['docfile_name']}")
        
        if 'doc_type' in doc:
            prompt_parts.append(f"Documentation type: {doc['doc_type']}")
        
        if 'intent' in doc:
            prompt_parts.append(f"Module purpose: {doc['intent']}")
        
        # 添加相关代码文件信息（如果配置中启用）
        if self.config.get('include_metadata', True) and 'relevant_code_files' in doc:
            relevant_files = doc['relevant_code_files']
            if relevant_files:
                prompt_parts.append("\nRelevant code files:")
                # 限制文件数量以控制上下文长度
                max_files = min(5, len(relevant_files))
                for file_info in relevant_files[:max_files]:
                    if isinstance(file_info, dict):
                        file_path = file_info.get('path', 'unknown')
                        file_content = file_info.get('content', '')[:1000]  # 限制内容长度
                        prompt_parts.append(f"File: {file_path}")
                        prompt_parts.append(f"Content snippet:\n{file_content}")
                        prompt_parts.append("")
        
        # 添加相关代码目录信息
        if 'relevant_code_dir' in doc and doc['relevant_code_dir']:
            code_dir = doc['relevant_code_dir'][:2000]  # 限制长度
            prompt_parts.append(f"Code directory structure:\n{code_dir}")
            prompt_parts.append("")
        
        prompt_parts.append("Please write a comprehensive module summary that explains:")
        prompt_parts.append("1. What this module does")
        prompt_parts.append("2. Key components and their functions")
        prompt_parts.append("3. How to use this module")
        prompt_parts.append("4. Any important notes or considerations")
        prompt_parts.append("")
        prompt_parts.append("Write the summary in clear, technical English suitable for developers.")
        
        return "\n".join(prompt_parts)
    
    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        """Extract the target summary from the document"""
        return doc.get('target_text', '')
    
    def construct_requests(self, doc: Dict[str, Any], ctx: str, **kwargs) -> List[Instance]:
        """Construct requests for generation"""
        # 为chat模型构建消息格式
        messages = [
            {
                "role": "system",
                "content": "You are an expert software engineer and technical writer specialized in creating clear, comprehensive module documentation."
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
                    "max_gen_toks": self.config.get('max_summary_length', 1000),
                    "temperature": self.config.get('temperature', 0.1),
                    "do_sample": False
                }),
                idx=0,
                metadata=("lca_module_summarization", doc.get('idx', 0), 1)
            )
        ]
    
    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        """Process the model's generated results"""
        pred_text = results[0].strip() if results else ""
        target_summary = self.doc_to_target(doc)
        
        # 使用评估器计算指标
        evaluation_result = self._evaluator.evaluate_single(
            prediction=pred_text,
            reference=target_summary,
            doc=doc
        )
        
        result = {
            "idx": doc.get('idx', ''),
            "repo": doc.get('repo', ''),
            "docfile_name": doc.get('docfile_name', ''),
            "prediction_text": pred_text,
            "target_summary": target_summary,
            
            # 从评估结果中提取指标
            "module_summary_bleu": evaluation_result.get('bleu_score', 0.0),
            "module_summary_rouge_l": evaluation_result.get('rouge_l_score', 0.0),
            "module_summary_bert_score": evaluation_result.get('bert_score', 0.0),
            "module_summary_semantic_sim": evaluation_result.get('semantic_similarity', 0.0),
            "module_summary_length_ratio": evaluation_result.get('length_ratio', 0.0),
            "module_summary_coverage": evaluation_result.get('coverage_score', 0.0)
        }
        
        return result
    
    def aggregation(self) -> Dict[str, Any]:
        """Define how to aggregate results across all examples"""
        return {
            "module_summary_bleu": ["module_summary_bleu", "mean"],
            "module_summary_rouge_l": ["module_summary_rouge_l", "mean"],
            "module_summary_bert_score": ["module_summary_bert_score", "mean"],
            "module_summary_semantic_sim": ["module_summary_semantic_sim", "mean"],
            "module_summary_length_ratio": ["module_summary_length_ratio", "mean"],
            "module_summary_coverage": ["module_summary_coverage", "mean"]
        }
    
    def higher_is_better(self) -> Dict[str, bool]:
        """Define which metrics should be maximized"""
        return {
            "module_summary_bleu": True,
            "module_summary_rouge_l": True,
            "module_summary_bert_score": True,
            "module_summary_semantic_sim": True,
            "module_summary_length_ratio": False,  # 接近1.0更好
            "module_summary_coverage": True
        }


@register_task("lca_module_summarization_verbose")
class LCAModuleSummarizationVerboseTask(LCAModuleSummarizationTask):
    """Module Summarization任务的详细版本，包含更多调试信息"""
    
    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        """Process results with additional verbose information"""
        result = super().process_results(doc, results)
        
        # 添加详细信息
        result.update({
            "verbose_info": {
                "doc_type": doc.get('doc_type', ''),
                "intent": doc.get('intent', ''),
                "license": doc.get('license', ''),
                "path_to_docfile": doc.get('path_to_docfile', ''),
                "prediction_length": len(result["prediction_text"]),
                "target_length": len(result["target_summary"])
            }
        })
        
        return result