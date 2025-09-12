"""
LCA CI Builds Repair 任务实现

用于评估模型在CI构建修复任务上的表现
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task

try:
    from .config import CIBuildsRepairConfig
    from .data_loader import load_lca_ci_builds_repair_dataset
    from .metrics import CIBuildsRepairEvaluator, evaluate_predictions_batch
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from lm_eval.tasks.long_code_arena.ci_builds_repair.config import CIBuildsRepairConfig
    from lm_eval.tasks.long_code_arena.ci_builds_repair.data_loader import load_lca_ci_builds_repair_dataset
    from lm_eval.tasks.long_code_arena.ci_builds_repair.metrics import CIBuildsRepairEvaluator, evaluate_predictions_batch

logger = logging.getLogger(__name__)


@register_task("lca_ci_builds_repair")
class LCACIBuildsRepairTask(Task):
    """
    LCA CI Builds Repair 任务实现
    评估模型在CI构建修复任务上的表现
    """
    
    VERSION = 0.1
    
    def __init__(self, config: Union[CIBuildsRepairConfig, Dict[str, Any], None] = None):
        """
        初始化CI Builds Repair任务
        
        Args:
            config: 配置对象、配置字典或None（使用默认配置）
        """
        # 处理配置
        if config is None:
            self._task_config = CIBuildsRepairConfig()
        elif isinstance(config, dict):
            self._task_config = CIBuildsRepairConfig(config)
        elif isinstance(config, CIBuildsRepairConfig):
            self._task_config = config
        else:
            raise ValueError(f"不支持的配置类型: {type(config)}")
        
        # 初始化数据集
        self._dataset = None
        self._evaluator = CIBuildsRepairEvaluator()
    
    @property
    def config(self) -> CIBuildsRepairConfig:
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
            self._dataset = load_lca_ci_builds_repair_dataset(load_config)
            
            # 应用样本限制
            limit = self.config.get('limit')
            if limit:
                self._dataset = self._dataset[:limit]
                
        return self._dataset
    
    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to input text for the model"""
        
        # 构建CI修复提示
        prompt_parts = []
        
        # 添加系统说明
        prompt_parts.append("You are an expert software engineer helping to fix CI build failures.")
        prompt_parts.append("Given the build logs and failure information, provide a fix for the failing build.")
        prompt_parts.append("")
        
        # 添加仓库信息
        if 'repo_name' in doc:
            prompt_parts.append(f"Repository: {doc['repo_name']}")
        
        # 添加构建失败信息
        if 'build_failure_reason' in doc:
            prompt_parts.append(f"Build failure reason: {doc['build_failure_reason']}")
        
        # 添加构建日志（如果配置中启用）
        if self.config.get('include_logs', True) and 'logs' in doc:
            prompt_parts.append("\nBuild logs:")
            for log_entry in doc['logs'][:5]:  # 限制日志条目数量
                step_name = log_entry.get('step_name', 'Unknown step')
                log_content = log_entry.get('log', '')[:1000]  # 限制日志长度
                prompt_parts.append(f"Step: {step_name}")
                prompt_parts.append(f"Log: {log_content}")
                prompt_parts.append("")
        
        # 添加相关文件（如果有）
        if 'related_files' in doc:
            prompt_parts.append("Related files:")
            for file_info in doc['related_files'][:3]:  # 限制文件数量
                if isinstance(file_info, dict):
                    filename = file_info.get('filename', 'unknown')
                    content = file_info.get('content', '')[:2000]  # 限制内容长度
                    prompt_parts.append(f"File: {filename}")
                    prompt_parts.append(f"Content:\n{content}")
                    prompt_parts.append("")
        
        prompt_parts.append("Please provide a fix for this CI build failure.")
        prompt_parts.append("Output your response in JSON format with the key 'fix'.")
        
        return "\n".join(prompt_parts)
    
    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        """Extract the target fix from the document"""
        return doc.get('target_fix', doc.get('fix', ''))
    
    def construct_requests(self, doc: Dict[str, Any], ctx: str, **kwargs) -> List[Instance]:
        """Construct requests for generation"""
        # 为chat模型构建消息格式
        messages = [
            {
                "role": "system",
                "content": "You are an expert software engineer specialized in fixing CI build failures. "
                          "Provide clear, actionable fixes in JSON format."
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
                    "max_gen_toks": self.config.get('max_tokens', 2048),
                    "temperature": self.config.get('temperature', 0.1),
                    "do_sample": False
                }),
                idx=0,
                metadata={
                    "text_id": doc.get('text_id', ''),
                    "repo_name": doc.get('repo_name', ''),
                    "build_failure_reason": doc.get('build_failure_reason', '')
                }
            )
        ]
    
    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        """Process the model's generated results"""
        pred_text = results[0].strip() if results else ""
        target_fix = self.doc_to_target(doc)
        
        # 尝试解析JSON输出
        predicted_fix = ""
        try:
            # 清理输出中可能的多余文本
            pred_text = pred_text.strip()
            if pred_text.startswith('```json'):
                pred_text = pred_text[7:]
            if pred_text.endswith('```'):
                pred_text = pred_text[:-3]
            
            pred_json = json.loads(pred_text)
            predicted_fix = pred_json.get('fix', pred_text)
        except json.JSONDecodeError:
            # 如果JSON解析失败，直接使用原始输出
            predicted_fix = pred_text
        
        # 使用评估器计算指标
        evaluation_result = self._evaluator.evaluate_single_prediction(
            prediction=predicted_fix,
            datapoint=doc
        )
        
        result = {
            "text_id": doc.get('text_id', ''),
            "repo_name": doc.get('repo_name', ''),
            "prediction_text": pred_text,
            "predicted_fix": predicted_fix,
            "target_fix": target_fix,
            
            # 从评估结果中提取指标
            "ci_repair_score": evaluation_result.score,
            "ci_repair_success_rate": 1.0 if evaluation_result.success else 0.0,
            "ci_repair_bleu": evaluation_result.details.get('similarity_score', 0.0),
            "ci_repair_rouge_l": evaluation_result.details.get('key_fixes_score', 0.0),
            "ci_repair_exact_match": 1.0 if evaluation_result.score >= 0.95 else 0.0
        }
        
        return result
    
    def aggregation(self) -> Dict[str, Any]:
        """Define how to aggregate results across all examples"""
        return {
            "ci_repair_success_rate": ["exact_match", "mean"],
            "ci_repair_score": ["exact_match", "mean"],
            "ci_repair_bleu": ["exact_match", "mean"],
            "ci_repair_rouge_l": ["exact_match", "mean"],
            "ci_repair_exact_match": ["exact_match", "mean"]
        }
    
    def higher_is_better(self) -> Dict[str, bool]:
        """Define which metrics should be maximized"""
        return {
            "ci_repair_success_rate": True,
            "ci_repair_score": True,
            "ci_repair_bleu": True,
            "ci_repair_rouge_l": True,
            "ci_repair_exact_match": True
        }