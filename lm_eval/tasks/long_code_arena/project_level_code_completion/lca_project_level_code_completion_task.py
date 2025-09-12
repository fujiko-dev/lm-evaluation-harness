"""
LCA Project-level Code Completion自定义任务实现
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datetime import datetime

from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task

try:
    from .data_loader import (
        load_project_level_code_completion_dataset,
        validate_dataset_format,
        prepare_completion_context,
        extract_completion_lines,
        create_line_completion_prompt,
        get_available_context_sizes
    )
    from .metrics import (
        ProjectLevelCodeCompletionEvaluator,
        evaluate_predictions_batch,
        DetailedResults
    )
    from .api_models import create_api_model, EXAMPLE_CONFIGS
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from lm_eval.tasks.long_code_arena.project_level_code_completion.data_loader import (
        load_project_level_code_completion_dataset,
        validate_dataset_format,
        prepare_completion_context,
        extract_completion_lines,
        create_line_completion_prompt,
        get_available_context_sizes
    )
    from lm_eval.tasks.long_code_arena.project_level_code_completion.metrics import (
        ProjectLevelCodeCompletionEvaluator,
        evaluate_predictions_batch,
        DetailedResults
    )
    from lm_eval.tasks.long_code_arena.project_level_code_completion.api_models import create_api_model, EXAMPLE_CONFIGS

logger = logging.getLogger(__name__)

# 结果保存路径
RESULTS_BASE_PATH = os.getenv(
    'LCA_PLCC_OUTPUT_PATH',
    '/Users/xiaoyunting/Documents/ModelDev/working_path/projects/lm-evaluation-harness/results/long_code_arena/project_level_code_completion'
)

# 数据集默认路径
DEFAULT_DATASET_PATH = os.getenv(
    'LCA_PLCC_DATASET_PATH',
    '/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/datasets/lca_project_level_code_completion'
)


@register_task("lca_project_level_code_completion")
class LCAProjectLevelCodeCompletionTask(Task):
    """
    LCA Project-level Code Completion任务实现
    集成官方评估方法，支持实时进度显示和结果保存
    """
    
    VERSION = 1.0
    
    def __init__(self, config=None):
        # 处理配置
        if config is None:
            config = {}
        elif hasattr(config, 'to_dict'):
            config = config.to_dict()
        
        # 不调用super().__init__()，避免HuggingFace数据集加载
        self._dataset = None
        self._evaluator = None
        self._results_dir = None
        self._processed_count = 0
        self._total_count = 0
        self._api_model = None
        
        # 配置参数
        self.dataset_path = config.get('dataset_path', DEFAULT_DATASET_PATH)
        self.context_size = config.get('context_size', 'small_context')
        self.max_context_chars = config.get('max_context_chars', 50000)
        self.max_prompt_chars = config.get('max_prompt_chars', 80000)
        self.max_completion_tokens = config.get('max_completion_tokens', 150)
        self.line_types_to_evaluate = config.get('line_types_to_evaluate', None)
        
        # API配置
        self.model_type = config.get('model_type', 'openai')
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.api_key = config.get('api_key', None)
        self.api_config = config.get('api_config', {})
        
        # 确保结果目录存在
        self._setup_results_directory()
        
        # 初始化评估器
        self._initialize_evaluator()
        
        # 初始化API模型
        self._initialize_api_model()
    
    def _setup_results_directory(self):
        """设置结果保存目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._results_dir = os.path.join(
            RESULTS_BASE_PATH,
            "project_level_code_completion",
            f"{self.model_name}_{self.context_size}_{timestamp}"
        )
        
        os.makedirs(self._results_dir, exist_ok=True)
        logger.info(f"结果将保存到: {self._results_dir}")
    
    def _initialize_evaluator(self):
        """初始化评估器"""
        self._evaluator = ProjectLevelCodeCompletionEvaluator(
            results_save_path=self._results_dir
        )
        logger.info("评估器初始化完成")
    
    def _initialize_api_model(self):
        """初始化API模型"""
        if not self.api_key:
            logger.warning("未提供API密钥，将无法进行模型推理")
            return
        
        try:
            self._api_model = create_api_model(
                model_type=self.model_type,
                model_name=self.model_name,
                api_key=self.api_key,
                **self.api_config
            )
            logger.info(f"API模型初始化完成: {self.model_name}")
        except Exception as e:
            logger.error(f"API模型初始化失败: {str(e)}")
            raise
    
    def has_training_docs(self) -> bool:
        return False
    
    def has_validation_docs(self) -> bool:
        return False
    
    def has_test_docs(self) -> bool:
        return True
    
    def test_docs(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        if self._dataset is None:
            logger.info(f"正在加载数据集: {self.dataset_path}")
            logger.info(f"上下文大小: {self.context_size}")
            
            # 检查数据集路径
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
            
            # 获取可用的上下文大小
            available_sizes = get_available_context_sizes(self.dataset_path)
            logger.info(f"可用的上下文大小: {available_sizes}")
            
            if self.context_size not in available_sizes:
                raise ValueError(f"不支持的上下文大小: {self.context_size}")
            
            # 加载数据集
            self._dataset = load_project_level_code_completion_dataset(
                dataset_path=self.dataset_path,
                context_size=self.context_size
            )
            
            self._total_count = len(self._dataset)
            logger.info(f"数据集加载完成，共 {self._total_count} 个样本")
        
        return self._dataset
    
    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """将文档转换为输入文本"""
        # 准备上下文
        repo_context, completion_info = prepare_completion_context(
            sample=doc,
            max_context_chars=self.max_context_chars
        )
        
        # 提取需要补全的行
        completion_lines = extract_completion_lines(
            completion_info=completion_info,
            line_types=self.line_types_to_evaluate
        )
        
        if not completion_lines:
            return "# No lines to complete"
        
        # 为了简化，我们选择第一行作为示例
        # 在实际评估中，我们会处理所有行
        line_num, line_content, line_type = completion_lines[0]
        
        # 获取文件前缀（到当前行之前的内容）
        file_lines = completion_info['content'].split('\n')
        file_prefix = '\n'.join(file_lines[:line_num])
        
        # 创建提示
        prompt = create_line_completion_prompt(
            repo_context=repo_context,
            completion_file_path=completion_info['filename'],
            file_prefix=file_prefix,
            max_prompt_chars=self.max_prompt_chars
        )
        
        return prompt
    
    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        """将文档转换为目标输出"""
        # 准备补全信息
        _, completion_info = prepare_completion_context(
            sample=doc,
            max_context_chars=self.max_context_chars
        )
        
        # 提取需要补全的行
        completion_lines = extract_completion_lines(
            completion_info=completion_info,
            line_types=self.line_types_to_evaluate
        )
        
        if not completion_lines:
            return ""
        
        # 返回第一行的内容作为目标
        line_num, line_content, line_type = completion_lines[0]
        return line_content.strip()
    
    def construct_requests(self, doc: Dict[str, Any], ctx: str, **kwargs) -> List[Instance]:
        """构造请求实例"""
        # 在这里，我们需要处理所有的完成行，而不仅仅是第一行
        # 准备上下文
        repo_context, completion_info = prepare_completion_context(
            sample=doc,
            max_context_chars=self.max_context_chars
        )
        
        # 提取需要补全的行
        completion_lines = extract_completion_lines(
            completion_info=completion_info,
            line_types=self.line_types_to_evaluate
        )
        
        instances = []
        file_lines = completion_info['content'].split('\n')
        
        for line_num, line_content, line_type in completion_lines:
            # 获取文件前缀
            file_prefix = '\n'.join(file_lines[:line_num])
            
            # 创建提示
            prompt = create_line_completion_prompt(
                repo_context=repo_context,
                completion_file_path=completion_info['filename'],
                file_prefix=file_prefix,
                max_prompt_chars=self.max_prompt_chars
            )
            
            # 创建实例
            instance = Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(prompt, {"max_tokens": self.max_completion_tokens}),
                idx=len(instances),
                metadata=(
                    "lca_project_level_code_completion",
                    doc.get('idx', len(instances)),
                    1
                )
            )
            instances.append(instance)
        
        return instances
    
    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        """处理结果"""
        # 获取元数据
        repo_context, completion_info = prepare_completion_context(
            sample=doc,
            max_context_chars=self.max_context_chars
        )
        
        completion_lines = extract_completion_lines(
            completion_info=completion_info,
            line_types=self.line_types_to_evaluate
        )
        
        # 处理每个结果
        processed_results = []
        for i, (result, (line_num, line_content, line_type)) in enumerate(zip(results, completion_lines)):
            # 提取第一行作为预测结果（代码补全通常只需要一行）
            predicted_line = result.split('\n')[0].strip() if result else ""
            
            # 评估单个预测
            metrics = self._evaluator.evaluate_single_prediction(
                predicted_line=predicted_line,
                target_line=line_content,
                line_type=line_type,
                context_info={
                    'repo': completion_info['repo'],
                    'filename': completion_info['filename'],
                    'line_num': line_num
                }
            )
            
            processed_results.append({
                'line_num': line_num,
                'line_type': line_type,
                'predicted': predicted_line,
                'target': line_content,
                'exact_match': metrics.exact_match,
                'edit_distance': metrics.edit_distance,
                'bleu_score': metrics.bleu_score
            })
        
        # 更新进度
        self._processed_count += 1
        if self._processed_count % 10 == 0 or self._processed_count == self._total_count:
            progress = (self._processed_count / self._total_count) * 100
            logger.info(f"进度: {self._processed_count}/{self._total_count} ({progress:.1f}%)")
        
        return {
            'sample_results': processed_results,
            'repo': completion_info['repo'],
            'filename': completion_info['filename']
        }
    
    def aggregation(self) -> Dict[str, Any]:
        """聚合所有结果"""
        logger.info("正在聚合评估结果...")
        
        # 获取聚合结果
        detailed_results = self._evaluator.aggregate_results()
        
        # 保存详细结果
        results_file = self._evaluator.save_results(
            results=detailed_results,
            model_name=self.model_name,
            context_size=self.context_size
        )
        
        # 打印摘要
        self._evaluator.print_summary(
            results=detailed_results,
            model_name=self.model_name,
            context_size=self.context_size
        )
        
        logger.info(f"评估完成，详细结果已保存到: {results_file}")
        
        # 返回主要指标
        return {
            'exact_match': detailed_results.overall.exact_match,
            'edit_distance': detailed_results.overall.edit_distance,
            'bleu_score': detailed_results.overall.bleu_score,
            'total_lines': detailed_results.overall.line_count,
            'results_file': results_file
        }
    
    def higher_is_better(self) -> Dict[str, bool]:
        """指定哪些指标越高越好"""
        return {
            'exact_match': True,
            'bleu_score': True,
            'edit_distance': False,  # 编辑距离越小越好
        }


@register_task("lca_project_level_code_completion_verbose")
class LCAProjectLevelCodeCompletionVerboseTask(LCAProjectLevelCodeCompletionTask):
    """
    详细版本的Project-level Code Completion任务
    提供更多的评估信息和调试输出
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        # 启用更详细的日志
        logging.getLogger().setLevel(logging.DEBUG)
    
    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        """处理结果（详细版本）"""
        processed_results = super().process_results(doc, results)
        
        # 添加详细的调试信息
        logger.debug(f"处理文档: {processed_results['repo']}/{processed_results['filename']}")
        for result in processed_results['sample_results']:
            logger.debug(f"  行 {result['line_num']} ({result['line_type']}):")
            logger.debug(f"    目标: {result['target']}")
            logger.debug(f"    预测: {result['predicted']}")
            logger.debug(f"    精确匹配: {result['exact_match']}")
            logger.debug(f"    编辑距离: {result['edit_distance']:.4f}")
            logger.debug(f"    BLEU: {result['bleu_score']:.4f}")
        
        return processed_results


# 便捷函数用于直接运行评估
def run_project_level_code_completion_evaluation(
    model_type: str,
    model_name: str,
    api_key: str,
    context_size: str = "small_context",
    dataset_path: str = DEFAULT_DATASET_PATH,
    line_types: Optional[List[str]] = None,
    **api_config
) -> DetailedResults:
    """
    直接运行Project-level Code Completion评估
    
    Args:
        model_type: 模型类型
        model_name: 模型名称
        api_key: API密钥
        context_size: 上下文大小
        dataset_path: 数据集路径
        line_types: 要评估的行类型
        **api_config: API配置参数
        
    Returns:
        详细的评估结果
    """
    logger.info("开始Project-level Code Completion评估")
    logger.info(f"模型: {model_name}")
    logger.info(f"上下文大小: {context_size}")
    
    # 初始化API模型
    api_model = create_api_model(
        model_type=model_type,
        model_name=model_name,
        api_key=api_key,
        **api_config
    )
    
    # 加载数据集
    dataset = load_project_level_code_completion_dataset(
        dataset_path=dataset_path,
        context_size=context_size
    )
    
    logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")
    
    # 初始化评估器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        RESULTS_BASE_PATH,
        "project_level_code_completion",
        f"{model_name}_{context_size}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)
    
    evaluator = ProjectLevelCodeCompletionEvaluator(
        results_save_path=results_dir
    )
    
    # 处理每个样本
    total_processed = 0
    
    for i, sample in enumerate(tqdm(dataset, desc="评估进度")):
        try:
            # 准备上下文
            repo_context, completion_info = prepare_completion_context(
                sample=sample,
                max_context_chars=50000
            )
            
            # 提取需要补全的行
            completion_lines = extract_completion_lines(
                completion_info=completion_info,
                line_types=line_types
            )
            
            # 处理每一行
            file_lines = completion_info['content'].split('\n')
            
            for line_num, line_content, line_type in completion_lines:
                # 创建提示
                file_prefix = '\n'.join(file_lines[:line_num])
                prompt = create_line_completion_prompt(
                    repo_context=repo_context,
                    completion_file_path=completion_info['filename'],
                    file_prefix=file_prefix,
                    max_prompt_chars=80000
                )
                
                # 生成预测
                try:
                    predicted_text = api_model.generate(prompt, max_tokens=150)
                    predicted_line = predicted_text.split('\n')[0].strip() if predicted_text else ""
                except Exception as e:
                    logger.error(f"生成失败: {str(e)}")
                    predicted_line = ""
                
                # 评估
                evaluator.evaluate_single_prediction(
                    predicted_line=predicted_line,
                    target_line=line_content,
                    line_type=line_type,
                    context_info={
                        'repo': completion_info['repo'],
                        'filename': completion_info['filename'],
                        'line_num': line_num
                    }
                )
                
                total_processed += 1
        
        except Exception as e:
            logger.error(f"处理样本 {i} 失败: {str(e)}")
    
    # 获取结果
    results = evaluator.aggregate_results()
    
    # 保存结果
    results_file = evaluator.save_results(
        results=results,
        model_name=model_name,
        context_size=context_size
    )
    
    # 打印摘要
    evaluator.print_summary(
        results=results,
        model_name=model_name,
        context_size=context_size
    )
    
    logger.info(f"评估完成，总共处理了 {total_processed} 个代码行")
    logger.info(f"详细结果已保存到: {results_file}")
    
    return results
