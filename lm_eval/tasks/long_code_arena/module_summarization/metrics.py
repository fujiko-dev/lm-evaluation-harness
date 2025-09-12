"""
评估工具函数，集成LCA官方评估方法
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# 本地评估函数路径
EVAL_FUNC_PATH = os.getenv(
    'LCA_MS_EVAL_FUNC_PATH',
    '/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/eval_func/module_summarization'
)

def setup_evaluation_environment():
    """
    设置评估环境，导入LCA官方评估函数
    """
    try:
        # 将评估函数路径添加到Python路径
        if EVAL_FUNC_PATH not in sys.path:
            sys.path.append(EVAL_FUNC_PATH)
        
        # 导入官方评估组件
        from utils.scorer import OptionsScoringModel
        from utils.context_utils import trim_context
        from utils.files_utils import load_config
        
        print(f"✅ 成功导入LCA官方评估组件从: {EVAL_FUNC_PATH}")
        return {
            'OptionsScoringModel': OptionsScoringModel,
            'trim_context': trim_context,
            'load_config': load_config
        }
    except ImportError as e:
        logger.error(f"导入LCA评估组件失败: {e}")
        print(f"❌ 导入LCA评估组件失败: {e}")
        return None
    except Exception as e:
        logger.error(f"设置评估环境失败: {e}")
        print(f"❌ 设置评估环境失败: {e}")
        return None


class ModuleSummarizationEvaluator:
    """
    模块摘要评估器，使用LCA官方评估方法
    """
    
    def __init__(self, scorer_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 device: str = "cpu", max_context_tokens: int = 6000):
        """
        初始化评估器
        
        Args:
            scorer_model_name: 用于评估的模型名称
            device: 计算设备
            max_context_tokens: 最大上下文token数
        """
        self.scorer_model_name = scorer_model_name
        self.device = device
        self.max_context_tokens = max_context_tokens
        self.scorer = None
        self.tokenizer = None
        self.eval_components = None
        
        # 设置评估环境
        self.eval_components = setup_evaluation_environment()
        if self.eval_components is None:
            print("⚠️ 警告: 无法加载LCA官方评估组件，将使用简化评估方法")
            return
        
        try:
            # 初始化评估模型
            print(f"🔄 初始化评估模型: {scorer_model_name}")
            OptionsScoringModel = self.eval_components['OptionsScoringModel']
            self.scorer = OptionsScoringModel(scorer_model_name, device)
            
            # 初始化tokenizer用于上下文裁剪
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf"  # 用于tokenize的默认模型
            )
            
            print("✅ 评估器初始化成功")
            
        except Exception as e:
            logger.error(f"初始化评估器失败: {e}")
            print(f"❌ 初始化评估器失败: {e}")
            self.scorer = None
    
    def evaluate_single(self, prediction: str, reference: str, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个预测结果
        
        Args:
            prediction: 模型生成的摘要
            reference: 参考摘要
            doc: 文档信息
            
        Returns:
            包含评估指标的字典
        """
        try:
            # 简化的评估指标
            result = {
                'bleu_score': self._calculate_bleu(prediction, reference),
                'rouge_l_score': self._calculate_rouge_l(prediction, reference),
                'bert_score': self._calculate_bert_score(prediction, reference),
                'semantic_similarity': self._calculate_semantic_similarity(prediction, reference),
                'length_ratio': len(prediction) / max(len(reference), 1),
                'coverage_score': self._calculate_coverage(prediction, reference)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {
                'bleu_score': 0.0,
                'rouge_l_score': 0.0,
                'bert_score': 0.0,
                'semantic_similarity': 0.0,
                'length_ratio': 1.0,
                'coverage_score': 0.0
            }
    
    def _calculate_bleu(self, prediction: str, reference: str) -> float:
        """计算BLEU分数（简化版本）"""
        try:
            # 简单的n-gram重叠计算
            pred_words = prediction.lower().split()
            ref_words = reference.lower().split()
            
            if not pred_words or not ref_words:
                return 0.0
            
            # 计算unigram重叠
            common_words = set(pred_words) & set(ref_words)
            precision = len(common_words) / len(pred_words) if pred_words else 0
            recall = len(common_words) / len(ref_words) if ref_words else 0
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * precision * recall / (precision + recall)
            
        except Exception:
            return 0.0
    
    def _calculate_rouge_l(self, prediction: str, reference: str) -> float:
        """计算ROUGE-L分数（简化版本）"""
        try:
            pred_words = prediction.lower().split()
            ref_words = reference.lower().split()
            
            if not pred_words or not ref_words:
                return 0.0
            
            # 简单的最长公共子序列近似
            common_words = set(pred_words) & set(ref_words)
            lcs_length = len(common_words)
            
            precision = lcs_length / len(pred_words) if pred_words else 0
            recall = lcs_length / len(ref_words) if ref_words else 0
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * precision * recall / (precision + recall)
            
        except Exception:
            return 0.0
    
    def _calculate_bert_score(self, prediction: str, reference: str) -> float:
        """计算BERT分数（简化版本）"""
        try:
            # 简单的基于词汇重叠的近似
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            
            if not pred_words or not ref_words:
                return 0.0
            
            intersection = len(pred_words & ref_words)
            union = len(pred_words | ref_words)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_semantic_similarity(self, prediction: str, reference: str) -> float:
        """计算语义相似度（简化版本）"""
        try:
            # 基于词汇重叠的语义相似度近似
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            
            if not pred_words or not ref_words:
                return 0.0
            
            intersection = len(pred_words & ref_words)
            union = len(pred_words | ref_words)
            
            # Jaccard相似度
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_coverage(self, prediction: str, reference: str) -> float:
        """计算内容覆盖度"""
        try:
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            
            if not ref_words:
                return 1.0 if not pred_words else 0.0
            
            covered_words = len(pred_words & ref_words)
            return covered_words / len(ref_words)
            
        except Exception:
            return 0.0
    
    def get_comparison_metric(self, intent: str, code_context: str, 
                            gold_doc: str, pred_doc: str) -> float:
        """
        使用LCA官方方法计算比较指标
        
        Args:
            intent: 文档意图描述
            code_context: 代码上下文
            gold_doc: 金标准文档
            pred_doc: 预测文档
            
        Returns:
            float: 比较得分 (0-1)
        """
        if self.scorer is None:
            # 使用简化评估方法
            return self._simple_comparison_metric(gold_doc, pred_doc)
        
        try:
            # 使用LCA官方评估方法
            return self._lca_comparison_metric(intent, code_context, gold_doc, pred_doc)
        except Exception as e:
            logger.error(f"LCA评估方法失败: {e}")
            print(f"⚠️ LCA评估方法失败，使用简化方法: {e}")
            return self._simple_comparison_metric(gold_doc, pred_doc)
    
    def _lca_comparison_metric(self, intent: str, code_context: str, 
                              gold_doc: str, pred_doc: str) -> float:
        """
        LCA官方比较评估方法
        """
        # 准备第一个prompt (gold作为A，pred作为B)
        prompt1 = f'I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n'
        prompt1 += f'My code:\n\n{code_context}\n\n\n\n'
        prompt1 += f'Documentation A:\n\n{gold_doc}\n\n\n\n'
        prompt1 += f'Documentation B:\n\n{pred_doc}\n\n\n\n'
        prompt1 += 'Better documentation is documentation '
        
        options = ["A", "B"]
        unnorm_logprobs1 = self.scorer.score_options(prompt1, options)
        norm_probs1 = torch.exp(torch.log_softmax(unnorm_logprobs1, dim=0))
        
        # 准备第二个prompt (pred作为A，gold作为B)
        prompt2 = f'I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n'
        prompt2 += f'My code:\n\n{code_context}\n\n\n\n'
        prompt2 += f'Documentation A:\n\n{pred_doc}\n\n\n\n'
        prompt2 += f'Documentation B:\n\n{gold_doc}\n\n\n\n'
        prompt2 += 'Better documentation is documentation '
        
        unnorm_logprobs2 = self.scorer.score_options(prompt2, options)
        norm_probs2 = torch.exp(torch.log_softmax(unnorm_logprobs2, dim=0))
        
        # 计算预测文档更好的概率
        # norm_probs1[1] = P(B better | gold=A, pred=B)
        # norm_probs2[0] = P(A better | pred=A, gold=B) 
        p_better = (norm_probs1[1] + norm_probs2[0]) / 2
        
        return float(p_better)
    
    def _simple_comparison_metric(self, gold_doc: str, pred_doc: str) -> float:
        """
        简化比较评估方法（备用）
        """
        if not pred_doc.strip():
            return 0.0
        
        # 基于长度和单词重叠的简单评估
        gold_words = set(gold_doc.lower().split())
        pred_words = set(pred_doc.lower().split())
        
        if not gold_words:
            return 1.0 if not pred_words else 0.5
        
        # 计算Jaccard相似度
        intersection = len(gold_words & pred_words)
        union = len(gold_words | pred_words)
        jaccard = intersection / union if union > 0 else 0.0
        
        # 考虑长度因子
        len_ratio = min(len(pred_doc), len(gold_doc)) / max(len(pred_doc), len(gold_doc), 1)
        
        return (jaccard + len_ratio) / 2
    
    def trim_context_if_needed(self, context: str) -> str:
        """
        如果需要，裁剪上下文长度
        """
        if self.tokenizer is None:
            return context
        
        try:
            trim_context = self.eval_components['trim_context']
            return trim_context(context, self.tokenizer, self.max_context_tokens)
        except Exception as e:
            logger.error(f"上下文裁剪失败: {e}")
            # 简单截断
            return context[:self.max_context_tokens * 4]  # 粗略估计
    
    def evaluate_single_prediction(self, pred_doc: str, gold_doc: str, 
                                 intent: str, code_context: str) -> Dict[str, float]:
        """
        评估单个预测
        
        Args:
            pred_doc: 预测的文档
            gold_doc: 金标准文档
            intent: 文档意图
            code_context: 代码上下文
            
        Returns:
            Dict[str, float]: 评估结果
        """
        # 裁剪上下文
        trimmed_context = self.trim_context_if_needed(code_context)
        
        # 计算比较指标
        comparison_score = self.get_comparison_metric(
            intent, trimmed_context, gold_doc, pred_doc
        )
        
        # 计算其他基础指标
        basic_metrics = self._calculate_basic_metrics(pred_doc, gold_doc)
        
        result = {
            'comparison_score': comparison_score,
            **basic_metrics
        }
        
        return result
    
    def _calculate_basic_metrics(self, pred_doc: str, gold_doc: str) -> Dict[str, float]:
        """
        计算基础评估指标
        """
        result = {}
        
        # 1. 文档长度指标
        pred_len = len(pred_doc.strip())
        gold_len = len(gold_doc.strip())
        
        result['length_ratio'] = min(pred_len, gold_len) / max(pred_len, gold_len, 1)
        result['has_content'] = 1.0 if pred_len > 0 else 0.0
        
        # 2. 词汇重叠指标
        pred_words = set(pred_doc.lower().split())
        gold_words = set(gold_doc.lower().split())
        
        if gold_words:
            result['word_overlap'] = len(pred_words & gold_words) / len(gold_words)
        else:
            result['word_overlap'] = 1.0 if not pred_words else 0.0
        
        # 3. BLEU类似指标（简化版）
        if pred_words and gold_words:
            result['jaccard_similarity'] = len(pred_words & gold_words) / len(pred_words | gold_words)
        else:
            result['jaccard_similarity'] = 0.0
        
        return result


def evaluate_predictions_batch(predictions: List[str], references: List[str],
                             intents: List[str], code_contexts: List[str],
                             evaluator: Optional[ModuleSummarizationEvaluator] = None) -> Dict[str, float]:
    """
    批量评估预测结果
    
    Args:
        predictions: 预测文档列表
        references: 参考文档列表
        intents: 意图描述列表
        code_contexts: 代码上下文列表
        evaluator: 评估器实例
        
    Returns:
        Dict[str, float]: 聚合的评估结果
    """
    if evaluator is None:
        evaluator = ModuleSummarizationEvaluator()
    
    if not all(len(lst) == len(predictions) for lst in [references, intents, code_contexts]):
        raise ValueError("所有输入列表的长度必须相同")
    
    all_results = []
    total_samples = len(predictions)
    
    print(f"\n📊 开始批量评估 {total_samples} 个模块摘要样本...")
    print("=" * 60)
    
    for i, (pred, ref, intent, context) in enumerate(zip(predictions, references, intents, code_contexts)):
        sample_num = i + 1
        print(f"\n🔍 评估样本 {sample_num}/{total_samples}")
        print("-" * 40)
        print(f"📝 意图: {intent[:50]}...")
        print(f"🤖 预测长度: {len(pred)} 字符")
        print(f"📚 参考长度: {len(ref)} 字符")
        
        try:
            result = evaluator.evaluate_single_prediction(pred, ref, intent, context)
            all_results.append(result)
            
            # 显示这个样本的评估结果
            comp_score = result.get('comparison_score', 0.0)
            word_overlap = result.get('word_overlap', 0.0)
            print(f"✅ 比较得分: {comp_score:.3f}")
            print(f"   词汇重叠: {word_overlap:.3f}")
            
        except Exception as e:
            logger.error(f"评估样本{sample_num}失败: {e}")
            print(f"❌ 评估失败: {e}")
            # 使用默认值
            all_results.append({
                'comparison_score': 0.0,
                'length_ratio': 0.0,
                'has_content': 0.0,
                'word_overlap': 0.0,
                'jaccard_similarity': 0.0
            })
        
        print("-" * 40)
    
    # 聚合结果
    if not all_results:
        return {}
    
    aggregated = {}
    for key in all_results[0].keys():
        values = [result[key] for result in all_results if key in result]
        if values:
            aggregated[f'mean_{key}'] = np.mean(values)
            aggregated[f'std_{key}'] = np.std(values)
    
    print(f"\n🎯 批量评估完成!")
    print(f"📈 主要指标:")
    print(f"   - 平均比较得分: {aggregated.get('mean_comparison_score', 0.0):.3f}")
    print(f"   - 平均词汇重叠: {aggregated.get('mean_word_overlap', 0.0):.3f}")
    print(f"   - 有效内容比例: {aggregated.get('mean_has_content', 0.0):.3f}")
    print("=" * 60)
    
    return aggregated


# lm-eval框架兼容的评估函数
def module_summarization_metric(items: List[Any]) -> float:
    """
    Module summarization评估指标，用于lm-eval框架
    
    Args:
        items: 包含[reference, prediction, context, intent]的列表
        
    Returns:
        float: 评估得分
    """
    if len(items) < 2:
        print("⚠️ Module Summarization评估: 输入格式错误")
        return 0.0
    
    # 解析输入
    reference = items[0] if len(items) > 0 else ""
    prediction = items[1] if len(items) > 1 else ""
    
    # 对于框架兼容性，我们可能无法获得所有必要的上下文信息
    # 在这种情况下使用简化评估
    if len(items) >= 4:
        context = items[2]
        intent = items[3]
        
        # 使用完整评估
        evaluator = ModuleSummarizationEvaluator()
        result = evaluator.evaluate_single_prediction(prediction, reference, intent, context)
        score = result.get('comparison_score', 0.0)
        
    else:
        # 使用简化评估
        evaluator = ModuleSummarizationEvaluator()
        score = evaluator._simple_comparison_metric(reference, prediction)
    
    print(f"📊 Module Summarization得分: {score:.3f}")
    return score


if __name__ == "__main__":
    # 测试评估功能
    print("🧪 测试模块摘要评估功能...")
    
    # 创建测试数据
    test_predictions = ["This module provides utility functions for data processing."]
    test_references = ["This is a utility module that contains helper functions for various data operations."]
    test_intents = ["Document the utility module functionality"]
    test_contexts = ["def process_data(data): return data.strip()"]
    
    # 测试评估
    try:
        evaluator = ModuleSummarizationEvaluator()
        results = evaluate_predictions_batch(
            test_predictions, test_references, test_intents, test_contexts, evaluator
        )
        print("✅ 评估功能测试成功")
    except Exception as e:
        print(f"❌ 评估功能测试失败: {e}")
