"""
Project-level Code Completion评估工具
基于论文中的评估方法实现
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CompletionMetrics:
    """代码补全评估指标"""
    exact_match: float = 0.0
    edit_distance: float = 0.0
    bleu_score: float = 0.0
    line_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'exact_match': self.exact_match,
            'edit_distance': self.edit_distance,
            'bleu_score': self.bleu_score,
            'line_count': self.line_count
        }


@dataclass
class DetailedResults:
    """详细的评估结果"""
    by_line_type: Dict[str, CompletionMetrics]
    overall: CompletionMetrics
    predictions: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'by_line_type': {k: v.to_dict() for k, v in self.by_line_type.items()},
            'overall': self.overall.to_dict(),
            'total_predictions': len(self.predictions)
        }


class ProjectLevelCodeCompletionEvaluator:
    """Project-level Code Completion评估器"""
    
    def __init__(self, results_save_path: str = None):
        self.results_save_path = results_save_path
        self.detailed_predictions = []
        self.line_type_stats = defaultdict(list)
    
    def _convert_to_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
        
    def calculate_edit_distance(self, pred: str, target: str) -> float:
        """
        计算编辑距离（归一化的Levenshtein距离）
        
        Args:
            pred: 预测的代码行
            target: 目标代码行
            
        Returns:
            归一化的编辑距离（0-1之间，0表示完全匹配）
        """
        # 实现动态规划计算编辑距离
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
                
            return previous_row[-1]
        
        # 预处理：去除多余的空白字符
        pred_clean = ' '.join(pred.split())
        target_clean = ' '.join(target.split())
        
        if not target_clean:
            return 0.0 if not pred_clean else 1.0
        
        distance = levenshtein_distance(pred_clean, target_clean)
        max_len = max(len(pred_clean), len(target_clean))
        
        return distance / max_len if max_len > 0 else 0.0
    
    def calculate_bleu_score(self, pred: str, target: str) -> float:
        """
        计算简化的BLEU分数（单句）
        
        Args:
            pred: 预测的代码行
            target: 目标代码行
            
        Returns:
            BLEU分数
        """
        def tokenize_code(code: str) -> List[str]:
            """简单的代码分词"""
            # 按空白字符和常见的代码符号分割
            tokens = re.findall(r'\w+|[^\w\s]', code)
            return [token.lower() for token in tokens if token.strip()]
        
        pred_tokens = tokenize_code(pred)
        target_tokens = tokenize_code(target)
        
        if not target_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        # 计算1-gram到4-gram的精确度
        def ngram_precision(pred_tokens: List[str], target_tokens: List[str], n: int) -> float:
            if len(pred_tokens) < n:
                return 0.0
            
            pred_ngrams = defaultdict(int)
            target_ngrams = defaultdict(int)
            
            # 生成n-grams
            for i in range(len(pred_tokens) - n + 1):
                ngram = tuple(pred_tokens[i:i+n])
                pred_ngrams[ngram] += 1
            
            for i in range(len(target_tokens) - n + 1):
                ngram = tuple(target_tokens[i:i+n])
                target_ngrams[ngram] += 1
            
            # 计算匹配的n-grams
            matches = 0
            total = sum(pred_ngrams.values())
            
            for ngram, count in pred_ngrams.items():
                matches += min(count, target_ngrams.get(ngram, 0))
            
            return matches / total if total > 0 else 0.0
        
        # 计算几何平均
        precisions = []
        for n in range(1, min(5, len(target_tokens) + 1)):
            precision = ngram_precision(pred_tokens, target_tokens, n)
            if precision > 0:
                precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        # 计算几何平均（简化版BLEU）
        bleu = np.exp(np.mean(np.log(precisions)))
        
        # 简化的长度惩罚
        brevity_penalty = min(1.0, len(pred_tokens) / len(target_tokens))
        
        return bleu * brevity_penalty
    
    def evaluate_single_prediction(
        self, 
        predicted_line: str, 
        target_line: str, 
        line_type: str,
        context_info: Optional[Dict[str, Any]] = None
    ) -> CompletionMetrics:
        """
        评估单个预测结果
        
        Args:
            predicted_line: 预测的代码行
            target_line: 目标代码行
            line_type: 行类型
            context_info: 上下文信息
            
        Returns:
            评估指标
        """
        # 预处理代码行
        pred_clean = predicted_line.strip()
        target_clean = target_line.strip()
        
        # 计算精确匹配
        exact_match = 1.0 if pred_clean == target_clean else 0.0
        
        # 计算编辑距离
        edit_distance = self.calculate_edit_distance(pred_clean, target_clean)
        
        # 计算BLEU分数
        bleu_score = self.calculate_bleu_score(pred_clean, target_clean)
        
        metrics = CompletionMetrics(
            exact_match=exact_match,
            edit_distance=edit_distance,
            bleu_score=bleu_score,
            line_count=1
        )
        
        # 保存详细预测信息
        prediction_detail = {
            'predicted_line': pred_clean,
            'target_line': target_clean,
            'line_type': line_type,
            'exact_match': float(exact_match),
            'edit_distance': float(edit_distance),
            'bleu_score': float(bleu_score),
            'context_info': self._convert_to_json_serializable(context_info or {})
        }
        
        self.detailed_predictions.append(prediction_detail)
        self.line_type_stats[line_type].append(metrics)
        
        return metrics
    
    def aggregate_results(self) -> DetailedResults:
        """
        聚合所有评估结果
        
        Returns:
            详细的评估结果
        """
        # 按行类型聚合结果
        by_line_type = {}
        all_metrics = []
        
        for line_type, metrics_list in self.line_type_stats.items():
            if metrics_list:
                aggregated = CompletionMetrics(
                    exact_match=np.mean([m.exact_match for m in metrics_list]),
                    edit_distance=np.mean([m.edit_distance for m in metrics_list]),
                    bleu_score=np.mean([m.bleu_score for m in metrics_list]),
                    line_count=len(metrics_list)
                )
                by_line_type[line_type] = aggregated
                all_metrics.extend(metrics_list)
        
        # 计算总体结果
        if all_metrics:
            overall = CompletionMetrics(
                exact_match=np.mean([m.exact_match for m in all_metrics]),
                edit_distance=np.mean([m.edit_distance for m in all_metrics]),
                bleu_score=np.mean([m.bleu_score for m in all_metrics]),
                line_count=len(all_metrics)
            )
        else:
            overall = CompletionMetrics()
        
        return DetailedResults(
            by_line_type=by_line_type,
            overall=overall,
            predictions=self.detailed_predictions
        )
    
    def save_results(self, results: DetailedResults, model_name: str, context_size: str) -> str:
        """
        保存评估结果到文件
        
        Args:
            results: 评估结果
            model_name: 模型名称
            context_size: 上下文大小
            
        Returns:
            保存的文件路径
        """
        if not self.results_save_path:
            return ""
        
        # 创建保存目录
        os.makedirs(self.results_save_path, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"project_level_code_completion_{model_name}_{context_size}_{timestamp}.json"
        filepath = os.path.join(self.results_save_path, filename)
        
        # 准备保存的数据
        save_data = {
            'model_name': model_name,
            'context_size': context_size,
            'timestamp': timestamp,
            'results': results.to_dict(),
            'detailed_predictions': results.predictions
        }
        
        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {filepath}")
        return filepath
    
    def print_summary(self, results: DetailedResults, model_name: str, context_size: str):
        """
        打印评估结果摘要
        
        Args:
            results: 评估结果
            model_name: 模型名称
            context_size: 上下文大小
        """
        print(f"\n{'='*60}")
        print(f"Project-level Code Completion 评估结果")
        print(f"模型: {model_name}")
        print(f"上下文大小: {context_size}")
        print(f"{'='*60}")
        
        # 总体结果
        overall = results.overall
        print(f"\n总体结果 (总计 {overall.line_count} 行):")
        print(f"  精确匹配率: {overall.exact_match:.4f}")
        print(f"  编辑距离: {overall.edit_distance:.4f}")
        print(f"  BLEU分数: {overall.bleu_score:.4f}")
        
        # 按行类型的结果
        print(f"\n按行类型的结果:")
        for line_type, metrics in results.by_line_type.items():
            print(f"  {line_type} ({metrics.line_count} 行):")
            print(f"    精确匹配率: {metrics.exact_match:.4f}")
            print(f"    编辑距离: {metrics.edit_distance:.4f}")
            print(f"    BLEU分数: {metrics.bleu_score:.4f}")
        
        print(f"\n{'='*60}\n")


def evaluate_predictions_batch(
    predictions: List[str],
    targets: List[str],
    line_types: List[str],
    context_infos: Optional[List[Dict[str, Any]]] = None,
    results_save_path: str = None
) -> DetailedResults:
    """
    批量评估预测结果
    
    Args:
        predictions: 预测结果列表
        targets: 目标结果列表
        line_types: 行类型列表
        context_infos: 上下文信息列表
        results_save_path: 结果保存路径
        
    Returns:
        详细的评估结果
    """
    if len(predictions) != len(targets) or len(predictions) != len(line_types):
        raise ValueError("预测结果、目标结果和行类型的数量必须一致")
    
    if context_infos and len(context_infos) != len(predictions):
        raise ValueError("如果提供上下文信息，数量必须与预测结果一致")
    
    evaluator = ProjectLevelCodeCompletionEvaluator(results_save_path)
    
    # 逐个评估
    for i, (pred, target, line_type) in enumerate(zip(predictions, targets, line_types)):
        context_info = context_infos[i] if context_infos else None
        evaluator.evaluate_single_prediction(pred, target, line_type, context_info)
    
    # 聚合结果
    results = evaluator.aggregate_results()
    
    return results
