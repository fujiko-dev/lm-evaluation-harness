"""
CI Builds Repair evaluation metrics and functions
"""

import os
import re
import json
import tempfile
import subprocess
import difflib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CIRepairResult:
    """CI修复结果"""
    success: bool
    explanation: str
    score: float
    details: Dict[str, Any]


class CIBuildsRepairEvaluator:
    """
    CI Builds Repair评估器
    评估模型生成的修复方案的质量
    """
    
    def __init__(self):
        self.results = []
    
    def evaluate_single_prediction(
        self, 
        prediction: str, 
        datapoint: Dict[str, Any]
    ) -> CIRepairResult:
        """
        评估单个预测结果
        
        Args:
            prediction: 模型生成的修复方案
            datapoint: 包含参考答案的数据点
            
        Returns:
            CIRepairResult: 评估结果
        """
        try:
            # 获取参考答案
            target_diff = datapoint.get('target_diff', '')
            target_changes = datapoint.get('file_changes', {})
            
            # 解析预测的修复方案
            predicted_changes = self._parse_prediction(prediction)
            
            # 计算相似度分数
            similarity_score = self._calculate_similarity(predicted_changes, target_changes)
            
            # 检查关键修复点
            key_fixes_score = self._check_key_fixes(prediction, datapoint)
            
            # 综合评分
            final_score = (similarity_score * 0.6 + key_fixes_score * 0.4)
            
            # 判断是否成功（阈值可调整）
            success = final_score > 0.5
            
            result = CIRepairResult(
                success=success,
                explanation=f"Similarity: {similarity_score:.3f}, Key fixes: {key_fixes_score:.3f}",
                score=final_score,
                details={
                    'similarity_score': similarity_score,
                    'key_fixes_score': key_fixes_score,
                    'predicted_changes': predicted_changes,
                    'target_changes': target_changes,
                    'difficulty': datapoint.get('difficulty', 0)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"评估预测时出错: {str(e)}")
            return CIRepairResult(
                success=False,
                explanation=f"Evaluation error: {str(e)}",
                score=0.0,
                details={'error': str(e)}
            )
    
    def _parse_prediction(self, prediction: str) -> Dict[str, List[str]]:
        """
        解析模型预测的修复方案，提取文件修改
        
        Args:
            prediction: 原始预测文本
            
        Returns:
            Dict[str, List[str]]: 文件路径到修改行的映射
        """
        changes = {}
        
        # 尝试提取代码块
        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', prediction, re.DOTALL)
        
        # 提取文件路径和修改内容
        file_patterns = [
            r'(?:修改|change|edit|fix)\s*(?:文件|file)?\s*[:\-]?\s*`?([^\s`\n]+\.[a-zA-Z]+)`?',
            r'(?:文件|file)\s*[:\-]?\s*`?([^\s`\n]+\.[a-zA-Z]+)`?',
            r'`([^\s`\n]+\.[a-zA-Z]+)`',
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, prediction, re.IGNORECASE)
            for match in matches:
                if match not in changes:
                    changes[match] = []
        
        # 如果没有找到文件，尝试从代码块中推断
        if not changes and code_blocks:
            # 假设是对主要文件的修改
            changes['inferred_file'] = code_blocks
        
        # 提取关键修改内容
        for file_path in changes:
            # 查找与该文件相关的修改内容
            change_content = []
            
            # 提取代码块内容
            for block in code_blocks:
                change_content.append(block.strip())
            
            # 如果没有代码块，尝试提取其他修改描述
            if not change_content:
                lines = prediction.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['import', 'from', 'def', 'class', 'fix', '修复']):
                        change_content.append(line.strip())
            
            changes[file_path] = change_content
        
        return changes
    
    def _calculate_similarity(
        self, 
        predicted_changes: Dict[str, List[str]], 
        target_changes: Dict[str, str]
    ) -> float:
        """
        计算预测修改与目标修改的相似度
        
        Args:
            predicted_changes: 预测的修改
            target_changes: 目标修改
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if not target_changes:
            return 0.5  # 如果没有参考答案，给中等分数
        
        if not predicted_changes:
            return 0.0
        
        # 计算文件层面的匹配度
        target_files = set(target_changes.keys())
        predicted_files = set()
        
        # 提取预测中的文件名
        for file_path in predicted_changes.keys():
            # 尝试匹配文件名的不同形式
            for target_file in target_files:
                if (file_path in target_file or target_file in file_path or 
                    os.path.basename(file_path) == os.path.basename(target_file)):
                    predicted_files.add(target_file)
                    break
        
        # 文件匹配分数
        file_match_score = len(predicted_files.intersection(target_files)) / len(target_files) if target_files else 0
        
        # 内容相似度分数
        content_scores = []
        for target_file, target_content in target_changes.items():
            if target_file in predicted_files:
                # 找到对应的预测内容
                pred_content = ""
                for pred_file, pred_changes in predicted_changes.items():
                    if (pred_file in target_file or target_file in pred_file or 
                        os.path.basename(pred_file) == os.path.basename(target_file)):
                        pred_content = " ".join(pred_changes)
                        break
                
                # 计算文本相似度
                similarity = self._text_similarity(pred_content, target_content)
                content_scores.append(similarity)
            else:
                content_scores.append(0.0)
        
        avg_content_score = sum(content_scores) / len(content_scores) if content_scores else 0
        
        # 综合分数
        return (file_match_score * 0.4 + avg_content_score * 0.6)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # 使用difflib计算相似度
        similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # 检查关键词匹配
        keywords1 = set(re.findall(r'\b\w+\b', text1.lower()))
        keywords2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if keywords1 and keywords2:
            keyword_similarity = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
            # 结合字符级和关键词级相似度
            similarity = (similarity * 0.7 + keyword_similarity * 0.3)
        
        return similarity
    
    def _check_key_fixes(self, prediction: str, datapoint: Dict[str, Any]) -> float:
        """
        检查预测是否包含关键修复点
        
        Args:
            prediction: 预测文本
            datapoint: 数据点
            
        Returns:
            float: 关键修复点分数 (0-1)
        """
        score = 0.0
        checks_passed = 0
        total_checks = 0
        
        # 从日志中提取错误信息
        logs = datapoint.get('logs', [])
        error_keywords = []
        
        for log_entry in logs:
            log_content = log_entry.get('log', '').lower()
            
            # 提取常见错误类型
            if 'import' in log_content and ('error' in log_content or 'failed' in log_content):
                error_keywords.extend(['import', 'from'])
                total_checks += 1
                if any(keyword in prediction.lower() for keyword in ['import', 'from']):
                    checks_passed += 1
            
            if 'syntax' in log_content and 'error' in log_content:
                error_keywords.extend(['syntax', 'format'])
                total_checks += 1
                if any(keyword in prediction.lower() for keyword in ['syntax', 'format', 'fix']):
                    checks_passed += 1
            
            if 'lint' in log_content or 'format' in log_content:
                error_keywords.extend(['lint', 'format'])
                total_checks += 1
                if any(keyword in prediction.lower() for keyword in ['lint', 'format', 'style']):
                    checks_passed += 1
            
            if 'test' in log_content and ('fail' in log_content or 'error' in log_content):
                error_keywords.extend(['test', 'assertion'])
                total_checks += 1
                if any(keyword in prediction.lower() for keyword in ['test', 'assert', 'mock']):
                    checks_passed += 1
        
        # 检查是否提到了修复方案
        fix_indicators = ['fix', 'repair', 'change', 'modify', 'update', '修复', '修改', '更新']
        total_checks += 1
        if any(indicator in prediction.lower() for indicator in fix_indicators):
            checks_passed += 1
        
        # 检查是否提到了具体文件
        changed_files = datapoint.get('changed_files', [])
        if changed_files:
            total_checks += 1
            for file_path in changed_files:
                file_name = os.path.basename(file_path)
                if file_name in prediction or file_path in prediction:
                    checks_passed += 1
                    break
        
        # 计算分数
        if total_checks > 0:
            score = checks_passed / total_checks
        
        return score
    
    def evaluate_batch(
        self, 
        predictions: List[str], 
        datapoints: List[Dict[str, Any]]
    ) -> List[CIRepairResult]:
        """
        批量评估预测结果
        
        Args:
            predictions: 预测列表
            datapoints: 数据点列表
            
        Returns:
            List[CIRepairResult]: 评估结果列表
        """
        results = []
        
        for pred, datapoint in zip(predictions, datapoints):
            result = self.evaluate_single_prediction(pred, datapoint)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def get_aggregate_metrics(self, results: List[CIRepairResult] = None) -> Dict[str, float]:
        """
        计算聚合指标
        
        Args:
            results: 评估结果列表，如果为None则使用内部存储的结果
            
        Returns:
            Dict[str, float]: 聚合指标
        """
        if results is None:
            results = self.results
        
        if not results:
            return {}
        
        # 成功率
        success_rate = sum(1 for r in results if r.success) / len(results)
        
        # 平均分数
        avg_score = sum(r.score for r in results) / len(results)
        
        # 按难度分组的指标
        difficulty_metrics = {}
        for difficulty in [0, 1, 2, 3, 4]:
            diff_results = [r for r in results if r.details.get('difficulty') == difficulty]
            if diff_results:
                difficulty_metrics[f'success_rate_difficulty_{difficulty}'] = (
                    sum(1 for r in diff_results if r.success) / len(diff_results)
                )
                difficulty_metrics[f'avg_score_difficulty_{difficulty}'] = (
                    sum(r.score for r in diff_results) / len(diff_results)
                )
        
        metrics = {
            'success_rate': success_rate,
            'average_score': avg_score,
            'total_samples': len(results),
            **difficulty_metrics
        }
        
        return metrics


def ci_builds_repair_success_rate(predictions: List[str], references: List[Dict[str, Any]]) -> float:
    """
    计算CI修复成功率指标
    
    Args:
        predictions: 模型预测列表
        references: 参考数据列表
        
    Returns:
        float: 成功率
    """
    evaluator = CIBuildsRepairEvaluator()
    results = evaluator.evaluate_batch(predictions, references)
    metrics = evaluator.get_aggregate_metrics(results)
    return metrics.get('success_rate', 0.0)


def ci_builds_repair_average_score(predictions: List[str], references: List[Dict[str, Any]]) -> float:
    """
    计算CI修复平均分数指标
    
    Args:
        predictions: 模型预测列表
        references: 参考数据列表
        
    Returns:
        float: 平均分数
    """
    evaluator = CIBuildsRepairEvaluator()
    results = evaluator.evaluate_batch(predictions, references)
    metrics = evaluator.get_aggregate_metrics(results)
    return metrics.get('average_score', 0.0)


def evaluate_predictions_batch(
    predictions: List[str], 
    datapoints: List[Dict[str, Any]],
    save_results: bool = True,
    results_file: str = None
) -> Dict[str, Any]:
    """
    批量评估预测结果并保存详细结果
    
    Args:
        predictions: 预测列表
        datapoints: 数据点列表
        save_results: 是否保存结果
        results_file: 结果文件路径
        
    Returns:
        Dict[str, Any]: 详细评估结果
    """
    evaluator = CIBuildsRepairEvaluator()
    results = evaluator.evaluate_batch(predictions, datapoints)
    
    # 计算聚合指标
    metrics = evaluator.get_aggregate_metrics(results)
    
    # 构建详细结果
    detailed_results = {
        'metrics': metrics,
        'individual_results': []
    }
    
    for i, (result, datapoint) in enumerate(zip(results, datapoints)):
        detailed_results['individual_results'].append({
            'index': i,
            'id': datapoint.get('id'),
            'repo': f"{datapoint.get('repo_owner')}/{datapoint.get('repo_name')}",
            'difficulty': datapoint.get('difficulty'),
            'success': result.success,
            'score': result.score,
            'explanation': result.explanation,
            'details': result.details
        })
    
    # 保存结果
    if save_results and results_file:
        try:
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            logger.info(f"评估结果已保存到: {results_file}")
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
    
    return detailed_results
