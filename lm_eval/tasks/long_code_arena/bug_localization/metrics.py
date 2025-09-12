"""
Bug Localization Metrics

This module implements the evaluation metrics for bug localization task,
following the metrics from the Long Code Arena paper.
"""

from typing import List, Dict, Any, Set


def precision(expected_set: set, actual_set: set) -> float:
    """Calculate precision: TP / (TP + FP)"""
    if len(actual_set) == 0:
        return 0.0
    
    true_positives = len(expected_set & actual_set)
    false_positives = len(actual_set - expected_set)
    
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0


def recall(expected_set: set, actual_set: set) -> float:
    """Calculate recall: TP / (TP + FN)"""
    if len(expected_set) == 0:
        return 1.0 if len(actual_set) == 0 else 0.0
    
    true_positives = len(expected_set & actual_set)
    false_negatives = len(expected_set - actual_set)
    
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0


def f1_score(precision_value: float, recall_value: float) -> float:
    """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
    if precision_value + recall_value == 0:
        return 0.0
    
    return 2 * (precision_value * recall_value) / (precision_value + recall_value)


def false_positive_rate(expected_set: set, actual_set: set, all_files_set: set) -> float:
    """Calculate False Positive Rate: FP / (FP + TN)"""
    false_positives = len(actual_set - expected_set)
    true_negatives = len(all_files_set - expected_set - actual_set)
    
    return false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0


def compute_quality_metrics(all_files: List[str], expected_files: List[str], predicted_files: List[str]) -> Dict[str, Any]:
    """
    Compute all quality metrics for bug localization
    
    Args:
        all_files: List of all files in the repository
        expected_files: List of files that should be modified (ground truth)
        predicted_files: List of files predicted to be modified
    
    Returns:
        Dictionary containing all computed metrics
    """
    # Convert to sets for easier computation
    all_files_set = set(all_files)
    expected_set = set(expected_files)
    predicted_set = set(predicted_files)
    
    # Calculate basic metrics
    precision_value = precision(expected_set, predicted_set)
    recall_value = recall(expected_set, predicted_set)
    f1_value = f1_score(precision_value, recall_value)
    fpr_value = false_positive_rate(expected_set, predicted_set, all_files_set)
    
    # Boolean metrics
    all_correct = predicted_set == expected_set  # Check if all files were identified correctly
    at_least_one_correct = len(expected_set & predicted_set) > 0  # Check if at least one file is identified correctly
    all_incorrect = len(expected_set & predicted_set) == 0 and len(predicted_set) > 0  # Check if all identified files are incorrect
    
    # Additional metrics
    target_files_rate = len(predicted_set) / len(all_files_set) if len(all_files_set) > 0 else 0.0
    
    return {
        'bug_loc_precision': precision_value,
        'bug_loc_recall': recall_value,
        'bug_loc_f1': f1_value,
        'bug_loc_fpr': fpr_value,
        'bug_loc_all_correct': int(all_correct),
        'bug_loc_at_least_one_correct': int(at_least_one_correct),
        'bug_loc_all_incorrect': int(all_incorrect),
        'bug_loc_target_files_rate': target_files_rate,
        # Additional detailed metrics for analysis
        'all_files_count': len(all_files),
        'expected_files_count': len(expected_files),
        'predicted_files_count': len(predicted_files),
        'true_positives': len(expected_set & predicted_set),
        'false_positives': len(predicted_set - expected_set),
        'false_negatives': len(expected_set - predicted_set),
        'true_negatives': len(all_files_set - expected_set - predicted_set)
    }


def compute_context_metrics(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute context-related metrics from the conversation messages
    
    Args:
        messages: List of chat messages with role and content
    
    Returns:
        Dictionary containing context metrics
    """
    total_tokens = 0
    user_tokens = 0
    system_tokens = 0
    
    for message in messages:
        content = message.get('content', '')
        tokens = len(content.split())  # Simple token counting
        total_tokens += tokens
        
        if message.get('role') == 'user':
            user_tokens += tokens
        elif message.get('role') == 'system':
            system_tokens += tokens
    
    return {
        'context_total_tokens': total_tokens,
        'context_user_tokens': user_tokens,
        'context_system_tokens': system_tokens,
        'context_messages_count': len(messages)
    }


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across all examples
    
    Args:
        results: List of result dictionaries from individual examples
    
    Returns:
        Dictionary containing aggregated metrics
    """
    if not results:
        return {}
    
    # Metrics to aggregate
    metric_keys = [
        'bug_loc_precision', 'bug_loc_recall', 'bug_loc_f1', 'bug_loc_fpr',
        'bug_loc_all_correct', 'bug_loc_at_least_one_correct', 'bug_loc_all_incorrect',
        'bug_loc_target_files_rate'
    ]
    
    aggregated = {}
    
    for key in metric_keys:
        values = [r.get(key, 0.0) for r in results if key in r]
        if values:
            aggregated[f'{key}_mean'] = sum(values) / len(values)
            aggregated[f'{key}_count'] = len(values)
    
    # Additional statistics
    aggregated['total_examples'] = len(results)
    aggregated['successful_predictions'] = sum(1 for r in results if r.get('predicted_files_count', 0) > 0)
    
    return aggregated


def print_metrics_summary(metrics: Dict[str, Any], title: str = "Bug Localization Metrics"):
    """
    Print a formatted summary of metrics
    
    Args:
        metrics: Dictionary containing metrics
        title: Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    # Quality metrics
    quality_metrics = [
        ('Precision', 'bug_loc_precision'),
        ('Recall', 'bug_loc_recall'),
        ('F1 Score', 'bug_loc_f1'),
        ('False Positive Rate', 'bug_loc_fpr'),
        ('All Correct', 'bug_loc_all_correct'),
        ('At Least One Correct', 'bug_loc_at_least_one_correct'),
        ('All Incorrect', 'bug_loc_all_incorrect'),
        ('Target Files Rate', 'bug_loc_target_files_rate')
    ]
    
    for label, key in quality_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"{label:<25}: {value:.4f}")
            else:
                print(f"{label:<25}: {value}")
    
    # Count metrics
    count_metrics = [
        ('Total Examples', 'total_examples'),
        ('Successful Predictions', 'successful_predictions'),
        ('All Files Count', 'all_files_count'),
        ('Expected Files Count', 'expected_files_count'),
        ('Predicted Files Count', 'predicted_files_count')
    ]
    
    print(f"\n{'-'*30}")
    print("Count Statistics")
    print(f"{'-'*30}")
    
    for label, key in count_metrics:
        if key in metrics:
            print(f"{label:<25}: {metrics[key]}")
    
    print(f"{'='*50}\n")


# Test function
if __name__ == '__main__':
    # Test metrics with sample data
    all_files = ['file1.py', 'file2.py', 'file3.py', 'file4.py', 'file5.py']
    expected_files = ['file2.py', 'file4.py']
    predicted_files = ['file2.py', 'file3.py']
    
    metrics = compute_quality_metrics(all_files, expected_files, predicted_files)
    print_metrics_summary(metrics, "Test Metrics")
    
    # Test edge cases
    print("Testing edge cases...")
    
    # Empty predictions
    metrics_empty = compute_quality_metrics(all_files, expected_files, [])
    print(f"Empty predictions - Precision: {metrics_empty['bug_loc_precision']:.4f}, Recall: {metrics_empty['bug_loc_recall']:.4f}")
    
    # Perfect predictions
    metrics_perfect = compute_quality_metrics(all_files, expected_files, expected_files)
    print(f"Perfect predictions - Precision: {metrics_perfect['bug_loc_precision']:.4f}, Recall: {metrics_perfect['bug_loc_recall']:.4f}, F1: {metrics_perfect['bug_loc_f1']:.4f}")
    
    # All wrong predictions
    metrics_wrong = compute_quality_metrics(all_files, expected_files, ['file1.py', 'file5.py'])
    print(f"All wrong predictions - Precision: {metrics_wrong['bug_loc_precision']:.4f}, Recall: {metrics_wrong['bug_loc_recall']:.4f}")
