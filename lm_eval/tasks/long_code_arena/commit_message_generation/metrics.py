"""
Custom metrics for commit message generation evaluation.
Based on the evaluation metrics from Long Code Arena benchmark.
"""

import re
from typing import List, Dict, Any
import math
from collections import Counter

# Import direct libraries instead of evaluate
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from lm_eval.api.registry import register_metric


def split_puncts(sentence: str) -> str:
    """Split punctuation marks from words for better tokenization."""
    return re.sub(r'([.!?,:;])', r' \1 ', sentence)


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score using sacrebleu"""
    if not predictions or not references:
        return 0.0
    
    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, refs, tokenize="13a")
    return bleu.score


def compute_chrf_score(predictions: List[str], references: List[str]) -> float:
    """Compute chrF score"""
    if not predictions or not references:
        return 0.0
    
    refs = [[ref] for ref in references]
    chrf = sacrebleu.corpus_chrf(predictions, refs)
    return chrf.score


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores"""
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        "rouge1": (sum(rouge1_scores) / len(rouge1_scores)) * 100,
        "rouge2": (sum(rouge2_scores) / len(rouge2_scores)) * 100,
        "rougeL": (sum(rougeL_scores) / len(rougeL_scores)) * 100,
    }


def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    """Compute BERTScore"""
    if not predictions or not references:
        return 0.0
    
    try:
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        return F1.mean().item()
    except Exception as e:
        print(f"BERTScore计算失败: {e}")
        return 0.0


def compute_bertscore_normalized(predictions: List[str], references: List[str]) -> float:
    """Compute normalized BERTScore"""
    if not predictions or not references:
        return 0.0
    
    try:
        P, R, F1 = bert_score(predictions, references, lang="en", rescale_with_baseline=True, verbose=False)
        return F1.mean().item()
    except Exception as e:
        print(f"Normalized BERTScore计算失败: {e}")
        return 0.0


def compute_bnorm_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute B-Norm score (BLEU variation with smoothing).
    Simplified implementation based on the original B-Norm metric.
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
        """Extract n-grams from token list"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def compute_bleu_for_pair(pred: str, ref: str, max_n: int = 4) -> float:
        """Compute BLEU score for a single prediction-reference pair"""
        pred_tokens = split_puncts(pred.strip().lower()).split()
        ref_tokens = split_puncts(ref.strip().lower()).split()
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))
        
        # N-gram precision scores
        precisions = []
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
                
            pred_counter = Counter(pred_ngrams)
            ref_counter = Counter(ref_ngrams)
            
            # Count matches
            matches = 0
            for ngram in pred_counter:
                matches += min(pred_counter[ngram], ref_counter.get(ngram, 0))
            
            # Add smoothing (Lin and Och, 2004)
            precision = (matches + 0.1) / (len(pred_ngrams) + 0.1)
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = (math.prod(precisions)) ** (1.0 / len(precisions))
        else:
            geo_mean = 0.0
        
        return bp * geo_mean
    
    # Compute B-Norm for all pairs
    scores = []
    for pred, ref in zip(predictions, references):
        score = compute_bleu_for_pair(pred, ref)
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0


@register_metric(
    metric="cmg_bleu",
    higher_is_better=True,
    aggregation="mean"
)
def cmg_bleu_metric(predictions: List[str], references: List[str]) -> float:
    """BLEU metric for commit message generation"""
    return compute_bleu_score(predictions, references)


@register_metric(
    metric="cmg_chrf",
    higher_is_better=True,
    aggregation="mean"
)
def cmg_chrf_metric(predictions: List[str], references: List[str]) -> float:
    """chrF metric for commit message generation"""
    return compute_chrf_score(predictions, references)


@register_metric(
    metric="cmg_rouge1",
    higher_is_better=True,
    aggregation="mean"
)
def cmg_rouge1_metric(predictions: List[str], references: List[str]) -> float:
    """ROUGE-1 metric for commit message generation"""
    rouge_scores = compute_rouge_scores(predictions, references)
    return rouge_scores["rouge1"]


@register_metric(
    metric="cmg_rouge2",
    higher_is_better=True,
    aggregation="mean"
)
def cmg_rouge2_metric(predictions: List[str], references: List[str]) -> float:
    """ROUGE-2 metric for commit message generation"""
    rouge_scores = compute_rouge_scores(predictions, references)
    return rouge_scores["rouge2"]


@register_metric(
    metric="cmg_rougeL",
    higher_is_better=True,
    aggregation="mean"
)
def cmg_rougeL_metric(predictions: List[str], references: List[str]) -> float:
    """ROUGE-L metric for commit message generation"""
    rouge_scores = compute_rouge_scores(predictions, references)
    return rouge_scores["rougeL"]


@register_metric(
    metric="cmg_bertscore",
    higher_is_better=True,
    aggregation="mean"
)
def cmg_bertscore_metric(predictions: List[str], references: List[str]) -> float:
    """BERTScore metric for commit message generation"""
    return compute_bertscore(predictions, references)


@register_metric(
    metric="cmg_bnorm",
    higher_is_better=True,
    aggregation="mean"
)
def cmg_bnorm_metric(predictions: List[str], references: List[str]) -> float:
    """B-Norm metric for commit message generation"""
    return compute_bnorm_score(predictions, references)
