"""
Custom task implementation for Long Code Arena library-based code generation
that handles dataset loading failures gracefully.
"""

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task
from .data_loader import load_lca_dataset, validate_dataset_format
from .utils import pass_at_1, build_predictions
import logging

logger = logging.getLogger(__name__)


@register_task("lca_library_based_code_generation_custom")
class LCALibraryBasedCodeGeneration(ConfigurableTask):
    """
    Custom implementation of the LCA library-based code generation task
    that can handle dataset loading failures.
    """
    
    VERSION = 1.0
    
    def __init__(self, config=None):
        super().__init__(config)
        self._dataset = None
        
    def has_training_docs(self):
        return False
        
    def has_validation_docs(self):
        return False
        
    def has_test_docs(self):
        return True
        
    def training_docs(self):
        return []
        
    def validation_docs(self):
        return []
        
    def test_docs(self):
        if self._dataset is None:
            try:
                self._dataset = load_lca_dataset()
                if not validate_dataset_format(self._dataset):
                    logger.warning("Dataset format validation failed")
            except Exception as e:
                logger.error(f"Failed to load LCA dataset: {e}")
                self._dataset = []
        
        return self._dataset
    
    def doc_to_text(self, doc):
        """Convert document to input text for the model."""
        return f"Instruction: {doc['instruction']}\n\nGenerate Python code:"
    
    def doc_to_target(self, doc):
        """Extract target from document."""
        return doc['reference']
    
    def construct_requests(self, doc, ctx, **kwargs):
        """Construct requests for the model."""
        request = Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {
                "until": ["\n\n\n", "\n# ", "\nclass", "\ndef"],
                "max_gen_toks": 1024,
                "do_sample": False,
                "temperature": 0.0
            }),
            idx=0,
            **kwargs
        )
        return [request]
    
    def process_results(self, doc, results):
        """Process model results and compute metrics."""
        if not results:
            return {"pass_at_1": 0.0}
        
        prediction = results[0] if results else ""
        reference = self.doc_to_target(doc)
        
        # Use our build_predictions and pass_at_1 functions
        processed_preds = build_predictions([[prediction]], [doc])
        score = pass_at_1([reference], processed_preds)
        
        return {"pass_at_1": score}
    
    def aggregation(self):
        """Define how to aggregate results."""
        return {
            "pass_at_1": "mean"
        }
    
    def higher_is_better(self):
        """Define whether higher scores are better."""
        return {
            "pass_at_1": True
        }
