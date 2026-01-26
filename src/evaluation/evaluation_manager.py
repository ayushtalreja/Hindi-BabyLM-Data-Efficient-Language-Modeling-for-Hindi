import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any
from dataclasses import is_dataclass, asdict
from .indicglue_evaluator import IndicGLUEEvaluator
from .multiblimp_evaluator import MultiBLiMPEvaluator


class DataclassJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles dataclass objects and other non-serializable types"""

    def default(self, obj):
        # Handle dataclass objects
        if is_dataclass(obj):
            return asdict(obj)

        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Handle numpy types
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle other common non-serializable types
        if hasattr(obj, '__dict__'):
            return obj.__dict__

        # Let the base class handle other types or raise TypeError
        return super().default(obj)

class EvaluationManager:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Initialize evaluators
        self.indicglue_evaluator = IndicGLUEEvaluator(model, tokenizer, config)
        self.multiblimp_evaluator = MultiBLiMPEvaluator(model, tokenizer, config)

        # Results storage
        self.results = {}

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """
        Recursively convert objects to JSON-serializable format.
        Handles dataclasses, datetime objects, numpy types, and nested structures.
        """
        # Handle None
        if obj is None:
            return None

        # Handle numpy types first (before checking for primitive types)
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [EvaluationManager._make_serializable(item) for item in obj.tolist()]

        # Handle dataclass objects
        if is_dataclass(obj):
            return EvaluationManager._make_serializable(asdict(obj))

        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {k: EvaluationManager._make_serializable(v) for k, v in obj.items()}

        # Handle lists and tuples recursively
        if isinstance(obj, (list, tuple)):
            return [EvaluationManager._make_serializable(item) for item in obj]

        # Handle objects with __dict__ attribute
        if hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool)):
            return EvaluationManager._make_serializable(obj.__dict__)

        # Return primitive types as-is
        return obj
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run all evaluation tasks and compile results"""
        print("Starting comprehensive evaluation...")
        
        # 1. IndicGLUE Evaluation
        print("\n1. Running IndicGLUE evaluation...")
        indicglue_results = self.indicglue_evaluator.evaluate_all_tasks()
        self.results['indicglue'] = indicglue_results
        
        # 2. MultiBLiMP Evaluation
        print("\n2. Running MultiBLiMP evaluation...")
        multiblimp_results = self.multiblimp_evaluator.evaluate_all_phenomena()
        self.results['multiblimp'] = multiblimp_results

        # 3. Generate Summary
        summary = self.generate_summary()
        self.results['summary'] = summary

        # 4. Save Results
        self.save_results()
        
        return self.results
    
    def generate_summary(self) -> Dict:
        """Generate evaluation summary"""
        summary = {
            'evaluation_date': datetime.now().isoformat(),
            'model_config': self._make_serializable(self.config),
            'overall_scores': {}
        }

        # IndicGLUE average (defensive - handle missing results)
        if 'indicglue' in self.results:
            indicglue_scores = [v.get('accuracy', 0) for v in self.results['indicglue'].values()
                               if isinstance(v, dict) and 'accuracy' in v]
            if indicglue_scores:
                summary['overall_scores']['indicglue_avg'] = sum(indicglue_scores) / len(indicglue_scores)

        # MultiBLiMP overall (use correct key: 'average_accuracy' or 'overall_accuracy')
        if 'multiblimp' in self.results and 'overall' in self.results['multiblimp']:
            multiblimp_overall = self.results['multiblimp']['overall']
            # Try multiple possible keys for robustness
            summary['overall_scores']['multiblimp_accuracy'] = (
                multiblimp_overall.get('average_accuracy') or
                multiblimp_overall.get('overall_accuracy') or
                multiblimp_overall.get('accuracy', 0.0)
            )

        return summary
    
    def save_results(self):
        """Save evaluation results to files"""
        # Save results to experiment directory instead of timestamp-based directory
        experiment_name = self.config.get('experiment_name', 'default_experiment')
        results_dir = os.path.join(self.config.get('results_dir', 'results'), experiment_name)
        os.makedirs(results_dir, exist_ok=True)

        # Make results serializable before saving
        serializable_results = self._make_serializable(self.results)

        # Save comprehensive results as JSON with custom encoder as backup
        results_file = os.path.join(results_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, cls=DataclassJSONEncoder)
        
        # Save summary as CSV for easy analysis
        summary_df = pd.DataFrame([self.results['summary']['overall_scores']])
        summary_file = os.path.join(results_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Results saved to: {results_dir}")
        
        return results_dir