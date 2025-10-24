import json
import pandas as pd
from datetime import datetime
import os
from typing import Dict
from .indicglue_evaluator import IndicGLUEEvaluator
from .multiblimp_evaluator import MultiBLiMPEvaluator

class EvaluationManager:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize evaluators
        self.indicglue_evaluator = IndicGLUEEvaluator(model, tokenizer)
        self.multiblimp_evaluator = MultiBLiMPEvaluator(model, tokenizer)
        
        # Results storage
        self.results = {}
    
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
            'model_config': self.config,
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.config.get('results_dir', 'results'), f'evaluation_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save comprehensive results as JSON
        results_file = os.path.join(results_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save summary as CSV for easy analysis
        summary_df = pd.DataFrame([self.results['summary']['overall_scores']])
        summary_file = os.path.join(results_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Results saved to: {results_dir}")
        
        return results_dir