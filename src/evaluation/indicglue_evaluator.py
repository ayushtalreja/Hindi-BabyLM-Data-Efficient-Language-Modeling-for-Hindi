import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple

class IndicGLUEEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tasks = [
            'IndicNews',     # Article genre classification
            'IndicHeadline', # Headline prediction
            'IndicWiki',     # Section title prediction
            'IndicCQ',       # Cloze-style QA
            'IndicWNLI',     # Winograd NLI
            'IndicCOPA'      # COPA (Choice of Plausible Alternatives)
        ]
    
    def load_task_data(self, task_name: str) -> Dict:
        """Load specific IndicGLUE task data"""
        if task_name == 'IndicNews':
            dataset = load_dataset('ai4bharat/indicnews', 'hi')
        elif task_name == 'IndicHeadline':
            dataset = load_dataset('ai4bharat/IndicHeadline', 'hi')
        # Add other tasks...
        
        return dataset
    
    def evaluate_classification_task(self, task_name: str) -> Dict[str, float]:
        """Evaluate on classification tasks"""
        dataset = self.load_task_data(task_name)
        test_data = dataset['test']
        
        predictions = []
        labels = []
        
        for example in test_data:
            # Tokenize input
            inputs = self.tokenizer(
                example['text'], 
                truncation=True, 
                padding=True, 
                return_tensors='pt'
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1)
                predictions.append(pred.item())
                labels.append(example['label'])
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'task': task_name
        }
    
    def evaluate_all_tasks(self) -> Dict[str, Dict]:
        """Evaluate model on all IndicGLUE tasks"""
        results = {}
        
        for task in self.tasks:
            print(f"Evaluating {task}...")
            try:
                task_results = self.evaluate_classification_task(task)
                results[task] = task_results
            except Exception as e:
                print(f"Error evaluating {task}: {e}")
                results[task] = {'error': str(e)}
        
        return results