mport torch
import pandas as pd
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple

class MultiBLiMPEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Hindi linguistic phenomena to test
        self.phenomena = [
            'subject_verb_agreement',
            'morphological_inflection', 
            'case_marking',
            'word_order',
            'negation',
            'quantifier_scope'
        ]
    
    def load_multiblimp_hindi(self) -> Dict:
        """Load MultiBLiMP Hindi test cases"""
        try:
            # Load MultiBLiMP dataset for Hindi
            dataset = load_dataset('BabyLM/MultilingualBlimp', 'hi')
            return dataset
        except:
            # If not available, create minimal pairs manually
            return self.create_hindi_minimal_pairs()
    
    def create_hindi_minimal_pairs(self) -> Dict:
        """Create Hindi minimal pairs for evaluation"""
        minimal_pairs = {
            'subject_verb_agreement': [
                {
                    'good': 'लड़का खाता है',    # Boy eats (singular)
                    'bad': 'लड़का खाते हैं',     # Boy eat (plural) - incorrect
                    'phenomenon': 'subject_verb_agreement'
                },
                {
                    'good': 'लड़के खाते हैं',    # Boys eat (plural)  
                    'bad': 'लड़के खाता है',     # Boys eats (singular) - incorrect
                    'phenomenon': 'subject_verb_agreement'
                }
            ],
            'case_marking': [
                {
                    'good': 'राम ने किताब पढ़ी',  # Ram read book (ergative)
                    'bad': 'राम किताब पढ़ी',      # Ram book read - missing ergative
                    'phenomenon': 'case_marking'
                }
            ]
            # Add more phenomena...
        }
        
        return minimal_pairs
    
    def evaluate_minimal_pair(self, good_sentence: str, bad_sentence: str) -> bool:
        """Evaluate single minimal pair - returns True if model prefers good sentence"""
        
        # Tokenize both sentences
        good_inputs = self.tokenizer(good_sentence, return_tensors='pt')
        bad_inputs = self.tokenizer(bad_sentence, return_tensors='pt')
        
        with torch.no_grad():
            # Get perplexity scores
            good_outputs = self.model(**good_inputs, labels=good_inputs['input_ids'])
            bad_outputs = self.model(**bad_inputs, labels=bad_inputs['input_ids'])
            
            good_loss = good_outputs.loss.item()
            bad_loss = bad_outputs.loss.item()
            
            # Model prefers sentence with lower loss (higher likelihood)
            return good_loss < bad_loss
    
    def evaluate_phenomenon(self, phenomenon: str, pairs: List[Dict]) -> Dict:
        """Evaluate all pairs for a specific phenomenon"""
        correct_predictions = 0
        total_pairs = len(pairs)
        
        for pair in pairs:
            is_correct = self.evaluate_minimal_pair(pair['good'], pair['bad'])
            if is_correct:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_pairs
        
        return {
            'phenomenon': phenomenon,
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_pairs
        }
    
    def evaluate_all_phenomena(self) -> Dict:
        """Evaluate model on all Hindi linguistic phenomena"""
        data = self.load_multiblimp_hindi()
        results = {}
        
        for phenomenon in self.phenomena:
            if phenomenon in data:
                pairs = data[phenomenon]
                results[phenomenon] = self.evaluate_phenomenon(phenomenon, pairs)
        
        # Calculate overall accuracy
        total_correct = sum(r['correct'] for r in results.values())
        total_pairs = sum(r['total'] for r in results.values())
        
        results['overall'] = {
            'accuracy': total_correct / total_pairs if total_pairs > 0 else 0,
            'total_correct': total_correct,
            'total_pairs': total_pairs
        }
        
        return results