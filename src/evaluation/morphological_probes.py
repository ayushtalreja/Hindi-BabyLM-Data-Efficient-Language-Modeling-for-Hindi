import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from typing import Dict, List, Tuple

class MorphologicalProbe:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.probe_tasks = [
            'case_detection',      # Detect grammatical case
            'number_detection',    # Detect singular/plural
            'gender_detection',    # Detect grammatical gender
            'tense_detection'      # Detect verb tense
        ]
    
    def extract_representations(self, sentences: List[str], target_positions: List[int]) -> torch.Tensor:
        """Extract contextualized representations for target words"""
        representations = []
        
        for sentence, pos in zip(sentences, target_positions):
            inputs = self.tokenizer(sentence, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Get representation from specific layer (e.g., last layer)
                hidden_states = outputs.hidden_states[-1]  # Last layer
                target_repr = hidden_states[0, pos, :]  # Position of target word
                representations.append(target_repr)
        
        return torch.stack(representations)
    
    def create_case_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for case detection"""
        sentences = [
            "राम ने किताब पढ़ी",      # Ram-ERG book read
            "राम को पैसे मिले",       # Ram-DAT money got  
            "राम से बात हुई",        # Ram-ABL talk happened
            "राम में अच्छाई है",      # Ram-LOC goodness is
            # Add more examples...
        ]
        
        target_positions = [1, 1, 1, 1]  # Position of "राम" in each sentence
        labels = ['ergative', 'dative', 'ablative', 'locative']
        
        return sentences, target_positions, labels
    
    def train_probe(self, representations: torch.Tensor, labels: List[str]) -> LogisticRegression:
        """Train linear probe on representations"""
        # Convert to numpy
        X = representations.cpu().numpy()
        y = labels
        
        # Train logistic regression classifier
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)
        
        return probe
    
    def evaluate_case_detection(self) -> Dict[str, float]:
        """Evaluate case detection probe"""
        sentences, positions, labels = self.create_case_probe_data()
        
        # Split train/test (80/20)
        split_idx = int(0.8 * len(sentences))
        
        train_sentences = sentences[:split_idx]
        train_positions = positions[:split_idx] 
        train_labels = labels[:split_idx]
        
        test_sentences = sentences[split_idx:]
        test_positions = positions[split_idx:]
        test_labels = labels[split_idx:]
        
        # Extract representations
        train_repr = self.extract_representations(train_sentences, train_positions)
        test_repr = self.extract_representations(test_sentences, test_positions)
        
        # Train probe
        probe = self.train_probe(train_repr, train_labels)
        
        # Evaluate
        predictions = probe.predict(test_repr.cpu().numpy())
        accuracy = accuracy_score(test_labels, predictions)
        
        return {
            'task': 'case_detection',
            'accuracy': accuracy,
            'predictions': predictions.tolist(),
            'true_labels': test_labels
        }
    
    def run_all_probes(self) -> Dict:
        """Run all morphological probe tasks"""
        results = {}
        
        # Case detection
        results['case_detection'] = self.evaluate_case_detection()
        
        # Add other probe tasks...
        
        return results