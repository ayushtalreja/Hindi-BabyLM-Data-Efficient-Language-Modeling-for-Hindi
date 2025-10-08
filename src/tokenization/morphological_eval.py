import pandas as pd
from typing import List, Dict, Tuple

class MorphologicalEvaluator:
    def __init__(self):
        # Load Hindi morphological patterns
        self.inflection_patterns = self.load_inflection_patterns()
        self.compound_patterns = self.load_compound_patterns()
    
    def load_inflection_patterns(self) -> Dict:
        """Load common Hindi inflection patterns for evaluation"""
        # Common patterns: -ों (plural), -ने (ergative), -को (dative), etc.
        return {
            "plural": ["-ों", "-ें", "-यां"],
            "ergative": ["-ने"],
            "dative": ["-को"],
            "locative": ["-में", "-पर"],
            "ablative": ["-से"]
        }

    def load_compound_patterns(self) -> Dict:
        """Load compound word patterns for Hindi"""
        return {
            "noun_noun": ["रेलगाड़ी", "पाठशाला"],  # train, school
            "adj_noun": ["महापुरुष", "नवयुवक"],   # great-man, new-youth
        }

    def evaluate_morphological_preservation(self, tokenizer, test_words: List[str]) -> Dict:
        """Evaluate how well tokenizer preserves morphological structure"""
        results = {
            "over_segmentation": 0,  # Morphemes split incorrectly
            "under_segmentation": 0,  # Morphemes not split when they should be
            "correct_segmentation": 0
        }
        
        for word in test_words:
            tokens = tokenizer.tokenize(word)
            # Analyze morphological correctness
            score = self.score_morphological_tokenization(word, tokens)
            results[score] += 1
        
        return results
    
    def score_morphological_tokenization(self, word: str, tokens: List[str]) -> str:
        """Score individual word tokenization quality"""
        # Check if word has known morphological structure
        has_suffix = False
        expected_splits = 1  # Base form

        # Check for known suffixes
        for category, suffixes in self.inflection_patterns.items():
            for suffix in suffixes:
                if word.endswith(suffix):
                    has_suffix = True
                    expected_splits = 2  # Root + suffix
                    break
            if has_suffix:
                break

        num_tokens = len(tokens)

        # Evaluate tokenization
        if has_suffix:
            # Word should be split into root + suffix
            if num_tokens == 2:
                return "correct_segmentation"
            elif num_tokens > 2:
                return "over_segmentation"
            else:
                return "under_segmentation"
        else:
            # Simple word should not be split
            if num_tokens == 1:
                return "correct_segmentation"
            elif num_tokens > 1:
                return "over_segmentation"
            else:
                return "under_segmentation"
    
    def create_morphological_test_set(self) -> List[str]:
        """Create test set with known morphological patterns"""
        test_words = []

        # Base words
        base_words = [
            "लड़का",    # boy
            "किताब",    # book
            "घर",       # house
            "पानी",     # water
            "स्कूल",    # school
            "दोस्त",    # friend
        ]

        # Add base words
        test_words.extend(base_words)

        # Add inflected forms
        for base in base_words:
            # Plural forms
            test_words.append(base + "ों")
            test_words.append(base + "ें")

            # Case markers
            test_words.append(base + "ने")
            test_words.append(base + "को")
            test_words.append(base + "में")
            test_words.append(base + "से")

        # Add some complex compounds
        test_words.extend([
            "रेलगाड़ी",        # train (compound)
            "पाठशाला",        # school (compound)
            "जन्मदिन",        # birthday (compound)
        ])

        return test_words