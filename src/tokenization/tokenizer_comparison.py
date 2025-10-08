from typing import List, Dict
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

class TokenizerComparison:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.tokenizers = {}
    
    def train_bpe_tokenizer(self, corpus: List[str]) -> Tokenizer:
        """Train BPE tokenizer"""
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.pre_tokenizer = Whitespace()
        
        # Train on corpus
        tokenizer.train_from_iterator(corpus, trainer)
        return tokenizer
    
    def train_wordpiece_tokenizer(self, corpus: List[str]) -> Tokenizer:
        """Train WordPiece tokenizer"""
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.pre_tokenizer = Whitespace()
        
        tokenizer.train_from_iterator(corpus, trainer)
        return tokenizer
    
    def evaluate_tokenization_quality(self, tokenizers: Dict, test_texts: List[str]) -> Dict:
        """Compare tokenizers on various metrics"""
        results = {}
        
        for name, tokenizer in tokenizers.items():
            # Compute metrics: compression ratio, OOV rate, morphological preservation
            results[name] = self.compute_tokenizer_metrics(tokenizer, test_texts)
        
        return results
    
    def compute_tokenizer_metrics(self, tokenizer, test_texts: List[str]) -> Dict:
        """Compute comprehensive tokenizer evaluation metrics"""
        total_chars = 0
        total_tokens = 0
        total_words = 0

        tokenized_outputs = []

        for text in test_texts:
            # Tokenize the text
            if hasattr(tokenizer, 'encode'):
                # For Tokenizer objects (HuggingFace tokenizers)
                encoding = tokenizer.encode(text)
                tokens = encoding.tokens
            else:
                # For other tokenizer types
                tokens = tokenizer.tokenize(text)

            tokenized_outputs.append(tokens)

            # Collect statistics
            total_chars += len(text)
            total_tokens += len(tokens)
            total_words += len(text.split())

        # Compute metrics
        metrics = {
            # Compression ratio: chars per token (higher = more efficient)
            'compression_ratio': total_chars / total_tokens if total_tokens > 0 else 0,

            # Fertility: tokens per word (lower = better)
            'fertility': total_tokens / total_words if total_words > 0 else 0,

            # Average token length
            'avg_token_length': total_chars / total_tokens if total_tokens > 0 else 0,

            # Vocabulary efficiency
            'total_tokens': total_tokens,
            'total_chars': total_chars,
            'total_words': total_words,
        }

        # Compute OOV rate if tokenizer has vocab
        if hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            unk_token = '[UNK]'
            unk_count = 0

            for tokens in tokenized_outputs:
                unk_count += tokens.count(unk_token)

            metrics['oov_rate'] = unk_count / total_tokens if total_tokens > 0 else 0

        # Morphological preservation score (using MorphologicalEvaluator)
        from .morphological_eval import MorphologicalEvaluator
        morph_eval = MorphologicalEvaluator()
        test_words = morph_eval.create_morphological_test_set()

        morph_results = morph_eval.evaluate_morphological_preservation(tokenizer, test_words)
        total_morph_tests = sum(morph_results.values())

        if total_morph_tests > 0:
            metrics['morphological_correctness'] = (
                morph_results['correct_segmentation'] / total_morph_tests
            )
        else:
            metrics['morphological_correctness'] = 0.0

        return metrics