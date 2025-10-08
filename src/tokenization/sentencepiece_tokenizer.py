mport sentencepiece as spm
from typing import List, Dict

class HindiSentencePieceTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.model = None
    
    def train_tokenizer(self, corpus: List[str], model_prefix: str):
        """Train SentencePiece tokenizer on Hindi corpus"""
        # Create training data file
        training_file = f"{model_prefix}_training.txt"
        with open(training_file, 'w', encoding='utf-8') as f:
            for text in corpus:
                f.write(text + '\n')
        
        # Train tokenizer
        spm.SentencePieceTrainer.train(
            input=training_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=0.995,  # Important for Hindi
            model_type='bpe'
        )
        
        # Load trained model
        self.model = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using trained model"""
        return self.model.encode(text, out_type=str)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        return self.model.encode(text, out_type=int)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        return self.model.decode(ids)