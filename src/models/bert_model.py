import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForMaskedLM

class HindiBERTModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=config.get('hidden_size', 768),
            num_hidden_layers=config.get('num_layers', 12),
            num_attention_heads=config.get('num_heads', 12),
            intermediate_size=config.get('intermediate_size', 3072),
            max_position_embeddings=config.get('max_length', 512),
            hidden_dropout_prob=config.get('dropout', 0.1),
            attention_probs_dropout_prob=config.get('dropout', 0.1),
        )
        
        self.model = BertForMaskedLM(self.config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)