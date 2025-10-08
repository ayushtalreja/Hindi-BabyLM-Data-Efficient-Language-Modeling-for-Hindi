import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel

class HindiGPTModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=config.get('max_length', 512),
            n_embd=config.get('hidden_size', 768),
            n_layer=config.get('num_layers', 12),
            n_head=config.get('num_heads', 12),
            resid_pdrop=config.get('dropout', 0.1),
            embd_pdrop=config.get('dropout', 0.1),
            attn_pdrop=config.get('dropout', 0.1),
        )
        
        self.model = GPT2LMHeadModel(self.config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def generate(self, input_ids, max_length=50, temperature=1.0):
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.config.pad_token_id
        )