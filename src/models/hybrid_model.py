import torch
import torch.nn as nn
from transformers import BertModel, GPT2Model

class HybridGPTBERTModel(nn.Module):
    """Hybrid model combining causal and masked language modeling"""
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Shared embedding layer
        self.embeddings = nn.Embedding(vocab_size, config['hidden_size'])
        
        # BERT encoder for bidirectional understanding
        self.bert_encoder = BertModel(config['bert_config'])
        
        # GPT decoder for causal generation
        self.gpt_decoder = GPT2Model(config['gpt_config'])
        
        # Task-specific heads
        self.mlm_head = nn.Linear(config['hidden_size'], vocab_size)  # Masked LM
        self.lm_head = nn.Linear(config['hidden_size'], vocab_size)   # Causal LM
        
        self.loss_weights = {'mlm': 0.5, 'clm': 0.5}
    
    def forward(self, input_ids, attention_mask=None, task='both'):
        if task == 'mlm' or task == 'both':
            mlm_outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
            mlm_logits = self.mlm_head(mlm_outputs.last_hidden_state)
        
        if task == 'clm' or task == 'both':
            clm_outputs = self.gpt_decoder(input_ids=input_ids, attention_mask=attention_mask)
            clm_logits = self.lm_head(clm_outputs.last_hidden_state)
        
        if task == 'both':
            return {'mlm_logits': mlm_logits, 'clm_logits': clm_logits}
        elif task == 'mlm':
            return {'mlm_logits': mlm_logits}
        else:
            return {'clm_logits': clm_logits}