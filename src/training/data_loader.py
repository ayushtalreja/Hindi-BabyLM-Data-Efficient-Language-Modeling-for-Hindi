import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from typing import List, Dict

class HindiLanguageModelingDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class CurriculumDataLoader:
    def __init__(self, curriculum_splits: List[List[str]], tokenizer, batch_size: int):
        self.curriculum_splits = curriculum_splits
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.current_stage = 0
    
    def get_current_dataloader(self) -> DataLoader:
        """Get dataloader for current curriculum stage"""
        dataset = HindiLanguageModelingDataset(
            self.curriculum_splits[self.current_stage], 
            self.tokenizer
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Set to True for BERT-style training
            )
        )
    
    def advance_stage(self):
        """Move to next curriculum stage"""
        if self.current_stage < len(self.curriculum_splits) - 1:
            self.current_stage += 1