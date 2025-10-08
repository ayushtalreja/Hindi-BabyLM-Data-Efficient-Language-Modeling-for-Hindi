import yaml
import os
from dataclasses import dataclass
from typing import Dict, List, Any
import wandb

@dataclass
class ExperimentConfig:
    # Data configuration
    max_tokens: int = 10_000_000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Tokenization configuration  
    tokenizer_type: str = "sentencepiece"
    vocab_size: int = 32000
    
    # Model configuration
    model_type: str = "gpt"  # gpt, bert, hybrid
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_length: int = 512
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Curriculum learning
    use_curriculum: bool = False
    curriculum_strategy: str = "morphological"  # morphological, length, random
    
    # Evaluation configuration
    eval_steps: int = 500
    save_steps: int = 1000
    
    def save_config(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class ExperimentManager:
    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.experiments = []
    
    def create_tokenization_experiments(self) -> List[ExperimentConfig]:
        """Create experiments for different tokenization strategies"""
        tokenizers = ["sentencepiece", "wordpiece", "bpe"]
        experiments = []
        
        for tokenizer in tokenizers:
            config = self.base_config.__class__(**self.base_config.__dict__)
            config.tokenizer_type = tokenizer
            config.experiment_name = f"tokenization_{tokenizer}"
            experiments.append(config)
        
        return experiments
    
    def create_model_architecture_experiments(self) -> List[ExperimentConfig]:
        """Create experiments for different model architectures"""
        architectures = ["gpt", "bert", "hybrid"]
        experiments = []
        
        for arch in architectures:
            config = self.base_config.__class__(**self.base_config.__dict__)
            config.model_type = arch
            config.experiment_name = f"architecture_{arch}"
            experiments.append(config)
        
        return experiments
    
    def create_curriculum_experiments(self) -> List[ExperimentConfig]:
        """Create experiments for curriculum learning strategies"""
        strategies = ["morphological", "length", "random", "none"]
        experiments = []
        
        for strategy in strategies:
            config = self.base_config.__class__(**self.base_config.__dict__)
            config.use_curriculum = strategy != "none"
            config.curriculum_strategy = strategy
            config.experiment_name = f"curriculum_{strategy}"
            experiments.append(config)
        
        return experiments