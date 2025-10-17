import yaml
import os
from dataclasses import dataclass
from typing import Dict, List, Any
import wandb

@dataclass
class ExperimentConfig:
    # Experiment metadata
    experiment_name: str = "default_experiment"

    # Directory configuration
    data_dir: str = "data"
    model_dir: str = "models"
    tokenizer_dir: str = "tokenizers"
    results_dir: str = "results"

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
    dropout: float = 0.1
    intermediate_size: int = 3072

    # Training configuration
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 1000

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

        # Handle nested YAML structure
        flat_config = {}

        # Handle nested structure if it exists
        if 'data' in config_dict:
            flat_config.update(config_dict.get('data', {}))
        if 'tokenization' in config_dict:
            tokenization = config_dict.get('tokenization', {})
            if 'vocab_size' in tokenization:
                flat_config['vocab_size'] = tokenization['vocab_size']
            # Default to first method if multiple methods listed
            if 'methods' in tokenization and isinstance(tokenization['methods'], list):
                flat_config['tokenizer_type'] = tokenization['methods'][0]
        if 'training' in config_dict:
            training = config_dict.get('training', {})
            flat_config.update(training)
            # Map max_epochs to num_epochs if needed
            if 'max_epochs' in training:
                flat_config['num_epochs'] = training['max_epochs']
        if 'model' in config_dict:
            flat_config.update(config_dict.get('model', {}))

        # If config_dict is already flat (not nested), use it directly
        if not any(key in config_dict for key in ['data', 'tokenization', 'training', 'model']):
            flat_config = config_dict

        # Filter out keys that are not defined on the dataclass to avoid
        # TypeError: __init__() got an unexpected keyword argument '<key>' when
        # the YAML contains extra/unexpected fields (for example 'sources').
        if isinstance(flat_config, dict):
            allowed_keys = set(cls.__dataclass_fields__.keys())
            filtered_config = {k: v for k, v in flat_config.items() if k in allowed_keys}
        else:
            filtered_config = flat_config

        return cls(**filtered_config)

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