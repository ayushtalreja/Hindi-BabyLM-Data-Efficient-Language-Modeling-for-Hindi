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
    max_words: int = 10_000_000  # Maximum words in corpus (renamed from max_tokens)
    max_tokens: int = None  # Deprecated: use max_words instead

    # Separate word limits for each split (Phase 2)
    train_word_limit: int = 10_000_000  # 10M words for training
    val_word_limit: int = 10_000_000    # 10M words for validation
    test_word_limit: int = 10_000_000   # 10M words for test

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Tokenization configuration
    tokenizer_type: str = "sentencepiece"
    vocab_size: int = 32000

    # Model configuration
    model_type: str = "gpt"  # gpt, deberta
    model_size: str = "small"  # tiny, small, medium/base, large
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_length: int = 512
    dropout: float = 0.1
    intermediate_size: int = 3072

    # DeBERTa-specific configuration
    position_buckets: int = 256
    relative_attention: bool = True
    max_relative_positions: int = -1
    pooler_hidden_size: int = 768
    pooler_dropout: float = 0.1
    pooler_hidden_act: str = "gelu"

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
            model_config = config_dict.get('model', {})
            # Add top-level model config
            if 'type' in model_config:
                flat_config['model_type'] = model_config['type']
            if 'model_size' in model_config:
                flat_config['model_size'] = model_config['model_size']
            if 'activation' in model_config:
                flat_config['activation'] = model_config['activation']

            # Extract from nested architecture section
            if 'architecture' in model_config:
                arch = model_config['architecture']
                flat_config.update(arch)
                if 'num_hidden_layers' in arch:
                    flat_config['num_layers'] = arch['num_hidden_layers']
                if 'num_attention_heads' in arch:
                    flat_config['num_heads'] = arch['num_attention_heads']
                if 'max_position_embeddings' in arch:
                    flat_config['max_length'] = arch['max_position_embeddings']

            # Extract DeBERTa-specific config
            if 'deberta' in model_config:
                deberta_config = model_config['deberta']
                for key in ['position_buckets', 'relative_attention', 'max_relative_positions',
                           'pooler_hidden_size', 'pooler_dropout', 'pooler_hidden_act']:
                    if key in deberta_config:
                        flat_config[key] = deberta_config[key]

            # Extract from regularization section
            if 'regularization' in model_config:
                reg = model_config['regularization']
                if 'hidden_dropout_prob' in reg:
                    flat_config['dropout'] = reg['hidden_dropout_prob']

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
        architectures = ["gpt", "deberta"]
        experiments = []

        for arch in architectures:
            config = self.base_config.__class__(**self.base_config.__dict__)
            config.model_type = arch
            config.experiment_name = f"architecture_{arch}"
            experiments.append(config)

        return experiments