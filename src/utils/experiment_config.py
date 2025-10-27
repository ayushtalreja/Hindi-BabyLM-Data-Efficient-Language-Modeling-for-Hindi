import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import wandb


@dataclass
class GPTModelConfig:
    """Configuration specific to GPT models"""
    use_cache: bool = True
    scale_attn_weights: bool = True
    reorder_and_upcast_attn: bool = False


@dataclass
class DeBERTaModelConfig:
    """Configuration specific to DeBERTa models"""
    position_buckets: int = 256
    relative_attention: bool = True
    max_relative_positions: int = -1
    pooler_hidden_size: int = 768
    pooler_dropout: float = 0.1
    pooler_hidden_act: str = "gelu"


@dataclass
class ExperimentConfig:
    # Experiment metadata
    experiment_name: str = "default_experiment"
    experiment_description: str = ""
    experiment_tags: Optional[List[str]] = None

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

    # Model-specific configurations (only one should be populated based on model_type)
    gpt_config: Optional[GPTModelConfig] = None
    deberta_config: Optional[DeBERTaModelConfig] = None

    # Training configuration
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    # Evaluation configuration
    eval_steps: int = 500
    save_steps: int = 1000

    def __post_init__(self):
        """Auto-populate model-specific config based on model_type"""
        if self.model_type == 'gpt' and self.gpt_config is None:
            self.gpt_config = GPTModelConfig()
        elif self.model_type == 'deberta' and self.deberta_config is None:
            self.deberta_config = DeBERTaModelConfig()

    def get_model_specific_config(self):
        """Get the active model-specific configuration"""
        if self.model_type == 'gpt':
            return self.gpt_config
        elif self.model_type == 'deberta':
            return self.deberta_config
        else:
            return None

    def to_clean_dict(self):
        """
        Return a dictionary representation with only relevant model-specific params.
        This ensures saved configs don't contain cross-contamination.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            # Include the active model-specific config
            if key == 'gpt_config' and self.model_type == 'gpt' and value is not None:
                config_dict[key] = value.__dict__
            elif key == 'deberta_config' and self.model_type == 'deberta' and value is not None:
                config_dict[key] = value.__dict__
            # Skip the inactive model-specific config
            elif key in ('gpt_config', 'deberta_config'):
                continue
            else:
                config_dict[key] = value
        return config_dict

    def save_config(self, path: str):
        """Save configuration to YAML file (with clean model-specific params)"""
        with open(path, 'w') as f:
            yaml.dump(self.to_clean_dict(), f, default_flow_style=False)

    @classmethod
    def load_config(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Handle nested YAML structure
        flat_config = {}

        # Handle nested structure if it exists
        # Extract experiment configuration
        if 'experiment' in config_dict:
            exp_config = config_dict.get('experiment', {})
            if 'name' in exp_config:
                flat_config['experiment_name'] = exp_config['name']
            if 'description' in exp_config:
                flat_config['experiment_description'] = exp_config['description']
            if 'tags' in exp_config:
                flat_config['experiment_tags'] = exp_config['tags']

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

            # Extract model-specific configs and create nested dataclass instances
            if 'gpt' in model_config:
                gpt_params = model_config['gpt']
                flat_config['gpt_config'] = GPTModelConfig(**{
                    k: v for k, v in gpt_params.items()
                    if k in GPTModelConfig.__dataclass_fields__
                })

            if 'deberta' in model_config:
                deberta_params = model_config['deberta']
                flat_config['deberta_config'] = DeBERTaModelConfig(**{
                    k: v for k, v in deberta_params.items()
                    if k in DeBERTaModelConfig.__dataclass_fields__
                })

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

    @classmethod
    def from_checkpoint_config(cls, config_dict: Dict[str, Any]):
        """
        Create ExperimentConfig from a checkpoint config dict.
        Handles backward compatibility with old flat configs that have model-specific params.
        """
        if not isinstance(config_dict, dict):
            return None

        # Make a copy to avoid modifying the original
        config_dict = config_dict.copy()

        # Check if this is an old flat config with model-specific params mixed in
        model_type = config_dict.get('model_type', 'gpt')

        # Define old model-specific fields that should be migrated
        old_deberta_fields = {
            'position_buckets', 'relative_attention', 'max_relative_positions',
            'pooler_hidden_size', 'pooler_dropout', 'pooler_hidden_act'
        }
        old_gpt_fields = {
            'use_cache', 'scale_attn_weights', 'reorder_and_upcast_attn'
        }

        # Handle nested configs that are already dicts (from saved checkpoints)
        if 'gpt_config' in config_dict and isinstance(config_dict['gpt_config'], dict):
            config_dict['gpt_config'] = GPTModelConfig(**config_dict['gpt_config'])

        if 'deberta_config' in config_dict and isinstance(config_dict['deberta_config'], dict):
            config_dict['deberta_config'] = DeBERTaModelConfig(**config_dict['deberta_config'])

        # Check if we have old flat model-specific params
        has_old_deberta_params = any(k in config_dict for k in old_deberta_fields)
        has_old_gpt_params = any(k in config_dict for k in old_gpt_fields)

        # If we have old params, migrate them to nested structure
        if model_type == 'deberta' and has_old_deberta_params:
            # Create DeBERTa config from flat params
            deberta_params = {k: config_dict.pop(k) for k in old_deberta_fields if k in config_dict}
            config_dict['deberta_config'] = DeBERTaModelConfig(**deberta_params)
            print("⚠️  Migrated old flat DeBERTa config to nested structure")

        if model_type == 'gpt' and has_old_gpt_params:
            # Create GPT config from flat params
            gpt_params = {k: config_dict.pop(k) for k in old_gpt_fields if k in config_dict}
            config_dict['gpt_config'] = GPTModelConfig(**gpt_params)
            print("⚠️  Migrated old flat GPT config to nested structure")

        # Remove any remaining old model-specific params from other model types
        # (e.g., DeBERTa params in a GPT config)
        for field in old_deberta_fields | old_gpt_fields:
            config_dict.pop(field, None)

        # Filter to only allowed keys
        allowed_keys = set(cls.__dataclass_fields__.keys())
        filtered_config = {k: v for k, v in config_dict.items() if k in allowed_keys}

        return cls(**filtered_config)

    def get_tokenizer_path(self) -> Path:
        """Get the experiment-scoped tokenizer directory path"""
        return Path(self.results_dir) / self.experiment_name / 'tokenizer'

    def get_model_path(self) -> Path:
        """Get the experiment-scoped model directory path"""
        return Path(self.results_dir) / self.experiment_name / 'models'

    def get_results_path(self) -> Path:
        """Get the experiment-scoped results directory path"""
        return Path(self.results_dir) / self.experiment_name

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