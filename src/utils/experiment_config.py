import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import wandb


@dataclass
class GPTModelConfig:
    """Configuration specific to GPT models"""
    use_cache: Optional[bool] = None
    scale_attn_weights: Optional[bool] = None
    reorder_and_upcast_attn: Optional[bool] = None


@dataclass
class DeBERTaModelConfig:
    """Configuration specific to DeBERTa models"""
    position_buckets: Optional[int] = None
    relative_attention: Optional[bool] = None
    max_relative_positions: Optional[int] = None
    pooler_hidden_size: Optional[int] = None
    pooler_dropout: Optional[float] = None
    pooler_hidden_act: Optional[str] = None


@dataclass
class ExperimentConfig:
    # Experiment metadata
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None
    experiment_tags: Optional[List[str]] = None

    # Directory configuration
    data_dir: Optional[str] = None
    model_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    results_dir: Optional[str] = None

    # Data configuration
    max_words: Optional[int] = None  # Maximum words in corpus (renamed from max_tokens)
    max_tokens: Optional[int] = None  # Deprecated: use max_words instead

    # Separate word limits for each split (Phase 2)
    train_word_limit: Optional[int] = None  # 10M words for training
    val_word_limit: Optional[int] = None    # 10M words for validation
    test_word_limit: Optional[int] = None   # 10M words for test

    train_ratio: Optional[float] = None
    val_ratio: Optional[float] = None
    test_ratio: Optional[float] = None

    # Tokenization configuration
    tokenizer_type: Optional[str] = None
    vocab_size: Optional[int] = None

    # Model configuration
    model_type: Optional[str] = None  # gpt, deberta
    model_size: Optional[str] = None  # tiny, small, medium/base, large
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    max_length: Optional[int] = None
    dropout: Optional[float] = None
    intermediate_size: Optional[int] = None

    # Model-specific configurations (only one should be populated based on model_type)
    gpt_config: Optional[GPTModelConfig] = None
    deberta_config: Optional[DeBERTaModelConfig] = None

    # Training configuration
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    num_epochs: Optional[int] = None
    weight_decay: Optional[float] = None
    warmup_steps: Optional[int] = None

    # Evaluation configuration
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None

    # Resource configuration
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    device: Optional[str] = None

    # Logging configuration
    log_steps: Optional[int] = None

    # Mixed precision configuration
    mixed_precision_dtype: Optional[str] = None

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

        # Extract directories configuration
        if 'directories' in config_dict:
            directories = config_dict.get('directories', {})
            if 'data_dir' in directories:
                flat_config['data_dir'] = directories['data_dir']
            if 'model_dir' in directories:
                flat_config['model_dir'] = directories['model_dir']
            if 'tokenizer_dir' in directories:
                flat_config['tokenizer_dir'] = directories['tokenizer_dir']
            if 'results_dir' in directories:
                flat_config['results_dir'] = directories['results_dir']

        if 'data' in config_dict:
            flat_config.update(config_dict.get('data', {}))
        if 'tokenization' in config_dict:
            tokenization = config_dict.get('tokenization', {})
            if 'vocab_size' in tokenization:
                flat_config['vocab_size'] = tokenization['vocab_size']
            # Map 'type' field to 'tokenizer_type'
            if 'type' in tokenization:
                flat_config['tokenizer_type'] = tokenization['type']
            # Default to first method if multiple methods listed (backward compatibility)
            elif 'methods' in tokenization and isinstance(tokenization['methods'], list):
                flat_config['tokenizer_type'] = tokenization['methods'][0]
        if 'training' in config_dict:
            training = config_dict.get('training', {})
            flat_config.update(training)
            # Map max_epochs to num_epochs if needed
            if 'max_epochs' in training:
                flat_config['num_epochs'] = training['max_epochs']
            # Extract optimizer parameters
            if 'optimizer' in training:
                optimizer = training['optimizer']
                if 'learning_rate' in optimizer:
                    flat_config['learning_rate'] = optimizer['learning_rate']
                if 'weight_decay' in optimizer:
                    flat_config['weight_decay'] = optimizer['weight_decay']
            # Extract learning rate scheduler parameters
            if 'lr_scheduler' in training:
                lr_scheduler = training['lr_scheduler']
                if 'warmup_steps' in lr_scheduler:
                    flat_config['warmup_steps'] = lr_scheduler['warmup_steps']
            # Extract evaluation parameters
            if 'evaluation' in training:
                evaluation = training['evaluation']
                if 'eval_steps' in evaluation:
                    flat_config['eval_steps'] = evaluation['eval_steps']
            # Extract checkpointing parameters
            if 'checkpointing' in training:
                checkpointing = training['checkpointing']
                if 'save_steps' in checkpointing:
                    flat_config['save_steps'] = checkpointing['save_steps']
            # Extract logging parameters
            if 'logging' in training:
                logging = training['logging']
                if 'log_steps' in logging:
                    flat_config['log_steps'] = logging['log_steps']
            # Extract mixed precision parameters
            if 'mixed_precision' in training:
                mixed_precision = training['mixed_precision']
                if 'dtype' in mixed_precision:
                    flat_config['mixed_precision_dtype'] = mixed_precision['dtype']

        # Extract resources configuration
        if 'resources' in config_dict:
            resources = config_dict.get('resources', {})
            if 'num_workers' in resources:
                flat_config['num_workers'] = resources['num_workers']
            if 'pin_memory' in resources:
                flat_config['pin_memory'] = resources['pin_memory']
            if 'device' in resources:
                flat_config['device'] = resources['device']

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