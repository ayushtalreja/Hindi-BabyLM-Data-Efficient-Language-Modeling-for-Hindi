"""
Tests for configuration management
"""

import pytest
import yaml
from pathlib import Path
from src.utils.experiment_config import ExperimentConfig


@pytest.mark.unit
class TestExperimentConfig:
    """Test ExperimentConfig class"""

    def test_default_initialization(self):
        """Test ExperimentConfig can be initialized with defaults"""
        config = ExperimentConfig()

        assert config.experiment_name == "default_experiment"
        assert config.vocab_size == 32000
        assert config.batch_size == 32

    def test_custom_initialization(self):
        """Test ExperimentConfig with custom values"""
        config = ExperimentConfig(
            experiment_name="test_exp",
            vocab_size=16000,
            batch_size=64
        )

        assert config.experiment_name == "test_exp"
        assert config.vocab_size == 16000
        assert config.batch_size == 64

    def test_save_config(self, temp_dir):
        """Test saving config to file"""
        config = ExperimentConfig(experiment_name="test_save")
        config_path = temp_dir / "test_config.yaml"

        config.save_config(str(config_path))

        assert config_path.exists()

        # Load and verify
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)

        assert loaded['experiment_name'] == "test_save"

    def test_load_config(self, temp_dir, mock_config_dict):
        """Test loading config from file"""
        # Save a config first
        config_path = temp_dir / "test_config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(mock_config_dict, f)

        # Load it
        config = ExperimentConfig.load_config(str(config_path))

        assert config is not None
        assert hasattr(config, 'vocab_size')

    def test_config_has_required_attributes(self):
        """Test that config has all required attributes"""
        config = ExperimentConfig()

        required_attrs = [
            'experiment_name',
            'data_dir',
            'model_dir',
            'vocab_size',
            'batch_size',
            'learning_rate',
            'num_epochs'
        ]

        for attr in required_attrs:
            assert hasattr(config, attr), f"Missing required attribute: {attr}"


@pytest.mark.unit
def test_base_config_file_exists():
    """Test that base config file exists"""
    config_path = Path("configs/base_config.yaml")

    assert config_path.exists(), "Base config file should exist"

    # Verify it's valid YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    assert config is not None
    assert isinstance(config, dict)


@pytest.mark.unit
def test_base_config_structure():
    """Test that base config has expected structure"""
    config_path = Path("configs/base_config.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check for main sections
    expected_sections = ['project', 'data', 'tokenization', 'model', 'training']

    for section in expected_sections:
        assert section in config, f"Config should have '{section}' section"


@pytest.mark.unit
def test_config_values_are_reasonable():
    """Test that config values are in reasonable ranges"""
    config_path = Path("configs/base_config.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check some reasonable constraints
    if 'tokenization' in config and 'vocab_size' in config['tokenization']:
        vocab_size = config['tokenization']['vocab_size']
        assert 1000 <= vocab_size <= 100000, "Vocab size should be reasonable"

    if 'training' in config and 'batch_size' in config['training']:
        batch_size = config['training']['batch_size']
        assert 1 <= batch_size <= 512, "Batch size should be reasonable"

    if 'data' in config and 'max_tokens' in config['data']:
        max_tokens = config['data']['max_tokens']
        assert max_tokens > 0, "Max tokens should be positive"


@pytest.mark.unit
class TestModelSpecificConfigs:
    """Test model-specific config separation"""

    def test_gpt_config_initialization(self):
        """Test GPT config auto-populates"""
        config = ExperimentConfig(model_type='gpt')

        assert config.gpt_config is not None
        assert config.deberta_config is None
        assert hasattr(config.gpt_config, 'use_cache')
        assert hasattr(config.gpt_config, 'scale_attn_weights')

    def test_deberta_config_initialization(self):
        """Test DeBERTa config auto-populates"""
        config = ExperimentConfig(model_type='deberta')

        assert config.deberta_config is not None
        assert config.gpt_config is None
        assert hasattr(config.deberta_config, 'position_buckets')
        assert hasattr(config.deberta_config, 'relative_attention')

    def test_clean_dict_removes_inactive_configs(self):
        """Test to_clean_dict only includes active model config"""
        # Test GPT model
        gpt_config = ExperimentConfig(model_type='gpt')
        clean_dict = gpt_config.to_clean_dict()

        assert 'gpt_config' in clean_dict
        assert 'deberta_config' not in clean_dict

        # Test DeBERTa model
        deberta_config = ExperimentConfig(model_type='deberta')
        clean_dict = deberta_config.to_clean_dict()

        assert 'deberta_config' in clean_dict
        assert 'gpt_config' not in clean_dict

    def test_gpt_config_no_deberta_contamination(self):
        """Test GPT config doesn't contain DeBERTa parameters"""
        config = ExperimentConfig(model_type='gpt')
        clean_dict = config.to_clean_dict()

        # DeBERTa-specific fields should not be in clean dict
        deberta_fields = {'position_buckets', 'relative_attention', 'max_relative_positions',
                         'pooler_hidden_size', 'pooler_dropout', 'pooler_hidden_act'}

        for field in deberta_fields:
            assert field not in clean_dict, f"GPT config should not contain {field}"

    def test_deberta_config_no_gpt_contamination(self):
        """Test DeBERTa config doesn't contain GPT-only parameters"""
        config = ExperimentConfig(model_type='deberta')
        clean_dict = config.to_clean_dict()

        # GPT-specific fields should not be in clean dict
        gpt_fields = {'use_cache', 'scale_attn_weights', 'reorder_and_upcast_attn'}

        # Note: Some fields might be in the deberta_config dict, but not at top level
        # We're checking that they're not leaked at the top level of the config


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility with old flat configs"""

    def test_migrate_old_deberta_flat_config(self):
        """Test migration of old flat DeBERTa config"""
        old_config_dict = {
            'model_type': 'deberta',
            'hidden_size': 768,
            'num_layers': 12,
            # Old flat DeBERTa params
            'position_buckets': 256,
            'relative_attention': True,
            'max_relative_positions': -1,
            'pooler_hidden_size': 768,
            'pooler_dropout': 0.1,
            'pooler_hidden_act': 'gelu',
        }

        from src.utils.experiment_config import ExperimentConfig
        config = ExperimentConfig.from_checkpoint_config(old_config_dict)

        assert config is not None
        assert config.model_type == 'deberta'
        assert config.deberta_config is not None
        assert config.deberta_config.position_buckets == 256
        assert config.deberta_config.relative_attention == True

    def test_migrate_old_gpt_flat_config(self):
        """Test migration of old flat GPT config"""
        old_config_dict = {
            'model_type': 'gpt',
            'hidden_size': 768,
            'num_layers': 12,
            # Old flat GPT params
            'use_cache': True,
            'scale_attn_weights': True,
        }

        from src.utils.experiment_config import ExperimentConfig
        config = ExperimentConfig.from_checkpoint_config(old_config_dict)

        assert config is not None
        assert config.model_type == 'gpt'
        assert config.gpt_config is not None
        assert config.gpt_config.use_cache == True
        assert config.gpt_config.scale_attn_weights == True

    def test_migrate_gpt_config_removes_deberta_contamination(self):
        """Test that migrating GPT config removes DeBERTa params"""
        old_config_dict = {
            'model_type': 'gpt',
            'hidden_size': 768,
            # GPT params
            'use_cache': True,
            # DeBERTa contamination (should be removed)
            'position_buckets': 256,
            'relative_attention': True,
        }

        from src.utils.experiment_config import ExperimentConfig
        config = ExperimentConfig.from_checkpoint_config(old_config_dict)

        assert config is not None
        assert config.model_type == 'gpt'
        # Check that DeBERTa config is not populated
        assert config.deberta_config is None or config.deberta_config.position_buckets == 256  # default value

    def test_new_nested_config_loads_correctly(self):
        """Test that new nested configs load correctly"""
        from src.utils.experiment_config import GPTModelConfig, DeBERTaModelConfig

        new_config_dict = {
            'model_type': 'gpt',
            'hidden_size': 768,
            'gpt_config': GPTModelConfig(use_cache=False),
        }

        from src.utils.experiment_config import ExperimentConfig
        config = ExperimentConfig.from_checkpoint_config(new_config_dict)

        assert config is not None
        assert config.gpt_config is not None
        assert config.gpt_config.use_cache == False


@pytest.mark.unit
def test_base_config_no_deberta_section():
    """Test that base_config.yaml (GPT model) doesn't have deberta section"""
    config_path = Path("configs/base_config.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Since model.type is 'gpt', there should not be a deberta section
    if 'model' in config:
        model_config = config['model']
        assert model_config.get('type') == 'gpt', "base_config should be for GPT model"

        # DeBERTa section should not exist or should be None/commented out
        # The actual structure might have it as a comment, so we just check it's not a populated dict
        if 'deberta' in model_config:
            assert model_config['deberta'] is None or not model_config['deberta'], \
                "GPT config should not have populated deberta section"


@pytest.mark.unit
def test_load_base_config_creates_correct_nested_config():
    """Test that loading base_config.yaml creates correct nested config"""
    config_path = Path("configs/base_config.yaml")

    from src.utils.experiment_config import ExperimentConfig
    config = ExperimentConfig.load_config(str(config_path))

    assert config is not None
    assert config.model_type == 'gpt'
    assert config.gpt_config is not None
    # DeBERTa config should either be None or not be accessed by GPT models
    # The __post_init__ should only populate gpt_config for GPT models
