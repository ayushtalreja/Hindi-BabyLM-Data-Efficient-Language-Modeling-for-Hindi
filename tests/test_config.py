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
