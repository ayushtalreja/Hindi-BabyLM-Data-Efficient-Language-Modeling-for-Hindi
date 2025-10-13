"""
Tests for model architectures
"""

import pytest
import torch
from src.models.model_factory import ModelFactory
from src.models.gpt_model import HindiGPTModel


@pytest.mark.unit
@pytest.mark.model
class TestModelFactory:
    """Test ModelFactory class"""

    def test_factory_initialization(self, test_config):
        """Test ModelFactory can be initialized"""
        factory = ModelFactory(test_config)
        assert factory.model_type == test_config['model_type']

    def test_create_gpt_model(self, test_config):
        """Test GPT model creation"""
        test_config['model_type'] = 'gpt'
        factory = ModelFactory(test_config)

        model = factory.create_model(vocab_size=1000)

        assert model is not None
        assert isinstance(model, torch.nn.Module)

        # Check parameter count
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

    def test_model_info(self, test_config):
        """Test getting model information"""
        factory = ModelFactory(test_config)
        model = factory.create_model(vocab_size=1000)

        info = factory.get_model_info(model)

        assert 'model_type' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['total_parameters'] > 0


@pytest.mark.unit
@pytest.mark.model
class TestHindiGPTModel:
    """Test HindiGPTModel class"""

    def test_model_initialization(self):
        """Test model can be initialized"""
        config = {
            'hidden_size': 64,
            'num_layers': 2,
            'num_heads': 4,
            'max_length': 128
        }

        model = HindiGPTModel(vocab_size=1000, config=config)
        assert model is not None

    def test_model_forward_pass(self):
        """Test model forward pass"""
        config = {
            'hidden_size': 64,
            'num_layers': 2,
            'num_heads': 4,
            'max_length': 128
        }

        model = HindiGPTModel(vocab_size=1000, config=config)
        model.eval()

        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs is not None
        assert hasattr(outputs, 'logits')
        assert outputs.logits.shape[0] == batch_size
        assert outputs.logits.shape[1] == seq_length
        assert outputs.logits.shape[2] == 1000  # vocab_size

    def test_model_parameters_require_grad(self):
        """Test that model parameters require gradients"""
        config = {
            'hidden_size': 64,
            'num_layers': 2,
            'num_heads': 4,
            'max_length': 128
        }

        model = HindiGPTModel(vocab_size=1000, config=config)

        # Check that parameters require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0
