"""
Tests for Classification Head Wrappers

This module tests that language models can be properly wrapped with classification heads
and that the output shapes are correct for evaluation tasks.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gpt_model import HindiGPTModel
from src.models.deberta_model import HindiDeBERTaModel
from src.models.classification_models import (
    GPTForSequenceClassification,
    DeBERTaForSequenceClassification,
    wrap_model_for_classification
)


class TestClassificationHeads:
    """Test classification head wrappers"""

    @pytest.fixture
    def gpt_config(self):
        """GPT model configuration"""
        return {
            'model_size': 'tiny',
            'hidden_size': 256,
            'num_layers': 2,
            'num_heads': 4,
            'max_length': 128,
            'dropout': 0.1
        }

    @pytest.fixture
    def deberta_config(self):
        """DeBERTa model configuration"""
        return {
            'model_size': 'tiny',
            'hidden_size': 256,
            'num_layers': 2,
            'num_heads': 4,
            'max_length': 128,
            'dropout': 0.1
        }

    def test_gpt_wrapper_output_shape(self, gpt_config):
        """Test that GPT wrapper produces correct output shape"""
        vocab_size = 1000
        num_classes = 3
        batch_size = 4
        seq_len = 32

        # Create base GPT model
        base_model = HindiGPTModel(vocab_size=vocab_size, config=gpt_config)

        # Wrap with classification head
        model = GPTForSequenceClassification(
            lm_model=base_model,
            num_classes=num_classes,
            hidden_size=gpt_config['hidden_size']
        )

        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check output shape
        assert outputs.logits.shape == (batch_size, num_classes), \
            f"Expected shape ({batch_size}, {num_classes}), got {outputs.logits.shape}"

    def test_deberta_wrapper_output_shape(self, deberta_config):
        """Test that DeBERTa wrapper produces correct output shape"""
        vocab_size = 1000
        num_classes = 2
        batch_size = 4
        seq_len = 32

        # Create base DeBERTa model
        base_model = HindiDeBERTaModel(vocab_size=vocab_size, config=deberta_config)

        # Wrap with classification head
        model = DeBERTaForSequenceClassification(
            lm_model=base_model,
            num_classes=num_classes,
            hidden_size=deberta_config['hidden_size']
        )

        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check output shape
        assert outputs.logits.shape == (batch_size, num_classes), \
            f"Expected shape ({batch_size}, {num_classes}), got {outputs.logits.shape}"

    def test_gpt_wrapper_with_labels(self, gpt_config):
        """Test that GPT wrapper computes loss correctly"""
        vocab_size = 1000
        num_classes = 3
        batch_size = 4
        seq_len = 32

        # Create wrapped model
        base_model = HindiGPTModel(vocab_size=vocab_size, config=gpt_config)
        model = GPTForSequenceClassification(
            lm_model=base_model,
            num_classes=num_classes,
            hidden_size=gpt_config['hidden_size']
        )

        # Create dummy input with labels
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Check that loss is computed
        assert outputs.loss is not None, "Loss should be computed when labels are provided"
        assert outputs.loss.item() > 0, "Loss should be positive"

    def test_deberta_wrapper_with_labels(self, deberta_config):
        """Test that DeBERTa wrapper computes loss correctly"""
        vocab_size = 1000
        num_classes = 2
        batch_size = 4
        seq_len = 32

        # Create wrapped model
        base_model = HindiDeBERTaModel(vocab_size=vocab_size, config=deberta_config)
        model = DeBERTaForSequenceClassification(
            lm_model=base_model,
            num_classes=num_classes,
            hidden_size=deberta_config['hidden_size']
        )

        # Create dummy input with labels
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Check that loss is computed
        assert outputs.loss is not None, "Loss should be computed when labels are provided"
        assert outputs.loss.item() > 0, "Loss should be positive"

    def test_wrap_model_for_classification_gpt(self, gpt_config):
        """Test the wrap_model_for_classification function for GPT"""
        vocab_size = 1000
        num_classes = 3
        batch_size = 4
        seq_len = 32

        # Create base model
        base_model = HindiGPTModel(vocab_size=vocab_size, config=gpt_config)

        # Wrap using factory function
        wrapped_model = wrap_model_for_classification(
            lm_model=base_model,
            model_type='gpt',
            num_classes=num_classes,
            hidden_size=gpt_config['hidden_size']
        )

        # Test forward pass
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        outputs = wrapped_model(input_ids=input_ids)

        assert outputs.logits.shape == (batch_size, num_classes)

    def test_wrap_model_for_classification_deberta(self, deberta_config):
        """Test the wrap_model_for_classification function for DeBERTa"""
        vocab_size = 1000
        num_classes = 2
        batch_size = 4
        seq_len = 32

        # Create base model
        base_model = HindiDeBERTaModel(vocab_size=vocab_size, config=deberta_config)

        # Wrap using factory function
        wrapped_model = wrap_model_for_classification(
            lm_model=base_model,
            model_type='deberta',
            num_classes=num_classes,
            hidden_size=deberta_config['hidden_size']
        )

        # Test forward pass
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        outputs = wrapped_model(input_ids=input_ids)

        assert outputs.logits.shape == (batch_size, num_classes)

    def test_predictions_are_valid_classes(self, gpt_config):
        """Test that predictions are valid class indices"""
        vocab_size = 1000
        num_classes = 3
        batch_size = 10
        seq_len = 32

        # Create wrapped model
        base_model = HindiGPTModel(vocab_size=vocab_size, config=gpt_config)
        model = GPTForSequenceClassification(
            lm_model=base_model,
            num_classes=num_classes,
            hidden_size=gpt_config['hidden_size']
        )
        model.eval()

        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Check that all predictions are valid class indices
        assert predictions.shape == (batch_size,), f"Expected shape ({batch_size},), got {predictions.shape}"
        assert torch.all((predictions >= 0) & (predictions < num_classes)), \
            "All predictions should be valid class indices"

    def test_batch_prediction_count(self, gpt_config):
        """Test that the number of predictions matches the batch size"""
        vocab_size = 1000
        num_classes = 3
        batch_size = 8
        seq_len = 64

        # Create wrapped model
        base_model = HindiGPTModel(vocab_size=vocab_size, config=gpt_config)
        model = GPTForSequenceClassification(
            lm_model=base_model,
            num_classes=num_classes,
            hidden_size=gpt_config['hidden_size']
        )
        model.eval()

        # Simulate evaluation loop
        predictions = []
        num_batches = 5

        with torch.no_grad():
            for _ in range(num_batches):
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
                outputs = model(input_ids=input_ids)
                batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                predictions.extend(batch_preds.tolist())

        # Check total number of predictions
        expected_total = num_batches * batch_size
        assert len(predictions) == expected_total, \
            f"Expected {expected_total} predictions, got {len(predictions)}"

    def test_different_pooling_strategies(self, deberta_config):
        """Test different pooling strategies for DeBERTa"""
        vocab_size = 1000
        num_classes = 2
        batch_size = 4
        seq_len = 32

        base_model = HindiDeBERTaModel(vocab_size=vocab_size, config=deberta_config)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        pooling_strategies = ['mean', 'max', 'first', 'last']

        for strategy in pooling_strategies:
            model = DeBERTaForSequenceClassification(
                lm_model=base_model,
                num_classes=num_classes,
                hidden_size=deberta_config['hidden_size'],
                pooling_strategy=strategy
            )

            outputs = model(input_ids=input_ids)
            assert outputs.logits.shape == (batch_size, num_classes), \
                f"Pooling strategy '{strategy}' produced wrong shape: {outputs.logits.shape}"

    def test_freeze_base_model(self, gpt_config):
        """Test that base model can be frozen"""
        vocab_size = 1000
        num_classes = 3

        base_model = HindiGPTModel(vocab_size=vocab_size, config=gpt_config)

        # Wrap with frozen base
        model = GPTForSequenceClassification(
            lm_model=base_model,
            num_classes=num_classes,
            hidden_size=gpt_config['hidden_size'],
            freeze_base=True
        )

        # Check that base model parameters are frozen
        for param in model.lm_model.parameters():
            assert not param.requires_grad, "Base model parameters should be frozen"

        # Check that classifier parameters are trainable
        for param in model.classifier.parameters():
            assert param.requires_grad, "Classifier parameters should be trainable"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
