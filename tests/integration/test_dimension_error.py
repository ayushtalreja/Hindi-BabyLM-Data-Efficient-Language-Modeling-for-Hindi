"""
Test to reproduce the dimension error with DeBERTa classification wrapper
"""

import torch
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import importlib.util

# Import MultiBLiMPEvaluator directly
spec = importlib.util.spec_from_file_location(
    "multiblimp_evaluator",
    os.path.join(os.path.dirname(__file__), 'src', 'evaluation', 'multiblimp_evaluator.py')
)
multiblimp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multiblimp_module)
MultiBLiMPEvaluator = multiblimp_module.MultiBLiMPEvaluator

from models.deberta_model import HindiDeBERTaModel
from models.classification_models import DeBERTaForSequenceClassification

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_with_base_model():
    """Test with unwrapped base model - should work"""
    print("\n" + "="*80)
    print("Test 1: Base DeBERTa Model (should work)")
    print("="*80)

    vocab_size = 32000
    config = {'model_size': 'tiny', 'max_length': 128}

    # Create base model
    model = HindiDeBERTaModel(vocab_size, config)
    model.eval()

    # Create a simple tokenizer mock
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            # Return dummy tokens
            return {
                'input_ids': torch.tensor([[1, 100, 200, 300, 2]]),
                'attention_mask': torch.ones(1, 5)
            }

        def encode(self, text):
            return [1, 100, 200, 300, 2]

    tokenizer = MockTokenizer()

    try:
        # Create evaluator
        evaluator = MultiBLiMPEvaluator(model, tokenizer, config={})
        print("✓ Evaluator created successfully")

        # Test a minimal pair
        good_sentence = "यह एक अच्छा वाक्य है"
        bad_sentence = "यह एक बुरा वाक्य है"

        result = evaluator._evaluate_minimal_pair_detailed(good_sentence, bad_sentence)
        print(f"✓ Evaluation successful: is_correct={result[0]}")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_with_classification_wrapper():
    """Test with classification wrapper - should auto-unwrap"""
    print("\n" + "="*80)
    print("Test 2: Classification-Wrapped DeBERTa (should auto-unwrap)")
    print("="*80)

    vocab_size = 32000
    config = {'model_size': 'tiny', 'max_length': 128, 'hidden_size': 384}

    # Create base model
    base_model = HindiDeBERTaModel(vocab_size, config)

    # Wrap for classification
    wrapped_model = DeBERTaForSequenceClassification(
        lm_model=base_model,
        num_classes=3,
        hidden_size=384,
        pooling_strategy='mean'
    )
    wrapped_model.eval()

    # Create a simple tokenizer mock
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {
                'input_ids': torch.tensor([[1, 100, 200, 300, 2]]),
                'attention_mask': torch.ones(1, 5)
            }

        def encode(self, text):
            return [1, 100, 200, 300, 2]

    tokenizer = MockTokenizer()

    try:
        # Create evaluator - should detect and unwrap
        evaluator = MultiBLiMPEvaluator(wrapped_model, tokenizer, config={})
        print(f"✓ Evaluator created successfully")
        print(f"  Model type after init: {type(evaluator.model)}")

        # Test a minimal pair
        good_sentence = "यह एक अच्छा वाक्य है"
        bad_sentence = "यह एक बुरा वाक्य है"

        result = evaluator._evaluate_minimal_pair_detailed(good_sentence, bad_sentence)
        print(f"✓ Evaluation successful: is_correct={result[0]}")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_classification_output_dimensions():
    """Test what dimensions the classification wrapper actually outputs"""
    print("\n" + "="*80)
    print("Test 3: Check Classification Wrapper Output Dimensions")
    print("="*80)

    vocab_size = 32000
    config = {'model_size': 'tiny', 'max_length': 128, 'hidden_size': 384}

    # Create base model
    base_model = HindiDeBERTaModel(vocab_size, config)

    # Wrap for classification
    wrapped_model = DeBERTaForSequenceClassification(
        lm_model=base_model,
        num_classes=3,
        hidden_size=384,
        pooling_strategy='mean'
    )
    wrapped_model.eval()

    # Test inputs
    input_ids = torch.tensor([[1, 100, 200, 300, 2]])
    attention_mask = torch.ones_like(input_ids)

    print("Testing wrapped model output...")
    with torch.no_grad():
        outputs = wrapped_model(input_ids=input_ids, attention_mask=attention_mask)

    print(f"  Output type: {type(outputs)}")
    print(f"  Has 'logits': {hasattr(outputs, 'logits')}")

    if hasattr(outputs, 'logits'):
        print(f"  Logits shape: {outputs.logits.shape}")
        print(f"  Logits dim: {outputs.logits.dim()}")
        print(f"  ✗ PROBLEM: Classification wrapper outputs 2D logits!")

    print(f"\n  Has 'hidden_states': {hasattr(outputs, 'hidden_states')}")
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        print(f"  Hidden states shape: {outputs.hidden_states.shape}")

    # Try to access the base model logits
    print("\nTesting base model (unwrapped) output...")
    with torch.no_grad():
        base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)

    base_logits = base_outputs.logits if hasattr(base_outputs, 'logits') else base_outputs[0]
    print(f"  Base logits shape: {base_logits.shape}")
    print(f"  Base logits dim: {base_logits.dim()}")
    print(f"  ✓ Base model outputs 3D logits correctly!")


if __name__ == "__main__":
    test_with_base_model()
    test_with_classification_wrapper()
    test_classification_output_dimensions()
