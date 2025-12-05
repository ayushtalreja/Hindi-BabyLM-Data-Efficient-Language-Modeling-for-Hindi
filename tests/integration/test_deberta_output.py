"""
Test script to investigate DeBERTa output shape for MultiBLiMP evaluation
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.deberta_model import HindiDeBERTaModel
from transformers import AutoTokenizer

def test_deberta_output():
    """Test DeBERTa model output shape"""

    print("=" * 60)
    print("Testing DeBERTa Model Output for MultiBLiMP")
    print("=" * 60)

    # Create a small test model
    vocab_size = 32000
    config = {
        'model_size': 'tiny',  # Use tiny for quick testing
        'max_length': 128
    }

    print("\n1. Creating DeBERTa model...")
    model = HindiDeBERTaModel(vocab_size, config)
    model.eval()

    print(f"   Model type: {type(model)}")
    print(f"   Inner model type: {type(model.model)}")

    # Create dummy inputs
    print("\n2. Creating test inputs...")
    test_sentence = "यह एक परीक्षण वाक्य है"

    # Simulate tokenization (we'll use dummy token IDs)
    input_ids = torch.tensor([[1, 100, 200, 300, 400, 2]])  # [batch=1, seq_len=6]
    attention_mask = torch.ones_like(input_ids)

    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Input IDs: {input_ids}")

    # Forward pass WITHOUT labels
    print("\n3. Running forward pass WITHOUT labels...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    print(f"   Output type: {type(outputs)}")
    print(f"   Output attributes: {dir(outputs)}")

    # Check if outputs has logits
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
        print(f"\n4. Logits information:")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Logits dimensions: {logits.dim()}")
        print(f"   Expected shape: [batch=1, seq_len=6, vocab_size={vocab_size}]")
    else:
        print(f"\n4. No 'logits' attribute found!")
        print(f"   Trying to access outputs[0]...")
        logits = outputs[0]
        print(f"   outputs[0] shape: {logits.shape}")
        print(f"   outputs[0] dimensions: {logits.dim()}")

    # Test the slicing operation that MultiBLiMP uses
    print(f"\n5. Testing MultiBLiMP slicing operations:")
    try:
        # This is what MultiBLiMP does
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        print(f"   shift_logits shape: {shift_logits.shape}")
        print(f"   shift_labels shape: {shift_labels.shape}")
        print(f"   ✓ Slicing operations successful!")

        # Test the view operation
        print(f"\n6. Testing view operation:")
        viewed_logits = shift_logits.view(-1, shift_logits.size(-1))
        viewed_labels = shift_labels.view(-1)

        print(f"   viewed_logits shape: {viewed_logits.shape}")
        print(f"   viewed_labels shape: {viewed_labels.shape}")
        print(f"   ✓ View operations successful!")

        # Test loss computation
        print(f"\n7. Testing loss computation:")
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(viewed_logits, viewed_labels)
        print(f"   Loss: {loss.item()}")
        print(f"   ✓ Loss computation successful!")

    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

    # Test with forward pass WITH labels
    print(f"\n8. Testing forward pass WITH labels:")
    with torch.no_grad():
        outputs_with_labels = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

    if hasattr(outputs_with_labels, 'loss'):
        print(f"   Has loss: {outputs_with_labels.loss}")
    if hasattr(outputs_with_labels, 'logits'):
        print(f"   Logits shape: {outputs_with_labels.logits.shape}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_deberta_output()
