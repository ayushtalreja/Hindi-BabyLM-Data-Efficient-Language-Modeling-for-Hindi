"""
Test to verify the dimension error fix works correctly
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

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


class MockModel2D(torch.nn.Module):
    """Mock model that returns 2D logits (should be caught by our fix)"""

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0)
        num_classes = 3  # Classification output

        # Create mock output with 2D logits
        class MockOutput:
            def __init__(self):
                self.logits = torch.randn(batch_size, num_classes)  # 2D!

        return MockOutput()

    def parameters(self):
        return iter([torch.tensor([1.0])])


class MockModel3D(torch.nn.Module):
    """Mock model that returns 3D logits (correct)"""

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        vocab_size = 32000

        # Create mock output with 3D logits
        class MockOutput:
            def __init__(self):
                self.logits = torch.randn(batch_size, seq_len, vocab_size)  # 3D ✓

        return MockOutput()

    def parameters(self):
        return iter([torch.tensor([1.0])])


class MockModel1Token(torch.nn.Module):
    """Mock model that returns logits for only 1 token (should be caught)"""

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0)
        vocab_size = 32000

        # Create mock output with seq_len=1 (too short)
        class MockOutput:
            def __init__(self):
                self.logits = torch.randn(batch_size, 1, vocab_size)  # seq_len=1

        return MockOutput()

    def parameters(self):
        return iter([torch.tensor([1.0])])


class MockTokenizer:
    def __call__(self, text, **kwargs):
        return {
            'input_ids': torch.tensor([[1, 100, 200, 300, 2]]),
            'attention_mask': torch.ones(1, 5)
        }

    def encode(self, text):
        return [1, 100, 200, 300, 2]


def test_2d_logits_detection():
    """Test that 2D logits are properly detected and rejected"""
    print("\n" + "="*80)
    print("Test 1: Detect 2D Logits (should catch dimension error)")
    print("="*80)

    model = MockModel2D()
    tokenizer = MockTokenizer()

    try:
        evaluator = MultiBLiMPEvaluator(model, tokenizer, config={})

        # Try to evaluate a pair - should fail with clear error
        good_sentence = "यह एक अच्छा वाक्य है"
        bad_sentence = "यह एक बुरा वाक्य है"

        result = evaluator._evaluate_minimal_pair_detailed(good_sentence, bad_sentence)
        print(f"✗ FAIL: Should have raised ValueError for 2D logits, but got result: {result}")
        return False

    except ValueError as e:
        error_msg = str(e)
        if "3D logits" in error_msg and "2D tensor" in error_msg:
            print(f"✓ PASS: Correctly caught 2D logits error")
            print(f"  Error message: {error_msg[:200]}...")
            return True
        else:
            print(f"✗ FAIL: Got ValueError but wrong message: {error_msg}")
            return False

    except Exception as e:
        print(f"✗ FAIL: Got unexpected exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3d_logits_work():
    """Test that 3D logits work correctly"""
    print("\n" + "="*80)
    print("Test 2: 3D Logits Work Correctly")
    print("="*80)

    model = MockModel3D()
    tokenizer = MockTokenizer()

    try:
        evaluator = MultiBLiMPEvaluator(model, tokenizer, config={})

        good_sentence = "यह एक अच्छा वाक्य है"
        bad_sentence = "यह एक बुरा वाक्य है"

        result = evaluator._evaluate_minimal_pair_detailed(good_sentence, bad_sentence)
        print(f"✓ PASS: 3D logits evaluated successfully")
        print(f"  Result: is_correct={result[0]}, good_loss={result[1]:.4f}, bad_loss={result[2]:.4f}")
        return True

    except Exception as e:
        print(f"✗ FAIL: 3D logits should work but got error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_short_sequence_handling():
    """Test that sequences with only 1 token are handled gracefully"""
    print("\n" + "="*80)
    print("Test 3: Short Sequence Handling (seq_len=1)")
    print("="*80)

    model = MockModel1Token()
    tokenizer = MockTokenizer()

    try:
        evaluator = MultiBLiMPEvaluator(model, tokenizer, config={})

        good_sentence = "यह एक अच्छा वाक्य है"
        bad_sentence = "यह एक बुरा वाक्य है"

        result = evaluator._evaluate_minimal_pair_detailed(good_sentence, bad_sentence)

        # Should return failure gracefully (False, inf, inf, 0.0)
        if result[0] == False and result[1] == float('inf'):
            print(f"✓ PASS: Short sequence handled gracefully")
            print(f"  Result: {result}")
            return True
        else:
            print(f"✗ FAIL: Expected graceful failure for short sequence, got: {result}")
            return False

    except Exception as e:
        print(f"✗ FAIL: Should handle short sequences gracefully but got error: {type(e).__name__}: {e}")
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("Testing MultiBLiMP Dimension Error Fix")
    print("="*80)

    results = []
    results.append(("2D Logits Detection", test_2d_logits_detection()))
    results.append(("3D Logits Work", test_3d_logits_work()))
    results.append(("Short Sequence Handling", test_short_sequence_handling()))

    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
