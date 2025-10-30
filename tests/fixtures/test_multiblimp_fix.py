#!/usr/bin/env python3
"""
Test script to verify the MultiBLiMP evaluator fixes
"""

import sys
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path and import the module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from the module file to avoid package import issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "multiblimp_evaluator",
    os.path.join(os.path.dirname(__file__), 'src', 'evaluation', 'multiblimp_evaluator.py')
)
multiblimp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multiblimp_module)
MultiBLiMPEvaluator = multiblimp_module.MultiBLiMPEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_evaluator_initialization():
    """Test that the evaluator initializes correctly with the new phenomena names"""
    print("\n" + "="*80)
    print("Testing MultiBLiMP Evaluator Initialization")
    print("="*80)

    try:
        # Load a small model for testing (GPT-2 as a placeholder)
        print("\n1. Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        print("   ✓ Model loaded successfully")

        # Initialize evaluator
        print("\n2. Initializing MultiBLiMP evaluator...")
        evaluator = MultiBLiMPEvaluator(model, tokenizer)
        print("   ✓ Evaluator initialized successfully")

        # Check phenomena names
        print("\n3. Verifying phenomena names...")
        expected_phenomena = ['SV-#', 'SV-G', 'SV-P', 'SP-#', 'SP-G']

        print(f"   Expected phenomena: {expected_phenomena}")
        print(f"   Actual phenomena:   {evaluator.phenomena}")

        assert evaluator.phenomena == expected_phenomena, "Phenomena names don't match!"
        print("   ✓ Phenomena names are correct")

        # Check minimal pairs loaded
        print("\n4. Checking minimal pairs...")
        total_pairs = sum(len(pairs) for pairs in evaluator.minimal_pairs.values())
        print(f"   Total minimal pairs loaded: {total_pairs}")

        print("\n   Breakdown by phenomenon:")
        for phenomenon in evaluator.phenomena:
            if phenomenon in evaluator.minimal_pairs:
                count = len(evaluator.minimal_pairs[phenomenon])
                print(f"     {phenomenon}: {count} pairs")
            else:
                print(f"     {phenomenon}: NOT FOUND (ERROR!)")

        assert total_pairs > 0, "No minimal pairs loaded!"
        print(f"\n   ✓ Successfully loaded {total_pairs} minimal pairs")

        # Verify all expected phenomena are present
        print("\n5. Verifying all phenomena are present in dataset...")
        missing_phenomena = []
        for phenomenon in expected_phenomena:
            if phenomenon not in evaluator.minimal_pairs:
                missing_phenomena.append(phenomenon)

        if missing_phenomena:
            print(f"   ✗ Missing phenomena: {missing_phenomena}")
            return False
        else:
            print("   ✓ All expected phenomena are present")

        # Test a single minimal pair evaluation
        print("\n6. Testing minimal pair evaluation...")
        test_phenomenon = 'SV-#'
        if test_phenomenon in evaluator.minimal_pairs and len(evaluator.minimal_pairs[test_phenomenon]) > 0:
            test_pair = evaluator.minimal_pairs[test_phenomenon][0]
            print(f"   Test pair:")
            print(f"     Good: {test_pair['good'][:80]}...")
            print(f"     Bad:  {test_pair['bad'][:80]}...")

            is_correct, good_loss, bad_loss, loss_diff = evaluator._evaluate_minimal_pair_detailed(
                test_pair['good'],
                test_pair['bad']
            )

            print(f"   Results:")
            print(f"     Good sentence loss: {good_loss:.4f}")
            print(f"     Bad sentence loss:  {bad_loss:.4f}")
            print(f"     Loss difference:    {loss_diff:.4f}")
            print(f"     Model correct:      {is_correct}")
            print("   ✓ Evaluation works correctly")

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluator_initialization()
    sys.exit(0 if success else 1)
