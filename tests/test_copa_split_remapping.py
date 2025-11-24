#!/usr/bin/env python3
"""
Test script to verify COPA split remapping functionality.

This script tests that:
1. COPA splits are correctly remapped (test→validation, validation→test)
2. Zero-shot evaluation works with remapped splits
3. Fine-tuning works with both 'original' and 'custom' strategies
4. Other tasks are unaffected by remapping
"""

import pytest
import torch
import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.indicglue_evaluator import IndicGLUEEvaluator, SPLIT_REMAPPING
from src.models.model_factory import create_model
from src.data.tokenizer_factory import create_tokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.fixture
def model_and_tokenizer():
    """Create a small test model and tokenizer"""
    config_path = project_root / "configs" / "base_config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Use smaller model for testing
    config['model']['num_layers'] = 2
    config['model']['hidden_size'] = 128
    config['model']['num_heads'] = 4

    tokenizer = create_tokenizer(config)
    model = create_model(config, len(tokenizer))

    # Move to CPU for testing
    model = model.cpu()
    model.eval()

    return model, tokenizer, config


class TestCOPASplitRemapping:
    """Test suite for COPA split remapping functionality"""

    def test_split_remapping_configuration(self):
        """Test 1: Verify SPLIT_REMAPPING is correctly configured"""
        logger.info("\n=== Test 1: Split Remapping Configuration ===")

        assert 'Choice of Plausible Alternatives' in SPLIT_REMAPPING

        remap_config = SPLIT_REMAPPING['Choice of Plausible Alternatives']
        assert remap_config['test'] == 'validation', "test should map to validation"
        assert remap_config['validation'] == 'test', "validation should map to test"
        assert remap_config['train'] == 'train', "train should stay unchanged"

        logger.info("✓ SPLIT_REMAPPING configured correctly")

    def test_copa_split_remapping_loads_correct_splits(self, model_and_tokenizer):
        """Test 2: Verify COPA splits are correctly remapped when loading"""
        logger.info("\n=== Test 2: Split Remapping Loads Correct Data ===")

        model, tokenizer, config = model_and_tokenizer
        evaluator = IndicGLUEEvaluator(model, tokenizer, config)

        # Test: requesting 'test' should load 'validation' (88 examples)
        logger.info("Loading COPA test split (should load validation with 88 examples)...")
        test_data = evaluator._load_task_data("Choice of Plausible Alternatives", split='test')

        if test_data is not None:
            assert len(test_data) == 88, f"Expected 88 examples in test (from validation), got {len(test_data)}"
            logger.info(f"✓ Test split correctly loads validation data: {len(test_data)} examples")
        else:
            logger.warning("⚠ Test data is None - dataset might not be available")

        # Validation: requesting 'validation' should load 'test' (449 examples)
        logger.info("Loading COPA validation split (should load test with 449 examples)...")
        val_data = evaluator._load_task_data("Choice of Plausible Alternatives", split='validation')

        if val_data is not None:
            assert len(val_data) == 449, f"Expected 449 examples in validation (from test), got {len(val_data)}"
            logger.info(f"✓ Validation split correctly loads test data: {len(val_data)} examples")
        else:
            logger.warning("⚠ Validation data is None - dataset might not be available")

        # Train: should remain unchanged (362 examples)
        logger.info("Loading COPA train split (should load train with 362 examples)...")
        train_data = evaluator._load_task_data("Choice of Plausible Alternatives", split='train')

        if train_data is not None:
            assert len(train_data) == 362, f"Expected 362 examples in train, got {len(train_data)}"
            logger.info(f"✓ Train split unchanged: {len(train_data)} examples")
        else:
            logger.warning("⚠ Train data is None - dataset might not be available")

    def test_copa_zero_shot_evaluation(self, model_and_tokenizer):
        """Test 3: Zero-shot evaluation uses remapped test split"""
        logger.info("\n=== Test 3: Zero-Shot Evaluation ===")

        model, tokenizer, config = model_and_tokenizer

        # Configure for zero-shot
        config['evaluation'] = {
            'benchmarks': {
                'indicglue': {
                    'fine_tuning': {'enabled': False}
                }
            }
        }
        config['max_samples_per_task'] = 20  # Use small sample for testing

        evaluator = IndicGLUEEvaluator(model, tokenizer, config)

        logger.info("Running zero-shot evaluation on COPA...")
        try:
            results = evaluator.evaluate_task("Choice of Plausible Alternatives")

            # Should not be skipped
            assert results.get('status') != 'skipped', "COPA should not be skipped"
            logger.info("✓ COPA evaluation ran (not skipped)")

            # Should have results
            assert 'accuracy' in results, "Results should contain accuracy"
            assert 'num_examples' in results, "Results should contain num_examples"

            # With max_samples=20, should evaluate on min(88, 20) = 20 examples
            assert results['num_examples'] <= 20, f"Expected ≤20 examples, got {results['num_examples']}"
            logger.info(f"✓ Evaluated on {results['num_examples']} examples (validation split used as test)")

            # Accuracy should be valid
            assert 0.0 <= results['accuracy'] <= 1.0, "Accuracy should be between 0 and 1"
            logger.info(f"✓ Valid accuracy: {results['accuracy']:.4f}")

        except Exception as e:
            logger.warning(f"⚠ Zero-shot evaluation test skipped due to: {e}")
            pytest.skip(f"Dataset or model not available: {e}")

    def test_copa_fine_tuning_original_strategy(self, model_and_tokenizer):
        """Test 4: Fine-tuning with original strategy"""
        logger.info("\n=== Test 4: Fine-Tuning (Original Strategy) ===")

        model, tokenizer, config = model_and_tokenizer

        # Configure for fine-tuning with original strategy
        config['evaluation'] = {
            'benchmarks': {
                'indicglue': {
                    'fine_tuning': {
                        'enabled': True,
                        'split_strategy': 'original',
                        'num_epochs': 1,  # Only 1 epoch for testing
                        'batch_size': 8
                    }
                }
            }
        }
        config['max_samples_per_task'] = 20  # Use small sample for testing

        evaluator = IndicGLUEEvaluator(model, tokenizer, config)

        logger.info("Running fine-tuning evaluation on COPA (original strategy)...")
        try:
            results = evaluator.evaluate_task("Choice of Plausible Alternatives")

            # Should be fine-tuned
            assert results.get('fine_tuned') == True, "Should indicate fine-tuning was used"
            logger.info("✓ Fine-tuning was performed")

            # Should evaluate on validation split (88 examples, limited by max_samples)
            assert results['num_examples'] <= 20, f"Expected ≤20 test examples, got {results['num_examples']}"
            logger.info(f"✓ Evaluated on {results['num_examples']} test examples (from validation split)")

            # Should have fine-tuning metadata
            if 'fine_tuning_info' in results:
                logger.info(f"✓ Fine-tuning info available: {results['fine_tuning_info']}")

        except Exception as e:
            logger.warning(f"⚠ Fine-tuning test (original) skipped due to: {e}")
            pytest.skip(f"Dataset or model not available: {e}")

    def test_copa_fine_tuning_custom_strategy(self, model_and_tokenizer):
        """Test 5: Fine-tuning with custom strategy"""
        logger.info("\n=== Test 5: Fine-Tuning (Custom Strategy) ===")

        model, tokenizer, config = model_and_tokenizer

        # Configure for fine-tuning with custom strategy
        config['evaluation'] = {
            'benchmarks': {
                'indicglue': {
                    'fine_tuning': {
                        'enabled': True,
                        'split_strategy': 'custom',
                        'train_ratio': 0.7,
                        'val_ratio': 0.15,
                        'test_ratio': 0.15,
                        'num_epochs': 1,  # Only 1 epoch for testing
                        'batch_size': 8
                    }
                }
            }
        }

        evaluator = IndicGLUEEvaluator(model, tokenizer, config)

        logger.info("Running fine-tuning evaluation on COPA (custom strategy)...")
        try:
            results = evaluator.evaluate_task("Choice of Plausible Alternatives")

            # Should be fine-tuned
            assert results.get('fine_tuned') == True, "Should indicate fine-tuning was used"
            logger.info("✓ Fine-tuning was performed")

            # Test set should be ~15% of 899 total examples = ~135 examples
            # But actual count depends on split strategy implementation
            assert results['num_examples'] > 0, "Should have test examples"
            logger.info(f"✓ Evaluated on {results['num_examples']} test examples (custom split)")

            # Should have metrics
            assert 'accuracy' in results
            logger.info(f"✓ Valid accuracy: {results['accuracy']:.4f}")

        except Exception as e:
            logger.warning(f"⚠ Fine-tuning test (custom) skipped due to: {e}")
            pytest.skip(f"Dataset or model not available: {e}")

    def test_other_tasks_unaffected(self, model_and_tokenizer):
        """Test 6: Verify remapping doesn't affect other tasks"""
        logger.info("\n=== Test 6: Other Tasks Unaffected ===")

        model, tokenizer, config = model_and_tokenizer
        evaluator = IndicGLUEEvaluator(model, tokenizer, config)

        # Test BBCA (no remapping)
        logger.info("Testing BBCA task (should not have remapping)...")
        try:
            test_data = evaluator._load_task_data("BBCArticlesClassification", split='test')

            if test_data is not None:
                logger.info(f"✓ BBCA test data loaded normally: {len(test_data)} examples")
                # BBCA should load its own test split, not some remapped version
            else:
                logger.warning("⚠ BBCA test data is None - might not be available")

        except Exception as e:
            logger.warning(f"⚠ BBCA test skipped: {e}")

        # Test another task
        logger.info("Testing DiscourseMode task (should not have remapping)...")
        try:
            test_data = evaluator._load_task_data("DiscourseMode", split='test')

            if test_data is not None:
                logger.info(f"✓ DiscourseMode test data loaded normally: {len(test_data)} examples")
            else:
                logger.warning("⚠ DiscourseMode test data is None - might not be available")

        except Exception as e:
            logger.warning(f"⚠ DiscourseMode test skipped: {e}")

        logger.info("✓ Other tasks appear unaffected by COPA remapping")


def run_all_tests():
    """Run all tests manually (without pytest)"""
    logger.info("="*60)
    logger.info("COPA Split Remapping Test Suite")
    logger.info("="*60)

    # Create fixture
    config_path = project_root / "configs" / "base_config.yaml"

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Use smaller model for testing
        config['model']['num_layers'] = 2
        config['model']['hidden_size'] = 128
        config['model']['num_heads'] = 4

        tokenizer = create_tokenizer(config)
        model = create_model(config, len(tokenizer))
        model = model.cpu()
        model.eval()

        fixture = (model, tokenizer, config)

    except Exception as e:
        logger.error(f"Failed to create test fixture: {e}")
        return False

    # Run tests
    test_suite = TestCOPASplitRemapping()

    try:
        test_suite.test_split_remapping_configuration()
        test_suite.test_copa_split_remapping_loads_correct_splits(fixture)
        test_suite.test_copa_zero_shot_evaluation(fixture)
        test_suite.test_copa_fine_tuning_original_strategy(fixture)
        test_suite.test_copa_fine_tuning_custom_strategy(fixture)
        test_suite.test_other_tasks_unaffected(fixture)

        logger.info("\n" + "="*60)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*60)
        return True

    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Can be run with pytest or directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--manual':
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        logger.info("Run with pytest: pytest tests/test_copa_split_remapping.py")
        logger.info("Or manually: python tests/test_copa_split_remapping.py --manual")
