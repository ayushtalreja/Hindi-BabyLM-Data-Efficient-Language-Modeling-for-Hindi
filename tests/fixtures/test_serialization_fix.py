#!/usr/bin/env python3
"""
Test script to verify the JSON serialization fix for evaluation results.
This tests that GPTModelConfig and other dataclass objects can be properly serialized.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.experiment_config import GPTModelConfig, DeBERTaModelConfig
from src.evaluation.evaluation_manager import EvaluationManager, DataclassJSONEncoder


def test_dataclass_serialization():
    """Test that dataclass objects can be serialized"""
    print("Testing dataclass serialization...")

    # Test GPTModelConfig
    gpt_config = GPTModelConfig(
        use_cache=True,
        scale_attn_weights=True,
        reorder_and_upcast_attn=False
    )

    # Test DeBERTaModelConfig
    deberta_config = DeBERTaModelConfig(
        position_buckets=256,
        relative_attention=True,
        max_relative_positions=512,
        pooler_hidden_size=768,
        pooler_dropout=0.1,
        pooler_hidden_act="gelu"
    )

    # Test direct serialization with custom encoder
    try:
        gpt_json = json.dumps(gpt_config, cls=DataclassJSONEncoder, indent=2)
        print("✓ GPTModelConfig serialized successfully")
        print(f"  Result: {gpt_json[:100]}...")
    except Exception as e:
        print(f"✗ GPTModelConfig serialization failed: {e}")
        return False

    try:
        deberta_json = json.dumps(deberta_config, cls=DataclassJSONEncoder, indent=2)
        print("✓ DeBERTaModelConfig serialized successfully")
        print(f"  Result: {deberta_json[:100]}...")
    except Exception as e:
        print(f"✗ DeBERTaModelConfig serialization failed: {e}")
        return False

    return True


def test_nested_config_serialization():
    """Test serialization of nested config dictionaries"""
    print("\nTesting nested config serialization...")

    # Simulate config.__dict__ structure as passed to EvaluationManager
    config_dict = {
        'experiment_name': 'test_experiment',
        'model_type': 'gpt',
        'hidden_size': 768,
        'num_layers': 12,
        'gpt_config': GPTModelConfig(
            use_cache=True,
            scale_attn_weights=True,
            reorder_and_upcast_attn=False
        ),
        'batch_size': 32,
        'learning_rate': 5e-4,
    }

    # Test using _make_serializable
    try:
        serializable = EvaluationManager._make_serializable(config_dict)
        json_str = json.dumps(serializable, indent=2)
        print("✓ Nested config serialized successfully with _make_serializable")
        print(f"  Keys: {list(serializable.keys())}")
        print(f"  gpt_config type: {type(serializable.get('gpt_config'))}")
    except Exception as e:
        print(f"✗ Nested config serialization failed: {e}")
        return False

    # Test using custom encoder directly
    try:
        json_str = json.dumps(config_dict, cls=DataclassJSONEncoder, indent=2)
        print("✓ Nested config serialized successfully with DataclassJSONEncoder")
    except Exception as e:
        print(f"✗ Nested config serialization with encoder failed: {e}")
        return False

    return True


def test_evaluation_summary():
    """Test the full evaluation summary structure"""
    print("\nTesting evaluation summary structure...")

    # Simulate the summary as created by generate_summary()
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'model_config': {
            'experiment_name': 'test_experiment',
            'model_type': 'gpt',
            'gpt_config': GPTModelConfig(
                use_cache=True,
                scale_attn_weights=True,
                reorder_and_upcast_attn=False
            ),
            'deberta_config': None,
        },
        'overall_scores': {
            'indicglue_avg': 0.75,
            'multiblimp_accuracy': 0.82
        }
    }

    # Test with _make_serializable (as used in fixed code)
    try:
        serializable_summary = EvaluationManager._make_serializable(summary)
        json_str = json.dumps(serializable_summary, indent=2, ensure_ascii=False)
        print("✓ Full evaluation summary serialized successfully")
        print(f"  Summary size: {len(json_str)} bytes")

        # Verify it can be deserialized
        reloaded = json.loads(json_str)
        print(f"✓ Summary can be deserialized successfully")
        print(f"  Reloaded keys: {list(reloaded.keys())}")
    except Exception as e:
        print(f"✗ Full evaluation summary serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_results_structure():
    """Test the complete results structure as saved by save_results()"""
    print("\nTesting complete results structure...")

    # Simulate the self.results structure
    results = {
        'indicglue': {
            'task1': {'accuracy': 0.75, 'f1': 0.73},
            'task2': {'accuracy': 0.82, 'f1': 0.80},
        },
        'multiblimp': {
            'overall': {'average_accuracy': 0.85},
            'phenomenon1': {'accuracy': 0.83},
            'phenomenon2': {'accuracy': 0.87},
        },
        'summary': {
            'evaluation_date': datetime.now().isoformat(),
            'model_config': {
                'experiment_name': 'test_experiment',
                'model_type': 'gpt',
                'hidden_size': 768,
                'gpt_config': GPTModelConfig(
                    use_cache=True,
                    scale_attn_weights=True,
                    reorder_and_upcast_attn=False
                ),
            },
            'overall_scores': {
                'indicglue_avg': 0.785,
                'multiblimp_accuracy': 0.85
            }
        }
    }

    # Test serialization as done in save_results()
    try:
        serializable_results = EvaluationManager._make_serializable(results)
        json_str = json.dumps(serializable_results, indent=2, ensure_ascii=False, cls=DataclassJSONEncoder)
        print("✓ Complete results structure serialized successfully")
        print(f"  Results size: {len(json_str)} bytes")

        # Verify deserialization
        reloaded = json.loads(json_str)
        print(f"✓ Results can be deserialized successfully")
        print(f"  Top-level keys: {list(reloaded.keys())}")
        print(f"  Summary has model_config: {'model_config' in reloaded.get('summary', {})}")
    except Exception as e:
        print(f"✗ Complete results serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests"""
    print("="*80)
    print("JSON Serialization Fix - Test Suite")
    print("="*80)

    tests = [
        test_dataclass_serialization,
        test_nested_config_serialization,
        test_evaluation_summary,
        test_results_structure,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\n✓ All tests passed! The serialization fix is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
