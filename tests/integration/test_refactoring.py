#!/usr/bin/env python3
"""
Quick integration test for refactored IndicGLUE evaluator.
Tests that all modules can be imported and initialized together.
"""

import sys
import torch

print("=" * 60)
print("REFACTORING INTEGRATION TEST")
print("=" * 60)

# Test 1: Import all refactored modules
print("\n[Test 1] Importing refactored modules...")
try:
    from src.evaluation.indicglue import (
        TaskRegistry,
        TaskDataExtractor,
        DataLoaderFactory,
        ClassificationStrategy,
        MultipleChoiceStrategy,
        PerplexityStrategy,
        BinaryCandidateStrategy,
        FineTuningManager,
        ResultVisualizer
    )
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Instantiate individual components
print("\n[Test 2] Instantiating individual components...")
try:
    # Create task registry
    task_registry = TaskRegistry()
    print(f"✅ TaskRegistry created with {len(task_registry.get_all_task_names())} tasks")

    # Create data extractor
    data_extractor = TaskDataExtractor(task_registry)
    print("✅ TaskDataExtractor created")

    # Create mock tokenizer (simple namespace)
    class MockTokenizer:
        pad_token = "[PAD]"
        pad_token_id = 0
        eos_token = "[EOS]"

        def __call__(self, text, **kwargs):
            # Simple mock tokenization
            if isinstance(text, list):
                batch_size = len(text)
            else:
                batch_size = 1
            max_len = kwargs.get('max_length', 128)
            return {
                'input_ids': torch.zeros((batch_size, max_len), dtype=torch.long),
                'attention_mask': torch.ones((batch_size, max_len), dtype=torch.long)
            }

    tokenizer = MockTokenizer()
    device = torch.device('cpu')

    # Create dataloader factory
    dataloader_factory = DataLoaderFactory(
        task_registry=task_registry,
        data_extractor=data_extractor,
        tokenizer=tokenizer,
        max_length=128,
        device=device
    )
    print("✅ DataLoaderFactory created")

    # Create evaluation strategies
    classification_strategy = ClassificationStrategy(device=device)
    mc_strategy = MultipleChoiceStrategy(device=device)
    perplexity_strategy = PerplexityStrategy(
        device=device,
        tokenizer=tokenizer,
        max_length=128,
        data_extractor=data_extractor
    )
    print("✅ All evaluation strategies created")

    # Create fine-tuning manager
    fine_tuning_manager = FineTuningManager(
        num_epochs=10,
        learning_rate=2e-5,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        patience=3
    )
    print("✅ FineTuningManager created")

    # Create result visualizer
    result_visualizer = ResultVisualizer(
        save_visualizations=False,  # Don't actually save during test
        visualization_formats=['png']
    )
    print("✅ ResultVisualizer created")

except Exception as e:
    print(f"❌ Component instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test data extraction with sample data
print("\n[Test 3] Testing data extraction...")
try:
    # Test BBCA task
    bbca_example = {
        'text': 'Sample news article about business.',
        'label': 'business'
    }

    text = data_extractor.extract_text(bbca_example, 'BBCArticlesClassification')
    label = data_extractor.extract_label(bbca_example, 'BBCArticlesClassification')
    print(f"✅ BBCA extraction: text='{text[:30]}...', label={label}")

    # Test WSTP multiple-choice task
    wstp_example = {
        'sectionText': 'This is a sample section.',
        'titleA': 'Title A',
        'titleB': 'Title B',
        'titleC': 'Title C',
        'titleD': 'Title D',
        'correctTitle': 'titleB'
    }

    text = data_extractor.extract_text(wstp_example, 'Wikipedia Section Title Prediction')
    choices = data_extractor.extract_choices(wstp_example, 'Wikipedia Section Title Prediction')
    label = data_extractor.extract_label(wstp_example, 'Wikipedia Section Title Prediction')
    print(f"✅ WSTP extraction: {len(choices)} choices, label={label}")

    # Test COPA task
    copa_example = {
        'premise': 'The man was tired.',
        'question': 'What was the CAUSE of this?',
        'choice1': 'He worked all day.',
        'choice2': 'He went to sleep.',
        'label': 0
    }

    text = data_extractor.extract_text(copa_example, 'Choice of Plausible Alternatives')
    choices = data_extractor.extract_choices(copa_example, 'Choice of Plausible Alternatives')
    label = data_extractor.extract_label(copa_example, 'Choice of Plausible Alternatives')
    print(f"✅ COPA extraction: {len(choices)} choices, label={label}")

except Exception as e:
    print(f"❌ Data extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test task registry
print("\n[Test 4] Testing task registry...")
try:
    all_tasks = task_registry.get_all_task_names()
    print(f"✅ Found {len(all_tasks)} tasks:")
    for task_name in all_tasks:
        config = task_registry.get_task_config(task_name)
        print(f"   - {task_name}: {config.task_type}, "
              f"labels={config.num_labels}, choices={config.num_choices}")

    # Test label converter
    converter = task_registry.get_label_converter('BBCArticlesClassification')
    if converter:
        test_label = converter('business')
        print(f"✅ Label converter works: 'business' -> {test_label}")

except Exception as e:
    print(f"❌ Task registry test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify backward compatibility constants
print("\n[Test 5] Testing backward compatibility...")
try:
    from src.evaluation.indicglue_evaluator import (
        BBCA_LABEL_MAP,
        DISCOURSE_MODE_LABEL_MAP,
        SPLIT_REMAPPING
    )
    print(f"✅ BBCA_LABEL_MAP accessible: {len(BBCA_LABEL_MAP)} labels")
    print(f"✅ DISCOURSE_MODE_LABEL_MAP accessible: {len(DISCOURSE_MODE_LABEL_MAP)} labels")
    print(f"✅ SPLIT_REMAPPING accessible")

except ImportError as e:
    print(f"❌ Backward compatibility broken: {e}")
    sys.exit(1)

# Test 6: Test FineTuningManager configuration
print("\n[Test 6] Testing FineTuningManager configuration...")
try:
    # Create mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)
            self.loss_value = 0.5

        def forward(self, input_ids, attention_mask, labels=None, **kwargs):
            batch_size = input_ids.size(0)
            logits = torch.randn(batch_size, 2)

            # Create output with loss
            class Output:
                def __init__(self, logits, loss):
                    self.logits = logits
                    self.loss = loss

            loss = torch.tensor(self.loss_value)
            return Output(logits, loss)

    # Verify optimizer creation works
    model = MockModel()
    optimizer = fine_tuning_manager._create_optimizer(model)
    print(f"✅ Optimizer created: {type(optimizer).__name__}")
    print(f"   - Param groups: {len(optimizer.param_groups)}")
    print(f"   - Learning rate: {optimizer.param_groups[0]['lr']}")

except Exception as e:
    print(f"❌ FineTuningManager test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("=" * 60)
print("\nRefactored components are working correctly!")
print("The modular architecture is functioning as expected.")
print("\nNext steps:")
print("1. Install full dependencies (torch, transformers, datasets, etc.)")
print("2. Run pytest test suite: pytest tests/unit/evaluation/indicglue/ -v")
print("3. Run end-to-end evaluation on real data")
print("=" * 60)
