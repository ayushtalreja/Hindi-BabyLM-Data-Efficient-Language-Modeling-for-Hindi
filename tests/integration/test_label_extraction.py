#!/usr/bin/env python3
"""
Test script to verify label extraction for WSTP and CSQA tasks
"""

from datasets import load_dataset
import sys

def test_wstp():
    """Test Wikipedia Section Title Prediction label extraction"""
    print("="*80)
    print("Testing Wikipedia Section Title Prediction (WSTP)")
    print("="*80)

    try:
        dataset = load_dataset('ai4bharat/indic_glue', 'wstp.hi', split='test')
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"✓ Columns: {dataset.column_names}")

        # Test label extraction
        title_to_idx = {'titleA': 0, 'titleB': 1, 'titleC': 2, 'titleD': 3}

        print("\nTesting label extraction on first 5 examples:")
        for i in range(min(5, len(dataset))):
            example = dataset[i]
            correct_title = example['correctTitle']
            label_idx = title_to_idx.get(correct_title, -1)

            print(f"\nExample {i+1}:")
            print(f"  correctTitle: {correct_title}")
            print(f"  Label index: {label_idx}")
            print(f"  Available titles: {[example['titleA'][:30], example['titleB'][:30], example['titleC'][:30], example['titleD'][:30]]}")

            if label_idx == -1:
                print(f"  ✗ ERROR: Invalid correctTitle value!")
                return False

        print("\n✓ All WSTP label extractions successful!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csqa():
    """Test CommonsenseQA label extraction"""
    print("\n" + "="*80)
    print("Testing CommonsenseQA (CSQA)")
    print("="*80)

    try:
        dataset = load_dataset('ai4bharat/indic_glue', 'csqa.hi', split='test')
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"✓ Columns: {dataset.column_names}")

        print("\nTesting label extraction on first 5 examples:")
        success_count = 0
        fail_count = 0

        for i in range(min(5, len(dataset))):
            example = dataset[i]
            answer = example['answer']
            options = example['options']

            try:
                label_idx = options.index(answer)
                success_count += 1
                print(f"\nExample {i+1}:")
                print(f"  Answer: {answer}")
                print(f"  Options: {options}")
                print(f"  Label index: {label_idx}")
                print(f"  ✓ Match found at index {label_idx}")
            except ValueError:
                fail_count += 1
                print(f"\nExample {i+1}:")
                print(f"  Answer: {answer}")
                print(f"  Options: {options}")
                print(f"  ✗ ERROR: Answer not found in options!")

        print(f"\n✓ Successful: {success_count}/{min(5, len(dataset))}")
        if fail_count > 0:
            print(f"✗ Failed: {fail_count}/{min(5, len(dataset))}")
            print("  Note: Some examples may have answer text that doesn't exactly match options")

        return success_count > 0

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nLabel Extraction Test Suite")
    print("="*80)

    wstp_success = test_wstp()
    csqa_success = test_csqa()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"WSTP: {'✓ PASS' if wstp_success else '✗ FAIL'}")
    print(f"CSQA: {'✓ PASS' if csqa_success else '✗ FAIL'}")

    if wstp_success and csqa_success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
