#!/usr/bin/env python3
"""
Debug script to inspect MultiBLiMP dataset phenomenon names

This script loads the HuggingFace 'jumelet/multiblimp' dataset
and prints all unique phenomenon names to understand the naming schema.
"""

from datasets import load_dataset
from collections import Counter
import sys

def inspect_multiblimp_dataset():
    """Load and inspect the MultiBLiMP dataset for Hindi"""

    print("=" * 80)
    print("MultiBLiMP Dataset Inspector")
    print("=" * 80)

    try:
        print("\n1. Loading dataset from HuggingFace (jumelet/multiblimp, language: hin)...")
        dataset = load_dataset('jumelet/multiblimp', 'hin', split='train')
        print(f"   ✓ Successfully loaded {len(dataset)} examples")

        # Inspect dataset structure
        print("\n2. Dataset structure:")
        if len(dataset) > 0:
            example = dataset[0]
            print(f"   Fields: {list(example.keys())}")
            print(f"\n   Sample example:")
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"     {key}: {value[:100]}...")
                else:
                    print(f"     {key}: {value}")

        # Count phenomenon names
        print("\n3. Extracting phenomenon names...")
        phenomena = []
        for example in dataset:
            phenomenon = example.get('phenomenon', 'UNKNOWN')
            phenomena.append(phenomenon)

        phenomenon_counts = Counter(phenomena)

        print(f"\n4. Unique phenomena found: {len(phenomenon_counts)}")
        print("\n   Phenomenon name : Count")
        print("   " + "-" * 60)
        for phenomenon, count in sorted(phenomenon_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {phenomenon:50s} : {count:4d}")

        # Compare with expected names
        print("\n5. Expected phenomenon names (from code):")
        expected_phenomena = [
            'subject_verb_agreement_number',
            'subject_verb_agreement_person',
            'subject_verb_agreement_gender',
            'case_marking_ergative',
            'case_marking_accusative',
            'case_marking_dative',
            'word_order',
            'gender_agreement_adjective',
            'gender_agreement_verb',
            'number_agreement',
            'honorific_agreement',
            'negation',
            'binding',
            'control'
        ]

        print("\n   Checking which expected phenomena are present in dataset:")
        print("   " + "-" * 60)
        for expected in expected_phenomena:
            if expected in phenomenon_counts:
                print(f"   ✓ {expected:50s} : {phenomenon_counts[expected]:4d} pairs")
            else:
                print(f"   ✗ {expected:50s} : NOT FOUND")

        # Look for potential mappings
        print("\n6. Potential mapping suggestions:")
        print("   " + "-" * 60)

        dataset_phenomena = set(phenomenon_counts.keys())
        expected_set = set(expected_phenomena)

        # Dataset phenomena not in expected list
        extra_in_dataset = dataset_phenomena - expected_set
        if extra_in_dataset:
            print("\n   Phenomena in dataset but not in code:")
            for phenom in sorted(extra_in_dataset):
                print(f"     • {phenom} ({phenomenon_counts[phenom]} pairs)")

        # Expected phenomena not in dataset
        missing_from_dataset = expected_set - dataset_phenomena
        if missing_from_dataset:
            print("\n   Phenomena expected by code but not in dataset:")
            for phenom in sorted(missing_from_dataset):
                print(f"     • {phenom}")

        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)

        return phenomenon_counts, expected_phenomena

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    inspect_multiblimp_dataset()
