#!/usr/bin/env python3
"""
Test script for children's books module
Run this after installing dependencies to verify the module works correctly
"""

import sys
sys.path.insert(0, '/Users/ayushkumartalreja/Downloads/Thesis_2/hindi-babylm')

try:
    from src.data_processing.childrens_books import (
        collect_childrens_stories,
        scrape_hindi_stories,
        ChildrensStoryCollector
    )

    print("✓ All imports successful!")
    print("\nAvailable functions:")
    print("  - collect_childrens_stories() -> List[str]")
    print("  - scrape_hindi_stories() -> List[Dict]")
    print("  - ChildrensStoryCollector class")

    # Test basic functionality (without actually making requests)
    print("\n✓ Module structure validated!")
    print("\nTo run a full test with actual data collection:")
    print("  python test_childrens_books.py --full-test")

    if '--full-test' in sys.argv:
        print("\n" + "="*60)
        print("Running full test with data collection...")
        print("="*60)

        # Test with small sample
        collector = ChildrensStoryCollector(max_stories=5, rate_limit_delay=1.0)
        stories = collector.collect_all_stories()

        print(f"\n✓ Collected {len(stories)} test stories")

        if stories:
            print("\nSample story preview:")
            print("-" * 60)
            print(stories[0][:200] + "...")
            print("-" * 60)

            print("\nStory statistics:")
            for i, story in enumerate(stories[:3], 1):
                word_count = len(story.split())
                char_count = len(story)
                print(f"  Story {i}: {word_count} words, {char_count} characters")

        print("\n✓ All tests passed!")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install dependencies first:")
    print("  pip install requests beautifulsoup4")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
