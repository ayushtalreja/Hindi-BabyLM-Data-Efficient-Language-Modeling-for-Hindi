#!/usr/bin/env python3
"""
Test script for Wikipedia scraper with improvements
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processing.wiki_scraper import scrape_hindi_wikipedia

def test_scraper():
    """Test the improved Wikipedia scraper with a small dataset"""
    print("="*80)
    print("Testing Improved Wikipedia Scraper")
    print("="*80)

    # Test with 100 articles
    categories = ['विज्ञान', 'इतिहास']  # Just 2 categories for quick test
    max_articles = 100

    print(f"\nTest Configuration:")
    print(f"  Categories: {categories}")
    print(f"  Max Articles: {max_articles}")
    print(f"  Expected batches: ~{max_articles // 50} (50 articles per batch)")
    print("\n" + "="*80 + "\n")

    try:
        articles = scrape_hindi_wikipedia(categories, max_articles=max_articles)

        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"✅ Successfully scraped {len(articles)} articles")

        if len(articles) > 0:
            # Show sample article
            print(f"\nSample Article:")
            print(f"  Title: {articles[0]['title']}")
            print(f"  Category: {articles[0]['category']}")
            print(f"  Text length: {len(articles[0]['text'])} chars")
            print(f"  Text preview: {articles[0]['text'][:200]}...")

            # Show statistics
            total_chars = sum(len(article['text']) for article in articles)
            avg_chars = total_chars / len(articles)
            print(f"\nStatistics:")
            print(f"  Total characters: {total_chars:,}")
            print(f"  Average chars per article: {avg_chars:.0f}")

            # Count by category
            category_counts = {}
            for article in articles:
                cat = article['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1

            print(f"\nArticles per category:")
            for cat, count in category_counts.items():
                print(f"  {cat}: {count}")

        print("\n" + "="*80)
        print("✅ TEST PASSED")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    test_scraper()
