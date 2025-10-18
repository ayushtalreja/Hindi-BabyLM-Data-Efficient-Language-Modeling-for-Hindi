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

    if '--debug-freekidsbooks' in sys.argv:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        print("\n" + "="*60)
        print("DEBUGGING FREEKIDSBOOKS SCRAPING")
        print("="*60)

        collector = ChildrensStoryCollector(max_stories=10, rate_limit_delay=0.5)
        base_url = "https://freekidsbooks.org"
        hindi_page_url = f"{base_url}/subject/files/foreign-language/hindi-stories/"

        # Test 1: Access main page
        print("\n[TEST 1] Accessing main Hindi stories page...")
        try:
            response = requests.get(
                hindi_page_url,
                timeout=15,
                headers={'User-Agent': collector.user_agent}
            )
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find story links
                story_links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if any(pattern in href for pattern in ['/book/', '/story/', 'hindi']):
                        full_url = urljoin(base_url, href)
                        if full_url not in story_links:
                            story_links.append(full_url)

                print(f"  Found {len(story_links)} story links")

                if story_links:
                    # Test 2: Try to scrape first story
                    test_url = story_links[0]
                    print(f"\n[TEST 2] Testing story scraping: {test_url}")

                    response = requests.get(
                        test_url,
                        timeout=15,
                        headers={'User-Agent': collector.user_agent}
                    )
                    print(f"  Status: {response.status_code}")

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')

                        # Debug each extraction method
                        print("\n  [Method 1] Looking for content div...")
                        content_div = soup.find('div', class_=['content', 'story-content', 'entry-content'])
                        if content_div:
                            text = content_div.get_text(separator=' ', strip=True)
                            print(f"    Found: {len(text)} chars")
                            print(f"    Preview: {text[:100]}...")
                        else:
                            print("    Not found")

                        print("\n  [Method 2] Looking for article tag...")
                        article = soup.find('article')
                        if article:
                            text = article.get_text(separator=' ', strip=True)
                            print(f"    Found: {len(text)} chars")
                            print(f"    Preview: {text[:100]}...")
                        else:
                            print("    Not found")

                        print("\n  [Method 3] Looking for paragraphs...")
                        paragraphs = soup.find_all('p')
                        print(f"    Found {len(paragraphs)} paragraphs")
                        if paragraphs:
                            long_ps = [p for p in paragraphs if len(p.get_text(strip=True)) > 20]
                            print(f"    Long paragraphs (>20 chars): {len(long_ps)}")
                            text = ' '.join([p.get_text(strip=True) for p in long_ps])
                            print(f"    Combined length: {len(text)} chars")
                            print(f"    Preview: {text[:100]}...")

                        # Test actual method
                        print("\n  [ACTUAL METHOD] Using scrape_freekidsbooks_story...")
                        story_data = collector.scrape_freekidsbooks_story(test_url)

                        if story_data:
                            print(f"    ✓ SUCCESS: {len(story_data['text'])} chars")
                            print(f"    Preview: {story_data['text'][:150]}...")

                            # Test Hindi detection
                            is_hindi = collector.is_hindi_text(story_data['text'])
                            print(f"    Hindi text check: {is_hindi}")
                        else:
                            print("    ✗ FAILED: scrape_freekidsbooks_story returned None")

                            # Investigate why
                            print("\n  [DEBUGGING] Checking failure reasons...")

                            # Try each method manually
                            story_text = None
                            content_div = soup.find('div', class_=['content', 'story-content', 'entry-content'])
                            if content_div:
                                story_text = content_div.get_text(separator=' ', strip=True)
                                print(f"    Method 1 text length: {len(story_text)}")

                            if not story_text or len(story_text) < 50:
                                article = soup.find('article')
                                if article:
                                    story_text = article.get_text(separator=' ', strip=True)
                                    print(f"    Method 2 text length: {len(story_text)}")

                            if not story_text or len(story_text) < 50:
                                paragraphs = soup.find_all('p')
                                story_text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
                                print(f"    Method 3 text length: {len(story_text)}")

                            if story_text:
                                print(f"    Total text length: {len(story_text)}")
                                if len(story_text) < 100:
                                    print(f"    ✗ Text too short (< 100 chars)")
                                else:
                                    is_hindi = collector.is_hindi_text(story_text)
                                    print(f"    Is Hindi text: {is_hindi}")
                                    if not is_hindi:
                                        # Show character breakdown
                                        devanagari_count = sum(1 for char in story_text if '\u0900' <= char <= '\u097F')
                                        total_chars = sum(1 for char in story_text if not char.isspace())
                                        print(f"    Devanagari chars: {devanagari_count}/{total_chars}")
                                        print(f"    Ratio: {devanagari_count/total_chars if total_chars > 0 else 0:.2%}")
                                        print(f"    Text sample: {story_text[:200]}")
                else:
                    print("  ✗ No story links found!")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

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
