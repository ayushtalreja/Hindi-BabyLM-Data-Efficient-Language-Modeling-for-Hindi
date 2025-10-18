#!/usr/bin/env python3
"""
Standalone debug script for freekidsbooks scraping
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

def is_hindi_text(text: str) -> bool:
    """Check if text contains significant Hindi (Devanagari) content"""
    if not text or len(text) < 50:
        return False

    devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = sum(1 for char in text if not char.isspace())

    if total_chars == 0:
        return False

    hindi_ratio = devanagari_count / total_chars
    return hindi_ratio > 0.5  # At least 50% Devanagari

print("="*70)
print("DEBUGGING FREEKIDSBOOKS SCRAPING")
print("="*70)

base_url = "https://freekidsbooks.org"
hindi_page_url = f"{base_url}/subject/files/foreign-language/hindi-stories/"
user_agent = 'Mozilla/5.0 (Research Project) HindiBabyLM/1.0 (Linguistic Research)'

# Test 1: Access main page
print("\n[TEST 1] Accessing main Hindi stories page...")
print(f"URL: {hindi_page_url}")

try:
    response = requests.get(
        hindi_page_url,
        timeout=15,
        headers={'User-Agent': user_agent}
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
            print("\n  First 10 story links:")
            for i, url in enumerate(story_links[:10], 1):
                print(f"    {i}. {url}")

            # Filter out non-story URLs
            filtered_links = [
                url for url in story_links
                if not any(bad in url for bad in ['facebook.com', 'twitter.com', 'linkedin.com', 'subject/files'])
            ]
            print(f"\n  After filtering: {len(filtered_links)} actual story links")

            if filtered_links:
                # Test 2: Try to scrape first story
                test_url = filtered_links[0]
            else:
                print("  ✗ No valid story links after filtering!")
                test_url = None

            if test_url:
                print(f"\n[TEST 2] Testing story scraping")
                print(f"  URL: {test_url}")

                time.sleep(1)  # Rate limit

                response = requests.get(
                    test_url,
                    timeout=15,
                    headers={'User-Agent': user_agent}
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
                        print(f"    Hindi check: {is_hindi_text(text)}")
                    else:
                        print("    Not found")

                    print("\n  [Method 2] Looking for article tag...")
                    article = soup.find('article')
                    if article:
                        text = article.get_text(separator=' ', strip=True)
                        print(f"    Found: {len(text)} chars")
                        print(f"    Preview: {text[:100]}...")
                        print(f"    Hindi check: {is_hindi_text(text)}")
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
                        print(f"    Preview: {text[:150]}...")
                        print(f"    Hindi check: {is_hindi_text(text)}")

                        # Character analysis
                        if text:
                            devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
                            total_chars = sum(1 for char in text if not char.isspace())
                            ratio = devanagari_count / total_chars if total_chars > 0 else 0
                            print(f"    Devanagari ratio: {ratio:.2%} ({devanagari_count}/{total_chars})")

                    # Try to find actual story content more broadly
                    print("\n  [Method 4] Looking for main content area...")
                    main_content = soup.find(['main', 'div'], {'id': ['main', 'content', 'primary']})
                    if main_content:
                        text = main_content.get_text(separator=' ', strip=True)
                        print(f"    Found main: {len(text)} chars")
                        print(f"    Preview: {text[:150]}...")
                        print(f"    Hindi check: {is_hindi_text(text)}")

                    # Check for PDF/download links
                    print("\n  [Method 5] Looking for PDF/download links...")
                    pdf_links = soup.find_all('a', href=lambda href: href and ('.pdf' in href.lower() or 'download' in href.lower()))
                    print(f"    Found {len(pdf_links)} PDF/download links")
                    for i, link in enumerate(pdf_links[:5], 1):
                        href = link.get('href')
                        text = link.get_text(strip=True)[:50]
                        print(f"      {i}. {href} - {text}")

                    # Check for iframe/embedded content
                    print("\n  [Method 6] Looking for embedded content (iframe/embed)...")
                    iframes = soup.find_all(['iframe', 'embed', 'object'])
                    print(f"    Found {len(iframes)} embedded elements")
                    for i, frame in enumerate(iframes[:3], 1):
                        src = frame.get('src', frame.get('data', ''))
                        print(f"      {i}. {src}")

        else:
            print("  ✗ No story links found!")
            print("\n  Let's check what links do exist:")
            all_links = soup.find_all('a', href=True)
            print(f"  Total links: {len(all_links)}")
            print("\n  Sample links:")
            for i, link in enumerate(all_links[:10], 1):
                href = link.get('href', '')
                text = link.get_text(strip=True)[:40]
                print(f"    {i}. {href} - {text}")
    else:
        print(f"  ✗ Failed with status: {response.status_code}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DEBUG COMPLETE")
print("="*70)
