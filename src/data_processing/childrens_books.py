import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_hindi_stories():
    """Collect Hindi children's stories from various sources"""
    import time

    sources = {
        "freekidsbooks": "https://freekidsbooks.org/subject/files/foreign-language/hindi-stories/",
        "storyweaver": "https://storyweaver.org.in/",
        # Add more sources
    }

    stories = []

    for source, url in sources.items():
        print(f"Scraping from {source}...")

        try:
            # Add delay to be respectful to servers
            time.sleep(2)

            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Research Project) HindiBabyLM/1.0'
            })

            if response.status_code != 200:
                print(f"Failed to fetch {source}: Status {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Source-specific extraction
            if source == "freekidsbooks":
                # Extract story links and content from Free Kids Books
                story_links = soup.find_all('a', href=True)
                for link in story_links:
                    href = link.get('href', '')
                    if 'hindi' in href.lower() or link.text:
                        # This is a simplified example - actual implementation
                        # would need to follow links and extract story text
                        text = link.get_text(strip=True)
                        if text and len(text) > 50:
                            stories.append({
                                'source': source,
                                'text': text,
                                'url': href
                            })

            elif source == "storyweaver":
                # Extract from StoryWeaver (placeholder - API access preferred)
                # Note: StoryWeaver has an API which should be used instead
                paragraphs = soup.find_all('p')
                story_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                if story_text and len(story_text) > 100:
                    stories.append({
                        'source': source,
                        'text': story_text,
                        'url': url
                    })

        except Exception as e:
            print(f"Error scraping {source}: {str(e)}")
            continue

    print(f"Collected {len(stories)} stories from {len(sources)} sources")
    return stories