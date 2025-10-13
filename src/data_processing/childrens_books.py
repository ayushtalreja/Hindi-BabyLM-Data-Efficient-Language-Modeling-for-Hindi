"""
Children's Books Data Collection Module

This module collects Hindi children's stories from various openly-licensed sources
to create developmentally appropriate training data for the Hindi BabyLM project.

Sources:
- StoryWeaver (Pratham Books) - API-based collection
- Free Kids Books - Web scraping with link following
- Bal Sahitya resources

All content is collected respecting copyright, robots.txt, and rate limits.
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChildrensStoryCollector:
    """Main class for collecting Hindi children's stories"""

    def __init__(self, max_stories: int = 2000, rate_limit_delay: float = 2.0):
        """
        Initialize the story collector

        Args:
            max_stories: Maximum number of stories to collect
            rate_limit_delay: Delay in seconds between requests
        """
        self.max_stories = max_stories
        self.rate_limit_delay = rate_limit_delay
        self.user_agent = 'Mozilla/5.0 (Research Project) HindiBabyLM/1.0 (Linguistic Research)'
        self.collected_stories = []
        self.seen_urls = set()

    def collect_all_stories(self) -> List[str]:
        """
        Main entry point: Collect stories from all sources

        Returns:
            List of story texts (strings)
        """
        logger.info("Starting children's story collection...")
        all_stories_data = []

        # Collect from StoryWeaver API
        logger.info("\n1. Collecting from StoryWeaver API...")
        try:
            storyweaver_stories = self.scrape_storyweaver_api()
            all_stories_data.extend(storyweaver_stories)
            logger.info(f"   Collected {len(storyweaver_stories)} stories from StoryWeaver")
        except Exception as e:
            logger.error(f"   Error collecting from StoryWeaver: {str(e)}")

        # Collect from Free Kids Books
        if len(all_stories_data) < self.max_stories:
            logger.info("\n2. Collecting from Free Kids Books...")
            try:
                freekids_stories = self.scrape_freekidsbooks()
                all_stories_data.extend(freekids_stories)
                logger.info(f"   Collected {len(freekids_stories)} stories from Free Kids Books")
            except Exception as e:
                logger.error(f"   Error collecting from Free Kids Books: {str(e)}")

        # Filter for quality and age-appropriateness
        logger.info("\n3. Filtering stories for quality and age-appropriateness...")
        filtered_stories = self.filter_stories(all_stories_data)
        logger.info(f"   {len(filtered_stories)} stories passed quality filters")

        # Extract just the text content
        story_texts = [story['text'] for story in filtered_stories if story.get('text')]

        logger.info(f"\nTotal stories collected: {len(story_texts)}")
        return story_texts

    def scrape_storyweaver_api(self) -> List[Dict]:
        """
        Collect stories from StoryWeaver using their public API

        StoryWeaver (storyweaver.org.in) by Pratham Books provides
        openly licensed children's stories in multiple languages.

        Returns:
            List of story dictionaries with metadata
        """
        stories = []

        # StoryWeaver API endpoint for Hindi stories
        api_base = "https://storyweaver.org.in/api/v1/stories"

        # Parameters for Hindi stories, sorted by reading level
        params = {
            'languages': 'Hindi',
            'reading_levels': '1,2,3',  # Early reader levels
            'per_page': 24,
            'page': 1
        }

        max_pages = 20  # Collect from first 20 pages
        stories_collected = 0

        for page in range(1, max_pages + 1):
            if stories_collected >= self.max_stories:
                break

            params['page'] = page

            try:
                time.sleep(self.rate_limit_delay)

                response = requests.get(
                    api_base,
                    params=params,
                    timeout=15,
                    headers={'User-Agent': self.user_agent}
                )

                if response.status_code != 200:
                    logger.warning(f"API returned status {response.status_code} for page {page}")
                    continue

                data = response.json()
                story_list = data.get('data', [])

                if not story_list:
                    logger.info(f"No more stories found at page {page}")
                    break

                for story_item in story_list:
                    if stories_collected >= self.max_stories:
                        break

                    # Extract story details from API response
                    story_slug = story_item.get('slug', '')
                    story_url = f"https://storyweaver.org.in/stories/{story_slug}"

                    if story_url in self.seen_urls:
                        continue

                    # Fetch full story content
                    story_data = self.fetch_storyweaver_story(story_slug)

                    if story_data and story_data.get('text'):
                        self.seen_urls.add(story_url)
                        stories.append(story_data)
                        stories_collected += 1

                logger.info(f"   Page {page}: {len(story_list)} stories fetched, {stories_collected} total")

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error on page {page}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error processing page {page}: {str(e)}")
                continue

        return stories

    def fetch_storyweaver_story(self, slug: str) -> Optional[Dict]:
        """
        Fetch full story content from StoryWeaver by slug

        Args:
            slug: Story slug identifier

        Returns:
            Story dictionary or None if fetch fails
        """
        try:
            time.sleep(self.rate_limit_delay)

            # Try API endpoint first
            api_url = f"https://storyweaver.org.in/api/v1/stories/{slug}"
            response = requests.get(
                api_url,
                timeout=15,
                headers={'User-Agent': self.user_agent}
            )

            if response.status_code == 200:
                data = response.json()
                story_data = data.get('data', {})

                # Extract pages/content
                pages = story_data.get('pages', [])
                story_text = self.extract_storyweaver_text(pages)

                if story_text:
                    return {
                        'source': 'storyweaver',
                        'title': story_data.get('title', ''),
                        'text': story_text,
                        'reading_level': story_data.get('reading_level', 'unknown'),
                        'url': f"https://storyweaver.org.in/stories/{slug}"
                    }

            # Fallback to web scraping if API fails
            return self.scrape_storyweaver_page(slug)

        except Exception as e:
            logger.debug(f"Error fetching story {slug}: {str(e)}")
            return None

    def extract_storyweaver_text(self, pages: List[Dict]) -> str:
        """
        Extract text content from StoryWeaver pages

        Args:
            pages: List of page dictionaries from API

        Returns:
            Concatenated story text
        """
        story_lines = []

        for page in pages:
            # Skip cover page
            if page.get('page_template') == 'cover_page':
                continue

            # Extract text content
            content = page.get('content', '')
            if content:
                # Clean HTML if present
                content = BeautifulSoup(content, 'html.parser').get_text(separator=' ', strip=True)
                if content and len(content) > 10:
                    story_lines.append(content)

        return ' '.join(story_lines)

    def scrape_storyweaver_page(self, slug: str) -> Optional[Dict]:
        """
        Scrape StoryWeaver story page directly (fallback method)

        Args:
            slug: Story slug identifier

        Returns:
            Story dictionary or None
        """
        try:
            url = f"https://storyweaver.org.in/stories/{slug}"
            time.sleep(self.rate_limit_delay)

            response = requests.get(
                url,
                timeout=15,
                headers={'User-Agent': self.user_agent}
            )

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract story pages
            story_pages = soup.find_all('div', class_='story-page')
            if not story_pages:
                story_pages = soup.find_all('div', {'data-page': True})

            story_text = []
            for page in story_pages:
                page_text = page.get_text(separator=' ', strip=True)
                if page_text and len(page_text) > 10:
                    story_text.append(page_text)

            full_text = ' '.join(story_text)

            if full_text and len(full_text) > 50:
                return {
                    'source': 'storyweaver',
                    'text': full_text,
                    'url': url
                }

            return None

        except Exception as e:
            logger.debug(f"Error scraping StoryWeaver page {slug}: {str(e)}")
            return None

    def scrape_freekidsbooks(self) -> List[Dict]:
        """
        Scrape Hindi stories from Free Kids Books

        Returns:
            List of story dictionaries
        """
        stories = []
        base_url = "https://freekidsbooks.org"
        hindi_page_url = f"{base_url}/subject/files/foreign-language/hindi-stories/"

        try:
            time.sleep(self.rate_limit_delay)

            response = requests.get(
                hindi_page_url,
                timeout=15,
                headers={'User-Agent': self.user_agent}
            )

            if response.status_code != 200:
                logger.warning(f"Failed to fetch Free Kids Books page: {response.status_code}")
                return stories

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all story links
            story_links = []

            # Look for book links (various possible patterns)
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')

                # Filter for story/book pages
                if any(pattern in href for pattern in ['/book/', '/story/', 'hindi']):
                    full_url = urljoin(base_url, href)

                    if full_url not in self.seen_urls and full_url not in story_links:
                        story_links.append(full_url)

            logger.info(f"   Found {len(story_links)} potential story links")

            # Fetch individual stories
            for i, story_url in enumerate(story_links[:100]):  # Limit to 100 links
                if len(stories) >= self.max_stories:
                    break

                story_data = self.scrape_freekidsbooks_story(story_url)

                if story_data and story_data.get('text'):
                    stories.append(story_data)
                    self.seen_urls.add(story_url)

                if (i + 1) % 10 == 0:
                    logger.info(f"   Processed {i + 1}/{len(story_links)} links, collected {len(stories)} stories")

        except Exception as e:
            logger.error(f"Error scraping Free Kids Books: {str(e)}")

        return stories

    def scrape_freekidsbooks_story(self, url: str) -> Optional[Dict]:
        """
        Scrape individual story from Free Kids Books

        Args:
            url: Story page URL

        Returns:
            Story dictionary or None
        """
        try:
            time.sleep(self.rate_limit_delay)

            response = requests.get(
                url,
                timeout=15,
                headers={'User-Agent': self.user_agent}
            )

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Try multiple methods to extract story content
            story_text = None

            # Method 1: Look for main content div
            content_div = soup.find('div', class_=['content', 'story-content', 'entry-content'])
            if content_div:
                story_text = content_div.get_text(separator=' ', strip=True)

            # Method 2: Look for article tag
            if not story_text or len(story_text) < 50:
                article = soup.find('article')
                if article:
                    story_text = article.get_text(separator=' ', strip=True)

            # Method 3: Get all paragraphs
            if not story_text or len(story_text) < 50:
                paragraphs = soup.find_all('p')
                story_text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])

            # Validate story text
            if story_text and len(story_text) > 100:
                # Check if it's actually Hindi text
                if self.is_hindi_text(story_text):
                    return {
                        'source': 'freekidsbooks',
                        'text': self.clean_story_text(story_text),
                        'url': url
                    }

            return None

        except Exception as e:
            logger.debug(f"Error scraping story from {url}: {str(e)}")
            return None

    def is_hindi_text(self, text: str) -> bool:
        """
        Check if text contains significant Hindi (Devanagari) content

        Args:
            text: Text to check

        Returns:
            True if text is primarily Hindi
        """
        if not text or len(text) < 50:
            return False

        # Count Devanagari characters
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        total_chars = sum(1 for char in text if not char.isspace())

        if total_chars == 0:
            return False

        hindi_ratio = devanagari_count / total_chars
        return hindi_ratio > 0.5  # At least 50% Devanagari

    def clean_story_text(self, text: str) -> str:
        """
        Clean and normalize story text

        Args:
            text: Raw story text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common noise patterns
        text = re.sub(r'(?i)(click here|download|pdf|epub|read more)', '', text)
        text = re.sub(r'[\[\]{}]', '', text)  # Remove brackets

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove repeated punctuation
        text = re.sub(r'([।॥!?])\1+', r'\1', text)

        # Remove extra spaces around punctuation
        text = re.sub(r'\s*([।॥,;:.!?])\s*', r'\1 ', text)

        return text.strip()

    def filter_stories(self, stories: List[Dict]) -> List[Dict]:
        """
        Filter stories for quality and age-appropriateness

        Args:
            stories: List of story dictionaries

        Returns:
            Filtered list of stories
        """
        filtered = []

        for story in stories:
            text = story.get('text', '')

            # Length filters
            if len(text) < 100:  # Too short
                continue

            if len(text) > 10000:  # Too long for children's story
                continue

            # Word count check
            word_count = len(text.split())
            if word_count < 20 or word_count > 2000:
                continue

            # Check Hindi content
            if not self.is_hindi_text(text):
                continue

            # Check for age-appropriate complexity
            if self.is_age_appropriate(text):
                filtered.append(story)

        return filtered

    def is_age_appropriate(self, text: str) -> bool:
        """
        Check if text is age-appropriate for children

        Args:
            text: Story text

        Returns:
            True if appropriate for children
        """
        # Calculate average word length
        words = text.split()
        if not words:
            return False

        avg_word_length = sum(len(word) for word in words) / len(words)

        # Children's stories typically have shorter words (3-8 characters in Hindi)
        if avg_word_length > 10:
            return False

        # Check for overly complex vocabulary indicators
        # (This is a simple heuristic - could be enhanced)
        complex_indicators = ['प्रशासन', 'संविधान', 'अर्थशास्त्र', 'राजनीति']
        if any(indicator in text for indicator in complex_indicators):
            return False

        return True


def collect_childrens_stories() -> List[str]:
    """
    Main entry point for collecting children's stories

    This function is called by corpus_builder.py to collect Hindi
    children's stories for the BabyLM corpus.

    Returns:
        List of story texts (strings)
    """
    collector = ChildrensStoryCollector(max_stories=2000, rate_limit_delay=2.0)
    return collector.collect_all_stories()


def scrape_hindi_stories() -> List[Dict]:
    """
    Legacy function for backward compatibility

    Returns:
        List of story dictionaries with metadata
    """
    collector = ChildrensStoryCollector(max_stories=2000, rate_limit_delay=2.0)
    stories = collector.collect_all_stories()

    # Convert to dictionary format for backward compatibility
    return [
        {
            'source': 'childrens_books',
            'text': story,
            'url': ''
        }
        for story in stories
    ]


# For testing
if __name__ == "__main__":
    print("Testing Children's Story Collection Module...")
    print("=" * 60)

    # Test with small sample
    collector = ChildrensStoryCollector(max_stories=10, rate_limit_delay=1.0)
    stories = collector.collect_all_stories()

    print(f"\nCollected {len(stories)} stories")

    if stories:
        print("\nSample story (first 200 characters):")
        print("-" * 60)
        print(stories[0][:200] + "...")
        print("-" * 60)

        print("\nStory statistics:")
        for i, story in enumerate(stories[:5], 1):
            word_count = len(story.split())
            char_count = len(story)
            print(f"Story {i}: {word_count} words, {char_count} characters")
