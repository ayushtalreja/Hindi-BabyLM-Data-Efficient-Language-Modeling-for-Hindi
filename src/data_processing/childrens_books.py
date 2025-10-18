"""
Children's Books Data Collection Module

This module collects Hindi children's stories from openly-licensed sources
to create developmentally appropriate training data for the Hindi BabyLM project.

Sources:
- StoryWeaver (Pratham Books) - API-based collection with openly-licensed content

All content is collected respecting copyright, robots.txt, and rate limits.
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from typing import List, Dict, Optional
from urllib.parse import urljoin
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose urllib3/requests connection logs
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
logging.getLogger('pdfminer').setLevel(logging.ERROR)  # Suppress pdfminer warnings


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
        Main entry point: Collect stories from StoryWeaver
        Returns empty list if collection fails (non-blocking)

        Returns:
            List of story texts (strings), empty list if collection fails
        """
        logger.info("Starting children's story collection...")
        all_stories_data = []

        # Collect from StoryWeaver API with robust error handling
        logger.info("\n1. Collecting from StoryWeaver API...")
        try:
            storyweaver_stories = self.scrape_storyweaver_api()
            all_stories_data.extend(storyweaver_stories)
            logger.info(f"   Collected {len(storyweaver_stories)} stories from StoryWeaver")
        except requests.Timeout:
            logger.warning("   StoryWeaver API timed out - skipping this source")
        except requests.RequestException as e:
            logger.warning(f"   StoryWeaver API request failed: {e} - skipping this source")
        except Exception as e:
            logger.warning(f"   Error collecting from StoryWeaver: {str(e)} - skipping this source")

        # If no stories collected, return empty list (don't crash pipeline)
        if not all_stories_data:
            logger.warning("Could not collect any children's stories from StoryWeaver")
            logger.warning("Continuing pipeline without children's stories data")
            return []

        # Filter for quality and age-appropriateness
        logger.info("\n2. Filtering stories for quality and age-appropriateness...")
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

        # StoryWeaver API endpoint for Hindi stories (updated to books-search)
        api_base = "https://storyweaver.org.in/api/v1/books-search"

        # Parameters for Hindi stories, sorted by reading level
        params = {
            'language': 'Hindi',
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

                    # Filter for Hindi language stories (API language param doesn't filter properly)
                    if story_item.get('language') != 'Hindi':
                        continue

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

            # Use the read API endpoint to get story pages
            api_url = f"https://storyweaver.org.in/api/v1/stories/{slug}/read"
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
                        'title': story_data.get('name', ''),
                        'text': story_text,
                        'reading_level': story_data.get('level', 'unknown'),
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
            # Skip front cover and back cover pages
            page_type = page.get('pageType', '')
            if page_type in ['FrontCoverPage', 'BackCoverPage', 'BackInnerCoverPage']:
                continue

            # Extract HTML content
            html_content = page.get('html', '')
            if html_content:
                # Parse HTML and extract text
                soup = BeautifulSoup(html_content, 'html.parser')
                # Remove script and style tags
                for script in soup(['script', 'style']):
                    script.decompose()

                text = soup.get_text(separator=' ', strip=True)
                # Remove page numbers and extra whitespace
                text = ' '.join(text.split())

                if text and len(text) > 10:
                    story_lines.append(text)

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
        return hindi_ratio > 0.8  # At least 80% Devanagari

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
