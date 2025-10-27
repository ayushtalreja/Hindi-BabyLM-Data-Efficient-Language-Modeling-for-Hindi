import wikipediaapi
import pandas as pd
import os
import json
import pickle
import requests
import time
from pathlib import Path
from typing import List, Dict
from functools import wraps
from urllib3.exceptions import ProtocolError
from requests.exceptions import ConnectionError, Timeout

# Checkpoint file path
CHECKPOINT_FILE = Path('.wiki_checkpoint.json')

def retry_with_backoff(max_retries=5, initial_delay=1):
    """Decorator to retry functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, ProtocolError, Timeout, Exception) as e:
                    if "RemoteDisconnected" in str(e) or "Connection aborted" in str(e) or isinstance(e, (ConnectionError, ProtocolError, Timeout)):
                        if attempt < max_retries - 1:
                            print(f"   ⚠ Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                            print(f"   ⏳ Retrying in {delay} seconds...")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                        else:
                            print(f"   ❌ Failed after {max_retries} attempts")
                            raise
                    else:
                        raise
            return None
        return wrapper
    return decorator


def fetch_articles_batch(titles: List[str], language='hi', user_agent='HindiBabyLM/1.0 (Research Project)'):
    """
    Fetch multiple Wikipedia articles in a single API call using MediaWiki API

    Args:
        titles: List of article titles to fetch (max 50 per batch)
        language: Wikipedia language code
        user_agent: User agent string

    Returns:
        Dict mapping title to article text
    """
    if len(titles) > 50:
        raise ValueError("Maximum 50 titles per batch")

    url = f"https://{language}.wikipedia.org/w/api.php"

    params = {
        'action': 'query',
        'format': 'json',
        'titles': '|'.join(titles),
        'prop': 'extracts',
        'explaintext': True,
        'exsectionformat': 'plain'
    }

    headers = {
        'User-Agent': user_agent
    }

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()

    results = {}
    if 'query' in data and 'pages' in data['query']:
        for page_data in data['query']['pages'].values():
            if 'extract' in page_data and page_data.get('extract'):
                title = page_data.get('title', '')
                results[title] = page_data['extract']

    return results


def refresh_wiki_connection(language='hi', user_agent='HindiBabyLM/1.0 (Research Project)'):
    """Create a fresh Wikipedia API connection"""
    return wikipediaapi.Wikipedia(
        language=language,
        user_agent=user_agent
    )


def save_checkpoint(checkpoint_data: Dict):
    """Save scraping progress to checkpoint file"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)


def load_checkpoint() -> Dict:
    """Load scraping progress from checkpoint file"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def delete_checkpoint():
    """Delete checkpoint file after successful completion"""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def scrape_hindi_wikipedia(categories: List[str], max_articles: int = 10000):
    """
    Scrape Hindi Wikipedia articles from specified categories with batching and checkpointing

    Args:
        categories: List of category names to scrape
        max_articles: Maximum number of articles to collect

    Returns:
        List of article dictionaries with 'title', 'text', 'category'
    """
    # Load checkpoint if it exists
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"   📋 Found checkpoint with {len(checkpoint['articles'])} articles already scraped")
        print(f"   📋 Resuming from checkpoint...")
        articles = checkpoint['articles']
        seen_titles = set(checkpoint['seen_titles'])
        processed_categories = set(checkpoint.get('processed_categories', []))
    else:
        articles = []
        seen_titles = set()
        processed_categories = set()

    # Initialize Wikipedia connection
    wiki = refresh_wiki_connection()

    # Track when to refresh connection and save checkpoint
    articles_since_refresh = 0
    articles_since_checkpoint = len(articles) % 1000

    print(f"   🔍 Collecting article titles from categories...")

    # Step 1: Collect all article titles from categories (without fetching content)
    all_titles = []

    for category_name in categories:
        if category_name in processed_categories:
            print(f"   ⏭️  Skipping already processed category: {category_name}")
            continue

        if len(articles) >= max_articles:
            break

        print(f"   📚 Processing category: {category_name}")

        # Get category page
        cat = wiki.page(f"Category:{category_name}")

        if not cat.exists():
            print(f"   ⚠️  Category '{category_name}' not found")
            processed_categories.add(category_name)
            continue

        # Collect titles from this category
        def get_category_members(categorymembers, level=0, max_level=1):
            """Recursively collect article titles from category"""
            titles = []
            for c in categorymembers.values():
                if len(articles) + len(all_titles) >= max_articles:
                    return titles

                if c.ns == wikipediaapi.Namespace.MAIN:
                    # It's an article
                    if c.title not in seen_titles:
                        titles.append((c.title, category_name))
                elif c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                    # It's a subcategory - recursively get its members
                    titles.extend(get_category_members(c.categorymembers, level=level+1, max_level=max_level))

            return titles

        category_titles = get_category_members(cat.categorymembers)
        all_titles.extend(category_titles)
        processed_categories.add(category_name)

        print(f"   ✓ Found {len(category_titles)} articles in {category_name}")

    # Remove already seen titles
    all_titles = [(title, cat) for title, cat in all_titles if title not in seen_titles]

    # Limit to max_articles
    remaining_slots = max_articles - len(articles)
    all_titles = all_titles[:remaining_slots]

    print(f"   📊 Total new articles to fetch: {len(all_titles)}")

    # Step 2: Fetch articles in batches of 50
    batch_size = 50
    total_batches = (len(all_titles) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(all_titles), batch_size):
        batch_titles = all_titles[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch_titles)} articles)...")

        # Extract just the titles for API call
        titles_only = [title for title, _ in batch_titles]

        # Fetch batch with retry logic
        @retry_with_backoff(max_retries=5, initial_delay=1)
        def fetch_batch():
            return fetch_articles_batch(titles_only)

        try:
            batch_results = fetch_batch()

            # Process results
            batch_stats = {'requested': len(batch_titles), 'returned': len(batch_results), 'passed_filter': 0, 'too_short': 0}

            for title, category in batch_titles:
                if title in batch_results:
                    text = clean_wikipedia_text(batch_results[title])
                    if text and len(text) > 20:  # Minimum length filter (reduced from 100 to 20)
                        articles.append({
                            'title': title,
                            'text': text,
                            'category': category
                        })
                        seen_titles.add(title)
                        articles_since_refresh += 1
                        articles_since_checkpoint += 1
                        batch_stats['passed_filter'] += 1
                    elif text:
                        batch_stats['too_short'] += 1

            # Log batch statistics
            print(f"      Requested: {batch_stats['requested']}, Returned: {batch_stats['returned']}, " +
                  f"Passed: {batch_stats['passed_filter']}, Too short: {batch_stats['too_short']}")

            # Refresh connection every 2000 articles
            if articles_since_refresh >= 2000:
                print(f"   🔄 Refreshing Wikipedia connection...")
                wiki = refresh_wiki_connection()
                articles_since_refresh = 0

            # Save checkpoint every 1000 articles
            if articles_since_checkpoint >= 1000:
                checkpoint_data = {
                    'articles': articles,
                    'seen_titles': list(seen_titles),
                    'processed_categories': list(processed_categories)
                }
                save_checkpoint(checkpoint_data)
                print(f"   💾 Checkpoint saved ({len(articles)} articles)")
                articles_since_checkpoint = 0

            # Add small delay between batches to respect rate limits
            time.sleep(0.1)

        except Exception as e:
            print(f"   ❌ Error fetching batch {batch_num}: {e}")
            # Save checkpoint before failing
            checkpoint_data = {
                'articles': articles,
                'seen_titles': list(seen_titles),
                'processed_categories': list(processed_categories)
            }
            save_checkpoint(checkpoint_data)
            print(f"   💾 Emergency checkpoint saved. You can resume later.")
            raise

    # Final checkpoint save
    checkpoint_data = {
        'articles': articles,
        'seen_titles': list(seen_titles),
        'processed_categories': list(processed_categories)
    }
    save_checkpoint(checkpoint_data)

    print(f"   ✅ Successfully scraped {len(articles)} articles")

    # Delete checkpoint on successful completion
    delete_checkpoint()
    print(f"   🗑️  Checkpoint deleted (scraping completed successfully)")

    return articles


def clean_wikipedia_text(text: str) -> str:
    """Clean Wikipedia markup and formatting"""
    import re

    # Remove references like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Remove Wikipedia templates and markup
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)  # Keep link text only

    # Remove external links
    text = re.sub(r'http\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)

    # Remove lines that are just headers or metadata
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines
                     if line.strip() and
                     not line.strip().startswith('==') and
                     len(line.strip()) > 15]  # Filter very short lines

    return ' '.join(cleaned_lines)


def save_wikipedia_data(articles: List[Dict], output_dir: str = 'data/raw') -> Path:
    """
    Save Wikipedia articles to separate files

    Args:
        articles: List of article dictionaries with 'title', 'text', 'category'
        output_dir: Directory to save files

    Returns:
        Path to saved pickle file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract text content
    texts = [article['text'] for article in articles]

    # Save to pickle for fast loading
    pickle_path = output_dir / 'wikipedia.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(texts, f)

    # Save metadata
    metadata = {
        'num_articles': len(articles),
        'categories': list(set(article.get('category', 'unknown') for article in articles)),
        'total_chars': sum(len(article['text']) for article in articles),
        'titles': [article.get('title', '') for article in articles[:100]]  # Sample titles
    }
    metadata_path = output_dir / 'wikipedia_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(texts)} Wikipedia articles to {pickle_path}")
    print(f"✓ Saved metadata to {metadata_path}")

    return pickle_path
