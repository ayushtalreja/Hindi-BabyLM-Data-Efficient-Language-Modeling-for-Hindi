import wikipediaapi
import pandas as pd
from typing import List, Dict

def scrape_hindi_wikipedia(categories: List[str], max_articles: int = 10000):
    """Scrape Hindi Wikipedia articles from specified categories"""
    wiki = wikipediaapi.Wikipedia(
        language='hi',
        user_agent='HindiBabyLM/1.0 (Research Project)'
    )
    articles = []
    article_count = 0
    seen_titles = set()

    for category_name in categories:
        if article_count >= max_articles:
            break

        # Get category page
        cat = wiki.page(f"Category:{category_name}")

        if not cat.exists():
            print(f"Category '{category_name}' not found")
            continue

        # Get all pages in category
        def get_category_members(categorymembers, level=0, max_level=1):
            nonlocal article_count
            for c in categorymembers.values():
                if article_count >= max_articles:
                    return

                if c.ns == wikipediaapi.Namespace.MAIN:
                    # It's an article
                    if c.title not in seen_titles:
                        seen_titles.add(c.title)
                        text = clean_wikipedia_text(c.text)
                        if text and len(text) > 100:  # Minimum length filter
                            articles.append({
                                'title': c.title,
                                'text': text,
                                'category': category_name
                            })
                            article_count += 1

                elif c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                    # It's a subcategory
                    get_category_members(c.categorymembers, level=level+1, max_level=max_level)

        get_category_members(cat.categorymembers)

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
                     len(line.strip()) > 20]  # Filter very short lines

    return ' '.join(cleaned_lines)