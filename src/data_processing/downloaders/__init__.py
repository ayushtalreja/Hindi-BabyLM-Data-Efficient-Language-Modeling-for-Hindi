"""
Data downloaders for the Hindi BabyLM pipeline.

This module provides downloader classes for various Hindi data sources:
- IndicCorp (web-scraped formal Hindi)
- Wikipedia (encyclopedic Hindi)
- IndicDialogue (conversational Hindi from movie subtitles)
- Children's Books (simple narrative Hindi)
"""

from .base_downloader import BaseDownloader
from .indiccorp_downloader import IndicCorpDownloader
from .wiki_downloader import WikiDownloader
from .indicdialogue_loader import IndicDialogueLoader

__all__ = [
    "BaseDownloader",
    "IndicCorpDownloader",
    "WikiDownloader",
    "IndicDialogueLoader"
]
