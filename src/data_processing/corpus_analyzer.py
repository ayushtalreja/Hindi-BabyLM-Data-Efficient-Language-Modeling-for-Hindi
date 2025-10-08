from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

class CorpusAnalyzer:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.stats = {}
    
    def compute_basic_stats(self) -> Dict:
        """Compute basic corpus statistics"""
        total_texts = len(self.corpus)
        total_characters = sum(len(text) for text in self.corpus)
        total_words = sum(len(text.split()) for text in self.corpus)
        
        return {
            'total_texts': total_texts,
            'total_characters': total_characters,
            'total_words': total_words,
            'avg_chars_per_text': total_characters / total_texts,
            'avg_words_per_text': total_words / total_texts
        }
    
    def analyze_word_frequency(self, top_k=1000) -> Counter:
        """Analyze word frequency distribution"""
        all_words = []
        for text in self.corpus:
            words = text.split()
            all_words.extend(words)

        word_freq = Counter(all_words)
        return word_freq.most_common(top_k)
    
    def analyze_morphological_complexity(self) -> Dict:
        """Analyze morphological patterns in the corpus"""
        # Common Hindi suffixes and morphological markers
        suffixes = {
            'plural': ['ों', 'ें', 'यां', 'ओं', 'एं'],
            'case_markers': ['ने', 'को', 'में', 'पर', 'से', 'का', 'की', 'के'],
            'verb_markers': ['ता', 'ती', 'ते', 'या', 'ये', 'ना', 'ने']
        }

        suffix_counts = {category: 0 for category in suffixes}
        total_words = 0
        avg_word_length = 0

        for text in self.corpus:
            words = text.split()
            total_words += len(words)
            avg_word_length += sum(len(word) for word in words)

            # Count suffix occurrences
            for word in words:
                for category, suffix_list in suffixes.items():
                    for suffix in suffix_list:
                        if word.endswith(suffix):
                            suffix_counts[category] += 1
                            break

        return {
            'total_words': total_words,
            'avg_word_length': avg_word_length / total_words if total_words > 0 else 0,
            'suffix_frequencies': suffix_counts,
            'morphological_density': sum(suffix_counts.values()) / total_words if total_words > 0 else 0
        }
    
    def plot_statistics(self, save_path: str):
        """Generate and save statistical plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Text length distribution
        text_lengths = [len(text) for text in self.corpus]
        axes[0, 0].hist(text_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Text Length Distribution')
        axes[0, 0].set_xlabel('Character Count')
        axes[0, 0].set_ylabel('Frequency')

        # 2. Word frequency (top 20)
        word_freq = self.analyze_word_frequency(top_k=20)
        words, counts = zip(*word_freq) if word_freq else ([], [])
        axes[0, 1].barh(range(len(words)), counts)
        axes[0, 1].set_yticks(range(len(words)))
        axes[0, 1].set_yticklabels(words)
        axes[0, 1].set_title('Top 20 Most Frequent Words')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].invert_yaxis()

        # 3. Word length distribution
        all_words = []
        for text in self.corpus:
            all_words.extend(text.split())
        word_lengths = [len(word) for word in all_words]
        axes[1, 0].hist(word_lengths, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Word Length Distribution')
        axes[1, 0].set_xlabel('Character Count')
        axes[1, 0].set_ylabel('Frequency')

        # 4. Morphological complexity
        morph_stats = self.analyze_morphological_complexity()
        suffix_freqs = morph_stats['suffix_frequencies']
        axes[1, 1].bar(suffix_freqs.keys(), suffix_freqs.values())
        axes[1, 1].set_title('Morphological Marker Frequencies')
        axes[1, 1].set_xlabel('Marker Type')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Statistics plot saved to {save_path}")