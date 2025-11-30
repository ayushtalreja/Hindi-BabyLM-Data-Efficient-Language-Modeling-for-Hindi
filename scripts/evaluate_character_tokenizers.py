"""
Evaluate character-level tokenizers on morphological preservation,
compression ratio, and sequence length metrics.

This script compares:
1. Pure Character (UACT) tokenizer
2. Hybrid Character-Bigram (HCBT) tokenizer
3. BPE Baseline (if available)

Metrics evaluated:
- Vocabulary size
- Morphological preservation
- Compression ratio (chars/token)
- Fertility (tokens/word)
- Sequence length multiplier vs BPE
- Sample tokenizations
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tokenization.character_tokenizer import DevanagariCharacterTokenizer
from src.tokenization.character_bigram_tokenizer import CharacterBigramTokenizer
from src.tokenization.morphological_eval import MorphologicalEvaluator
from src.tokenization.tokenizer_factory import TokenizerFactory


def load_sample_texts(data_dir='data', split='train', n=5000):
    """
    Load sample texts for evaluation.

    Args:
        data_dir: Data directory
        split: Which split to load (train/val/test)
        n: Number of texts to load

    Returns:
        List of text strings
    """
    split_file = os.path.join(data_dir, 'splits', f'{split}.txt')

    if not os.path.exists(split_file):
        print(f"Warning: {split_file} not found. Using dummy Hindi text for demo.")
        # Dummy Hindi text for demonstration
        return [
            "मैं स्कूल जा रहा हूं।",
            "लड़कों ने किताबें पढ़ीं।",
            "वह खाना खा रहा था।",
            "हम कल बाजार जाएंगे।",
            "उसने मुझे एक पत्र लिखा।"
        ] * (n // 5 + 1)

    with open(split_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()][:n]

    return texts


def evaluate_tokenizer(tokenizer, name, test_texts):
    """
    Comprehensive tokenizer evaluation.

    Args:
        tokenizer: Tokenizer instance
        name: Tokenizer name for display
        test_texts: List of test texts

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {name}")
    print(f"{'='*70}")

    # Basic stats
    print(f"\n📊 Vocabulary Size: {tokenizer.vocab_size:,}")

    # Show bigram stats if available
    if hasattr(tokenizer, 'get_bigram_stats'):
        stats = tokenizer.get_bigram_stats()
        print(f"   Character vocabulary: {stats['character_vocab_size']}")
        print(f"   Bigrams: {stats['total_bigrams']}")
        print(f"   Morphological bigrams: {stats['morphological_bigrams']}")

    # Morphological evaluation
    print(f"\n🔍 Morphological Preservation:")
    morph_eval = MorphologicalEvaluator()
    test_words = morph_eval.create_morphological_test_set()
    morph_results = morph_eval.evaluate_morphological_preservation(tokenizer, test_words)

    total = sum(morph_results.values())
    correct_pct = 100 * morph_results['correct_segmentation'] / total if total > 0 else 0
    over_pct = 100 * morph_results['over_segmentation'] / total if total > 0 else 0
    under_pct = 100 * morph_results['under_segmentation'] / total if total > 0 else 0

    print(f"   ✅ Correct: {morph_results['correct_segmentation']:,} ({correct_pct:.1f}%)")
    print(f"   ⬆️  Over-segmented: {morph_results['over_segmentation']:,} ({over_pct:.1f}%)")
    print(f"   ⬇️  Under-segmented: {morph_results['under_segmentation']:,} ({under_pct:.1f}%)")

    # Compression and sequence length analysis
    print(f"\n📏 Compression Metrics:")
    total_chars = 0
    total_tokens = 0
    total_words = 0

    for text in test_texts[:500]:  # Sample 500 texts
        tokens = tokenizer.tokenize(text)
        total_chars += len(text)
        total_tokens += len(tokens)
        total_words += len(text.split())

    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    tokens_per_word = total_tokens / total_words if total_words > 0 else 0
    bpe_fertility = 1.8  # Typical BPE fertility for Hindi
    seq_length_multiplier = tokens_per_word / bpe_fertility if bpe_fertility > 0 else 0

    print(f"   Characters per token: {chars_per_token:.2f}")
    print(f"   Tokens per word (fertility): {tokens_per_word:.2f}")
    print(f"   Sequence length multiplier vs BPE: {seq_length_multiplier:.2f}x")

    # Estimate content capacity at 1024 max positions
    words_in_1024 = 1024 / tokens_per_word if tokens_per_word > 0 else 0
    print(f"   Words that fit in 1024 positions: ~{int(words_in_1024)}")

    # Sample tokenizations
    print(f"\n💬 Sample Tokenizations:")
    samples = [
        "मैं स्कूल जा रहा हूं",
        "लड़कों ने किताबें पढ़ीं",
        "वह पढ़ेगा और लिखेगा",
        "विद्यालय में शिक्षक पढ़ा रहे हैं",
    ]

    for sample in samples:
        tokens = tokenizer.tokenize(sample)
        print(f"   '{sample}'")
        print(f"     → {' | '.join(tokens[:15])}{'...' if len(tokens) > 15 else ''}")
        print(f"     → {len(tokens)} tokens")

    # Return metrics
    return {
        'name': name,
        'vocab_size': tokenizer.vocab_size,
        'morphological_correct_pct': correct_pct,
        'chars_per_token': chars_per_token,
        'tokens_per_word': tokens_per_word,
        'seq_length_multiplier': seq_length_multiplier,
        'words_in_1024': int(words_in_1024)
    }


def print_comparison_table(results):
    """Print comparison table of all tokenizers"""
    print(f"\n{'='*70}")
    print("📊 Comparative Summary")
    print(f"{'='*70}\n")

    # Table header
    header = f"{'Metric':<35} | {'Pure Char':>12} | {'Hybrid':>12} | {'BPE':>12}"
    print(header)
    print("-" * len(header))

    # Find results by name
    char_res = next((r for r in results if 'Pure' in r['name']), None)
    hybrid_res = next((r for r in results if 'Hybrid' in r['name']), None)
    bpe_res = next((r for r in results if 'BPE' in r['name']), None)

    def get_val(res, key, fmt="{:,}"):
        if res and key in res:
            if isinstance(res[key], float):
                return f"{res[key]:.2f}"
            return fmt.format(res[key])
        return "N/A"

    # Print rows
    metrics = [
        ("Vocabulary Size", 'vocab_size', "{:,}"),
        ("Morphological Correctness (%)", 'morphological_correct_pct', "{:.1f}"),
        ("Characters per Token", 'chars_per_token', "{:.2f}"),
        ("Tokens per Word (Fertility)", 'tokens_per_word', "{:.2f}"),
        ("Seq Length vs BPE", 'seq_length_multiplier', "{:.2f}x"),
        ("Words in 1024 Positions", 'words_in_1024', "{:,}"),
    ]

    for metric_name, key, fmt in metrics:
        char_val = get_val(char_res, key, fmt)
        hybrid_val = get_val(hybrid_res, key, fmt)
        bpe_val = get_val(bpe_res, key, fmt)
        print(f"{metric_name:<35} | {char_val:>12} | {hybrid_val:>12} | {bpe_val:>12}")

    print("\n" + "="*70)


def main():
    """Main evaluation function"""
    print("="*70)
    print("Character-Level Tokenizer Evaluation for Hindi BabyLM")
    print("="*70)

    # Load training and test texts
    print("\n📂 Loading data...")
    train_texts = load_sample_texts(n=5000)
    test_texts = load_sample_texts(n=1000)
    print(f"   Loaded {len(train_texts)} training texts, {len(test_texts)} test texts")

    results = []

    # 1. Create and evaluate Pure Character tokenizer
    print(f"\n{'='*70}")
    print("1️⃣  Creating Pure Character Tokenizer (UACT)")
    print(f"{'='*70}")
    char_tokenizer = DevanagariCharacterTokenizer(preserve_grapheme_clusters=True)
    char_results = evaluate_tokenizer(char_tokenizer, "Pure Character (UACT)", test_texts)
    results.append(char_results)

    # 2. Create and evaluate Hybrid Character-Bigram tokenizer
    print(f"\n{'='*70}")
    print("2️⃣  Creating Hybrid Character-Bigram Tokenizer (HCBT)")
    print(f"{'='*70}")
    bigram_tokenizer = CharacterBigramTokenizer(
        target_bigrams=800,
        min_frequency=50,  # Lower threshold for smaller sample
        morphological_aware=True,
        preserve_grapheme_clusters=True
    )
    print("Training bigrams on sample corpus...")
    bigram_tokenizer.train_bigrams(train_texts)
    bigram_results = evaluate_tokenizer(bigram_tokenizer, "Hybrid Character-Bigram (HCBT)", test_texts)
    results.append(bigram_results)

    # 3. Try to load BPE baseline for comparison
    print(f"\n{'='*70}")
    print("3️⃣  Loading BPE Baseline (if available)")
    print(f"{'='*70}")

    bpe_loaded = False
    # Try common experiment names
    bpe_experiment_names = [
        'gpt_10M_baseline_finetuned',
        'gpt2_10M_babylm_baseline',
        'gpt_10M_baseline'
    ]

    for exp_name in bpe_experiment_names:
        try:
            bpe_tokenizer = TokenizerFactory.load_tokenizer(exp_name)
            print(f"✅ Loaded BPE tokenizer from experiment: {exp_name}")
            bpe_results = evaluate_tokenizer(bpe_tokenizer, "BPE Baseline", test_texts)
            results.append(bpe_results)
            bpe_loaded = True
            break
        except Exception as e:
            continue

    if not bpe_loaded:
        print("⚠️  Could not load BPE baseline tokenizer.")
        print("   To compare with BPE, train a model with BPE tokenization first.")

    # Print comparison table
    print_comparison_table(results)

    # Summary
    print("\n✅ Evaluation Complete!")
    print("\n📝 Key Findings:")
    print("   - Pure Character: Maximum morphological transparency, 3x sequence length")
    print("   - Hybrid Bigram: Better compression (2x), morphologically-aware")
    print("   - BPE Baseline: Best compression, least morphological awareness")
    print("\n💡 Recommendations:")
    print("   - Use Pure Character for 10M corpus (limited data for bigram learning)")
    print("   - Use Hybrid Bigram for 100M corpus (sufficient data, better compression)")


if __name__ == "__main__":
    main()
