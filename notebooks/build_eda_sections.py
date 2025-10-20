#!/usr/bin/env python3
"""
Script to build remaining EDA notebook sections programmatically.
This avoids the character limit issues when creating large notebooks.
"""

import json
from pathlib import Path

# Load existing notebook
notebook_path = Path(__file__).parent / '01_data_exploration.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# New cells to append
new_cells = []

# ============================================================================
# SECTION 2: BASIC STATISTICS (ENHANCED)
# ============================================================================

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## Section 2: Enhanced Basic Statistics\n",
        "\n",
        "Comprehensive corpus statistics with source-wise comparisons, vocabulary metrics, and statistical tests."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# CALCULATE BASIC STATISTICS FOR ALL DATA\n",
        "# ============================================================================\n",
        "\n",
        "print(\"📊 Computing statistics for all sources...\\n\")\n",
        "\n",
        "# Calculate stats for each source\n",
        "source_stats = {}\n",
        "for source_name, texts in sources.items():\n",
        "    stats = calculate_basic_stats(texts)\n",
        "    source_stats[source_name] = stats\n",
        "    \n",
        "# Calculate stats for splits\n",
        "split_stats = {\n",
        "    'Train': calculate_basic_stats(train_texts),\n",
        "    'Val': calculate_basic_stats(val_texts),\n",
        "    'Test': calculate_basic_stats(test_texts),\n",
        "}\n",
        "\n",
        "# Convert to DataFrame for easy viewing\n",
        "df_sources = pd.DataFrame(source_stats).T\n",
        "df_splits = pd.DataFrame(split_stats).T\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"SOURCE-WISE STATISTICS\")\n",
        "print(\"=\"*80)\n",
        "print(df_sources.to_string())\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"SPLIT-WISE STATISTICS\")\n",
        "print(\"=\"*80)\n",
        "print(df_splits.to_string())\n",
        "\n",
        "# Export to CSV\n",
        "csv_path = DATA_DIR / 'source_statistics.csv'\n",
        "df_sources.to_csv(csv_path)\n",
        "print(f\"\\n💾 Source statistics saved to: {csv_path}\")"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# VOCABULARY RICHNESS ANALYSIS\n",
        "# ============================================================================\n",
        "\n",
        "print(\"📚 Analyzing vocabulary richness...\\n\")\n",
        "\n",
        "richness_stats = {}\n",
        "for source_name, texts in sources.items():\n",
        "    richness = calculate_vocabulary_richness(texts)\n",
        "    richness_stats[source_name] = richness\n",
        "\n",
        "df_richness = pd.DataFrame(richness_stats).T\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"VOCABULARY RICHNESS METRICS\")\n",
        "print(\"=\"*80)\n",
        "print(df_richness.to_string())\n",
        "print(\"\\nInterpretation:\")\n",
        "print(\"  - TTR: Type-Token Ratio (higher = more diverse vocabulary)\")\n",
        "print(\"  - Root TTR: Normalized TTR (accounts for text length)\")\n",
        "print(\"  - Hapax Ratio: Proportion of words appearing only once\")\n",
        "print(\"  - Dis Ratio: Proportion of words appearing exactly twice\")"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# VOCABULARY OVERLAP ANALYSIS\n",
        "# ============================================================================\n",
        "\n",
        "print(\"🔍 Analyzing vocabulary overlap between sources...\\n\")\n",
        "\n",
        "# Extract vocabularies\n",
        "vocab_sets = {}\n",
        "for source_name, texts in sources.items():\n",
        "    words = [word for text in texts for word in text.split()]\n",
        "    vocab_sets[source_name] = set(words)\n",
        "\n",
        "# Calculate pairwise overlaps\n",
        "source_names = list(vocab_sets.keys())\n",
        "n_sources = len(source_names)\n",
        "overlap_matrix = np.zeros((n_sources, n_sources))\n",
        "\n",
        "for i, source1 in enumerate(source_names):\n",
        "    for j, source2 in enumerate(source_names):\n",
        "        if i == j:\n",
        "            overlap_matrix[i, j] = 1.0\n",
        "        else:\n",
        "            vocab1 = vocab_sets[source1]\n",
        "            vocab2 = vocab_sets[source2]\n",
        "            overlap = len(vocab1 & vocab2) / len(vocab1 | vocab2)\n",
        "            overlap_matrix[i, j] = overlap\n",
        "\n",
        "# Visualize as heatmap\n",
        "create_heatmap(\n",
        "    overlap_matrix,\n",
        "    source_names,\n",
        "    source_names,\n",
        "    'Vocabulary Overlap (Jaccard Index)',\n",
        "    figsize=(8, 6),\n",
        "    save_path=FIGURES_DIR / 'vocabulary_overlap.png'\n",
        ")\n",
        "\n",
        "print(\"\\n📊 Vocabulary sizes:\")\n",
        "for source, vocab in vocab_sets.items():\n",
        "    print(f\"  {source}: {len(vocab):,} unique words\")"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# SPLIT BALANCE VERIFICATION\n",
        "# ============================================================================\n",
        "\n",
        "print(\"⚖️  Verifying split balance...\\n\")\n",
        "\n",
        "# Calculate actual vs expected splits\n",
        "total_docs = len(train_texts) + len(val_texts) + len(test_texts)\n",
        "actual = [len(train_texts)/total_docs, len(val_texts)/total_docs, len(test_texts)/total_docs]\n",
        "expected = [0.80, 0.10, 0.10]\n",
        "split_names = ['Train', 'Val', 'Test']\n",
        "\n",
        "# Create comparison plot\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Bar chart\n",
        "x = np.arange(len(split_names))\n",
        "width = 0.35\n",
        "\n",
        "bars1 = ax1.bar(x - width/2, [a*100 for a in actual], width, label='Actual', color='steelblue')\n",
        "bars2 = ax1.bar(x + width/2, [e*100 for e in expected], width, label='Expected', color='coral')\n",
        "\n",
        "ax1.set_ylabel('Percentage (%)')\n",
        "ax1.set_title('Train/Val/Test Split Distribution')\n",
        "ax1.set_xticks(x)\n",
        "ax1.set_xticklabels(split_names)\n",
        "ax1.legend()\n",
        "ax1.grid(True, alpha=0.3)\n",
        "\n",
        "# Add value labels on bars\n",
        "for bars in [bars1, bars2]:\n",
        "    for bar in bars:\n",
        "        height = bar.get_height()\n",
        "        ax1.text(bar.get_x() + bar.get_width()/2., height,\n",
        "                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)\n",
        "\n",
        "# Pie chart\n",
        "colors = ['steelblue', 'coral', 'mediumseagreen']\n",
        "explode = (0.05, 0, 0)\n",
        "\n",
        "ax2.pie([len(train_texts), len(val_texts), len(test_texts)], \n",
        "        labels=split_names, \n",
        "        autopct='%1.1f%%',\n",
        "        colors=colors,\n",
        "        explode=explode,\n",
        "        startangle=90,\n",
        "        textprops={'fontsize': 11})\n",
        "ax2.set_title('Actual Split Distribution')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(FIGURES_DIR / 'split_distribution.png', dpi=300, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(\"✅ Split distribution plot saved\")\n",
        "print(\"\\n📊 Split comparison:\")\n",
        "for name, act, exp in zip(split_names, actual, expected):\n",
        "    diff = (act - exp) * 100\n",
        "    print(f\"  {name:5s}: {act*100:5.1f}% (expected {exp*100:5.1f}%, diff: {diff:+5.1f}%)\")"
    ]
})

# Add the cells to the notebook
notebook['cells'].extend(new_cells)

# Save the updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✅ Added Section 2 cells to notebook")
print(f"📄 Notebook now has {len(notebook['cells'])} cells")

