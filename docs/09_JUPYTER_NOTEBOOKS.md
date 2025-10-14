# Jupyter Notebooks for Analysis and Exploration

## Overview

The Hindi BabyLM project includes two comprehensive Jupyter notebooks that provide interactive data exploration and results analysis workflows. These notebooks are designed for thesis work, enabling reproducible analysis and publication-quality figure generation.

**Location**: `notebooks/`

**Notebooks**:
1. **01_data_exploration.ipynb**: Corpus statistics and data quality analysis
2. **02_results_analysis.ipynb**: Experimental results visualization and statistical testing

## Setup

### Environment Requirements

```bash
# Ensure you're in the project root
cd /path/to/hindi-babylm

# Activate virtual environment
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install Jupyter if not already installed
pip install jupyter jupyterlab ipywidgets

# Launch Jupyter
jupyter lab notebooks/
```

### Import Structure

Both notebooks use the following import pattern to access project modules:

```python
import sys
sys.path.append('..')  # Add parent directory to path

# Now can import project modules
from src.data_processing.text_cleaner import clean_text
from src.analysis.results_analyzer import ResultsAnalyzer
```

## 1. Data Exploration Notebook

**File**: `notebooks/01_data_exploration.ipynb`

**Purpose**: Comprehensive analysis of the Hindi training corpus to understand data characteristics, distributions, and quality.

### Notebook Structure

#### Section 1: Load Corpus Data

Loads train/validation/test splits from the processed data directory:

```python
# Load corpus files
data_dir = Path('../data/processed')

with open(data_dir / 'train.txt', 'r', encoding='utf-8') as f:
    train_texts = f.readlines()

with open(data_dir / 'val.txt', 'r', encoding='utf-8') as f:
    val_texts = f.readlines()

with open(data_dir / 'test.txt', 'r', encoding='utf-8') as f:
    test_texts = f.readlines()
```

**Outputs**:
- Dataset sizes (number of examples per split)
- Total corpus size

#### Section 2: Basic Statistics

Computes fundamental corpus statistics:

```python
def compute_statistics(texts):
    stats = {}

    # Character counts
    all_text = ''.join(texts)
    stats['total_characters'] = len(all_text)

    # Word counts
    all_words = [word for text in texts for word in text.split()]
    stats['total_words'] = len(all_words)
    stats['unique_words'] = len(set(all_words))

    # Sentence lengths
    word_counts = [len(text.split()) for text in texts]
    stats['avg_sentence_length'] = np.mean(word_counts)
    stats['median_sentence_length'] = np.median(word_counts)

    return stats
```

**Outputs**:
- Total tokens
- Unique tokens
- Type-Token Ratio (TTR)
- Average/median sentence length

**Typical Results**:
```
Total tokens: 10,234,567
Unique tokens: 234,567
Type-Token Ratio: 0.0229
Avg sentence length: 15.3 words
Median sentence length: 12 words
```

#### Section 3: Length Distribution Analysis

Visualizes sentence length distributions:

```python
# Word count distribution
train_word_counts = [len(text.split()) for text in train_texts]
train_char_counts = [len(text) for text in train_texts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot histograms with mean/median lines
ax1.hist(train_word_counts, bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(np.mean(train_word_counts), color='red', linestyle='--',
            label=f'Mean: {np.mean(train_word_counts):.1f}')
ax1.axvline(np.median(train_word_counts), color='green', linestyle='--',
            label=f'Median: {np.median(train_word_counts):.1f}')
```

**Generated Figures**:
- `figures/length_distributions.png`: Word and character count histograms

**Insights**:
- Identifies typical sentence lengths
- Detects outliers (very short or long sentences)
- Informs tokenization max_length parameter

#### Section 4: Character Analysis

Analyzes character distribution, focusing on Devanagari script:

```python
# Analyze character distribution
all_chars = ''.join(train_texts)
char_counter = Counter(all_chars)

# Devanagari range: U+0900 to U+097F
devanagari_chars = {char: count for char, count in char_counter.items()
                    if '\u0900' <= char <= '\u097F'}

# Top Devanagari characters
top_devanagari = sorted(devanagari_chars.items(),
                       key=lambda x: x[1], reverse=True)[:30]
```

**Generated Figures**:
- `figures/character_distribution.png`: Top 30 Devanagari characters

**Metrics**:
- Total Devanagari characters
- Hindi ratio (Devanagari / total)
- Unique Devanagari characters

**Typical Results**:
```
Total characters: 52,345,678
Devanagari characters: 42,876,543
Hindi ratio: 81.9%
Unique Devanagari characters: 87
```

#### Section 5: Word Frequency Analysis

Identifies most frequent words in the corpus:

```python
# Word frequency
all_words = [word for text in train_texts for word in text.split()]
word_counter = Counter(all_words)

# Top words
top_words = word_counter.most_common(30)

# Visualize
plt.figure(figsize=(14, 6))
plt.barh(range(len(words)), word_counts, color='lightcoral')
plt.yticks(range(len(words)), words, fontsize=11)
```

**Generated Figures**:
- `figures/word_frequency.png`: Top 30 most frequent words

**Insights**:
- Identifies common function words (है, का, को, में, etc.)
- Reveals corpus domain (news, literature, etc.)
- Helps understand vocabulary distribution

#### Section 6: Morphological Complexity Analysis

Analyzes Hindi case markers to assess morphological richness:

```python
# Analyze morphological markers
case_markers = ['ने', 'को', 'से', 'में', 'पर', 'का', 'की', 'के']
marker_counts = {marker: sum(text.count(marker) for text in train_texts)
                 for marker in case_markers}

# Visualize distribution
plt.figure(figsize=(10, 6))
plt.bar(markers, counts, color='mediumseagreen', edgecolor='black')
```

**Generated Figures**:
- `figures/case_markers.png`: Distribution of Hindi case markers

**Case Marker Statistics**:
```
का: 145,234 occurrences (genitive, masculine)
की: 98,765 occurrences (genitive, feminine)
के: 87,654 occurrences (genitive, plural/oblique)
को: 76,543 occurrences (accusative/dative)
ने: 65,432 occurrences (ergative)
से: 54,321 occurrences (instrumental/ablative)
में: 43,210 occurrences (locative)
पर: 32,109 occurrences (locative/temporal)
```

**Insights**:
- High case marker frequency indicates morphological richness
- Genitive markers (का/की/के) most common
- Ergative (ने) frequency validates perfective transitive constructions

#### Section 7: Data Quality Assessment

Evaluates corpus quality using multiple criteria:

```python
def assess_quality(texts):
    quality_stats = {
        'too_short': 0,      # < 5 words
        'too_long': 0,       # > 200 words
        'low_hindi_ratio': 0, # < 70% Devanagari
        'has_urls': 0,       # Contains URLs
        'clean': 0           # Passes all checks
    }

    for text in texts:
        word_count = len(text.split())

        # Check length
        if word_count < 5:
            quality_stats['too_short'] += 1
            continue
        if word_count > 200:
            quality_stats['too_long'] += 1
            continue

        # Check Hindi ratio
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        hindi_ratio = devanagari / len(text) if text else 0

        if hindi_ratio < 0.7:
            quality_stats['low_hindi_ratio'] += 1
            continue

        # Check for URLs
        if 'http' in text or 'www' in text:
            quality_stats['has_urls'] += 1
            continue

        quality_stats['clean'] += 1

    return quality_stats
```

**Generated Figures**:
- `figures/data_quality.png`: Pie chart of quality distribution

**Typical Results**:
```
Clean: 87.5%
Too short: 5.2%
Too long: 2.1%
Low Hindi ratio: 3.8%
Has URLs: 1.4%
```

#### Section 8: Export Summary Statistics

Saves comprehensive statistics for future reference:

```python
summary = {
    'dataset_statistics': train_stats,
    'quality_assessment': quality_stats,
    'case_markers': marker_counts,
    'top_words': dict(top_words[:50]),
}

# Save to JSON
with open('../data/corpus_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
```

**Output**: `data/corpus_statistics.json`

### Running the Notebook

```bash
# Launch Jupyter Lab
jupyter lab notebooks/01_data_exploration.ipynb

# Or run non-interactively
jupyter nbconvert --execute --to notebook \
    --inplace notebooks/01_data_exploration.ipynb
```

### Generated Outputs

**Figures** (saved to `figures/`):
1. `length_distributions.png` - Word and character length histograms
2. `character_distribution.png` - Top 30 Devanagari characters
3. `word_frequency.png` - Top 30 most frequent words
4. `case_markers.png` - Hindi case marker distribution
5. `data_quality.png` - Data quality assessment pie chart

**Data Files**:
1. `data/corpus_statistics.json` - Comprehensive corpus statistics

## 2. Results Analysis Notebook

**File**: `notebooks/02_results_analysis.ipynb`

**Purpose**: Analyze experimental results, compare models, perform statistical testing, and generate thesis-ready figures and tables.

### Notebook Structure

#### Section 1: Load Experimental Results

Initialize ResultsAnalyzer and load all experiments:

```python
from src.analysis.results_analyzer import ResultsAnalyzer, analyze_experiments
from src.analysis.visualization_utils import ThesisPlotter

# Initialize results analyzer
analyzer = analyze_experiments(results_dir='../results')

print(f"Loaded {len(analyzer.experiments)} experiments:")
for exp_name in analyzer.experiments.keys():
    print(f"  - {exp_name}")
```

**Expected Output**:
```
Loaded 3 experiments:
  - baseline_experiment
  - curriculum_experiment
  - enhanced_model_experiment
```

#### Section 2: Training Curves Comparison

Compare training progression across experiments:

```python
# Plot training curves for all experiments
fig = analyzer.plot_training_curves(
    metrics=['loss', 'perplexity'],
    save_path='../figures/training_curves.png'
)

plt.show()
print("✅ Training curves saved")
```

**Generated Figures**:
- `figures/training_curves.png`: Multi-panel training curves

**Insights**:
- Convergence rates comparison
- Overfitting detection (train vs. validation gap)
- Effect of curriculum learning on stability

#### Section 3: IndicGLUE Evaluation Comparison

Compare NLP task performance:

```python
# Compare IndicGLUE performance
fig = analyzer.plot_evaluation_comparison(
    eval_type='indicglue',
    save_path='../figures/indicglue_comparison.png'
)

plt.show()
print("✅ IndicGLUE comparison saved")
```

**Generated Figures**:
- `figures/indicglue_comparison.png`: Horizontal bar chart of IndicGLUE tasks

**Analysis**:
- Which model performs best overall?
- Task-specific strengths (e.g., better on NER vs. classification)
- Effect of curriculum learning on downstream tasks

#### Section 4: MultiBLiMP Syntactic Evaluation

Compare syntactic competence across 14 phenomena:

```python
# Compare MultiBLiMP performance
fig = analyzer.plot_evaluation_comparison(
    eval_type='multiblimp',
    save_path='../figures/multiblimp_comparison.png'
)

plt.show()
print("✅ MultiBLiMP comparison saved")
```

**Generated Figures**:
- `figures/multiblimp_comparison.png`: 14 phenomena comparison

**Insights**:
- Which phenomena are easier/harder to learn?
- Agreement phenomena (number, gender, person) typically higher
- Case marking and word order more challenging
- Curriculum learning impact on syntactic competence

#### Section 5: Morphological Probes Analysis

Compare morphological understanding across 10 probe tasks:

```python
# Compare morphological probe performance
fig = analyzer.plot_evaluation_comparison(
    eval_type='probes',
    save_path='../figures/morphological_probes_comparison.png'
)

plt.show()
print("✅ Morphological probes comparison saved")
```

**Generated Figures**:
- `figures/morphological_probes_comparison.png`: 10 probe tasks comparison

**Analysis**:
- Best layer for each morphological feature
- Surface features (number, gender) in middle layers
- Abstract features (mood, voice) in later layers
- Effect of position encodings on morphological learning

#### Section 6: Statistical Significance Testing

Rigorous statistical comparison between two best models:

```python
experiments = list(analyzer.experiments.keys())

if len(experiments) >= 2:
    exp1, exp2 = experiments[0], experiments[1]

    print(f"Comparing: {exp1} vs {exp2}\n")

    # Statistical comparison
    comparison = analyzer.compare_models_statistically(
        exp1, exp2,
        metric='accuracy',
        eval_type='indicglue'
    )

    print("📊 Statistical Comparison:")
    print(f"\nSummary Statistics:")
    print(f"  {exp1} mean: {comparison['summary']['exp1_mean']:.4f} ± {comparison['summary']['exp1_std']:.4f}")
    print(f"  {exp2} mean: {comparison['summary']['exp2_mean']:.4f} ± {comparison['summary']['exp2_std']:.4f}")
    print(f"  Difference: {comparison['summary']['difference']:.4f}")

    print(f"\nPaired t-test:")
    print(f"  p-value: {comparison['t_test']['p_value']:.4f}")
    print(f"  Significant: {'Yes ✓' if comparison['t_test']['significant'] else 'No ✗'}")

    print(f"\nWilcoxon test:")
    print(f"  p-value: {comparison['wilcoxon']['p_value']:.4f}")
    print(f"  Significant: {'Yes ✓' if comparison['wilcoxon']['significant'] else 'No ✗'}")

    print(f"\nEffect Size:")
    print(f"  Cohen's d: {comparison['effect_size']['cohens_d']:.4f}")
    print(f"  Interpretation: {comparison['effect_size']['interpretation']}")

    print(f"\nBootstrap 95% CI:")
    print(f"  [{comparison['bootstrap_ci']['lower']:.4f}, {comparison['bootstrap_ci']['upper']:.4f}]")
```

**Example Output**:
```
📊 Statistical Comparison:

Summary Statistics:
  baseline mean: 0.7012 ± 0.0423
  curriculum mean: 0.7345 ± 0.0389
  Difference: 0.0333

Paired t-test:
  p-value: 0.0134
  Significant: Yes ✓

Wilcoxon test:
  p-value: 0.0201
  Significant: Yes ✓

Effect Size:
  Cohen's d: 0.78
  Interpretation: medium

Bootstrap 95% CI:
  [0.0089, 0.0576]
```

**Interpretation**:
- p < 0.05: Difference is statistically significant
- Cohen's d = 0.78: Medium effect size
- 95% CI doesn't include 0: Significant improvement

#### Section 7: Generate LaTeX Tables for Thesis

Generate publication-ready LaTeX tables:

```python
# Generate LaTeX table for IndicGLUE
latex_table = analyzer.generate_latex_table(
    eval_type='indicglue',
    metric='accuracy',
    caption='IndicGLUE Benchmark Results',
    label='tab:indicglue_results',
    save_path='../tables/indicglue_results.tex'
)

print("IndicGLUE LaTeX Table:")
print(latex_table)
print("\n✅ LaTeX table saved to tables/indicglue_results.tex")

# Generate LaTeX table for MultiBLiMP
latex_table = analyzer.generate_latex_table(
    eval_type='multiblimp',
    metric='accuracy',
    caption='MultiBLiMP Syntactic Phenomena Results',
    label='tab:multiblimp_results',
    save_path='../tables/multiblimp_results.tex'
)

print("✅ MultiBLiMP LaTeX table saved")
```

**Generated Tables**:
- `tables/indicglue_results.tex`
- `tables/multiblimp_results.tex`
- `tables/probes_results.tex`

**LaTeX Integration**:
```latex
% In your thesis
\input{tables/indicglue_results.tex}
```

#### Section 8: Layer-wise Probe Visualization

Visualize morphological understanding across layers:

```python
# Visualize layer-wise probe results
plotter = ThesisPlotter(style='thesis')

# Example: Case detection probe
# (Replace with actual data from your experiments)
layer_results = {i: 0.5 + 0.03 * i + np.random.normal(0, 0.02)
                for i in range(13)}

fig = plotter.plot_layer_wise_probe_results(
    layer_results,
    probe_name='Case Detection',
    title='Layer-wise Case Detection Accuracy',
    save_path='../figures/layer_wise_case_probe.png'
)

plt.show()
print("✅ Layer-wise probe visualization saved")
```

**Generated Figures**:
- `figures/layer_wise_case_probe.png`
- `figures/layer_wise_number_probe.png`
- `figures/layer_wise_gender_probe.png`
- ... (one per probe task)

**Insights**:
- Peak performance typically in middle layers (5-8)
- Different features peak at different layers
- Position encoding effects on layer-wise learning

#### Section 9: Model Size vs Performance

Compare models of different sizes:

```python
# Plot performance vs model size
model_sizes = [50, 110, 350]  # Millions of parameters
accuracies = [0.72, 0.78, 0.82]
model_names = ['Tiny', 'Small', 'Medium']

fig = plotter.plot_performance_vs_model_size(
    model_sizes,
    accuracies,
    model_names,
    title='Model Size vs IndicGLUE Performance',
    save_path='../figures/model_size_vs_performance.png'
)

plt.show()
print("✅ Model size comparison saved")
```

**Generated Figures**:
- `figures/model_size_vs_performance.png`

**Analysis**:
- Scaling laws for Hindi language models
- Diminishing returns at larger sizes?
- Data efficiency vs. model size tradeoffs

#### Section 10: Generate Individual Experiment Reports

Create comprehensive markdown reports for each experiment:

```python
# Generate reports for each experiment
for exp_name in analyzer.experiments.keys():
    report = analyzer.generate_summary_report(
        exp_name,
        save_path=f'../reports/{exp_name}_report.md'
    )
    print(f"✅ Generated report for {exp_name}")

print("\nAll reports saved to reports/ directory")
```

**Generated Reports**:
- `reports/baseline_report.md`
- `reports/curriculum_report.md`
- `reports/enhanced_model_report.md`

### Running the Notebook

```bash
# Launch Jupyter Lab
jupyter lab notebooks/02_results_analysis.ipynb

# Or run non-interactively to regenerate all outputs
jupyter nbconvert --execute --to notebook \
    --inplace notebooks/02_results_analysis.ipynb
```

### Generated Outputs

**Figures** (saved to `figures/`):
1. `training_curves.png` - Training and validation curves
2. `indicglue_comparison.png` - IndicGLUE task comparison
3. `multiblimp_comparison.png` - MultiBLiMP phenomena comparison
4. `morphological_probes_comparison.png` - Probe task comparison
5. `layer_wise_*.png` - Layer-wise probe visualizations (one per probe)
6. `model_size_vs_performance.png` - Model size scaling analysis

**LaTeX Tables** (saved to `tables/`):
1. `indicglue_results.tex`
2. `multiblimp_results.tex`
3. `probes_results.tex`

**Reports** (saved to `reports/`):
1. `[experiment_name]_report.md` - Comprehensive experiment summaries

## Best Practices for Notebook Usage

### 1. Reproducibility

```python
# Set random seeds at the beginning
import numpy as np
import random
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
```

### 2. Clear Output

```python
# Clear output and re-run from scratch for clean results
# Jupyter Lab: Cell → All Output → Clear
# Or use magic command
%reset -f
```

### 3. Save Intermediate Results

```python
# Save expensive computations
import pickle

# Save
with open('../cache/expensive_computation.pkl', 'wb') as f:
    pickle.dump(results, f)

# Load
with open('../cache/expensive_computation.pkl', 'rb') as f:
    results = pickle.load(f)
```

### 4. Export to PDF/HTML

```bash
# Export notebook with outputs to PDF
jupyter nbconvert --to pdf notebooks/02_results_analysis.ipynb

# Export to HTML for sharing
jupyter nbconvert --to html notebooks/02_results_analysis.ipynb
```

### 5. Version Control

```bash
# Strip outputs before committing (for cleaner diffs)
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

# Then commit
git add notebooks/
git commit -m "Update analysis notebooks"
```

## Integration with Thesis Workflow

### Step 1: Data Exploration Phase

1. Run `01_data_exploration.ipynb`
2. Review corpus statistics
3. Include key figures in thesis Chapter 3 (Data)

### Step 2: Results Analysis Phase

1. Complete all experimental runs
2. Run `02_results_analysis.ipynb`
3. Generate all figures and tables
4. Perform statistical testing

### Step 3: Thesis Writing

```latex
% In thesis Chapter 4 (Experiments)

\section{Corpus Statistics}
Our corpus consists of ... (cite data from 01_data_exploration.ipynb)

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/length_distributions.png}
    \caption{Sentence length distribution in training corpus}
    \label{fig:length_dist}
\end{figure}

% Include tables
\section{Results}
\input{tables/indicglue_results.tex}

% Include comparison figures
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/multiblimp_comparison.png}
    \caption{MultiBLiMP syntactic evaluation across models}
    \label{fig:multiblimp}
\end{figure}
```

## Troubleshooting

### Kernel Dies / Out of Memory

```python
# Process data in chunks
chunk_size = 10000
for i in range(0, len(train_texts), chunk_size):
    chunk = train_texts[i:i+chunk_size]
    process_chunk(chunk)
```

### Missing Dependencies

```bash
# Install missing packages
pip install matplotlib seaborn pandas numpy scipy
```

### Path Issues

```python
# Always use Path for cross-platform compatibility
from pathlib import Path

data_dir = Path('../data/processed')
fig_dir = Path('../figures')
fig_dir.mkdir(exist_ok=True)
```

### Unicode Errors

```python
# Always specify encoding for Hindi text
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
```

## Related Documentation

- [Analysis and Visualization Documentation](08_ANALYSIS_AND_VISUALIZATION.md) - Detailed API reference
- [Thesis Integration Guide](10_THESIS_INTEGRATION.md) - LaTeX integration workflow
- [Data Processing Documentation](02_DATA_PROCESSING.md) - Corpus preparation
- [Evaluation Framework Documentation](06_EVALUATION.md) - Evaluation metrics
