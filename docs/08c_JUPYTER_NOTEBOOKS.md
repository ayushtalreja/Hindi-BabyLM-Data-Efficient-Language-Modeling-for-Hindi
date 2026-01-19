# Jupyter Notebooks for Interactive Analysis

## Overview

The Hindi BabyLM project includes interactive Jupyter notebooks for data exploration and results analysis. These notebooks provide hands-on analysis workflows with visualizations, statistical testing, and thesis-ready outputs.

**Available Notebooks**:
1. **01_data_exploration.ipynb**: Comprehensive corpus analysis and data quality assessment
2. **02_results_analysis.ipynb**: Experimental results analysis with statistical testing

## Architecture

```
Data/Results
     ↓
┌────────────────────────────────────┐
│  01_data_exploration.ipynb         │
│  • Load corpus data                │
│  • Statistical analysis            │
│  • Distribution analysis           │
│  • Quality assessment              │
│  • Cross-source comparison         │
└────────┬───────────────────────────┘
     ↓
┌────────────────────────────────────┐
│  02_results_analysis.ipynb         │
│  • Load experiment results         │
│  • Training curve analysis         │
│  • Evaluation comparisons          │
│  • Statistical significance        │
│  • LaTeX table generation          │
└────────┬───────────────────────────┘
     ↓
  Interactive Insights
  • Live visualizations
  • Exploratory analysis
  • Publication outputs
```

## 1. Data Exploration Notebook

**Location**: `notebooks/01_data_exploration.ipynb`

**Purpose**: Interactive corpus analysis and data quality assessment

### Sections

#### Section 0: Setup & Configuration

- Project root path detection (handles both local and cluster environments)
- Hindi font configuration for Devanagari script visualization
- Font discovery and registration (Noto Sans Devanagari)
- Import project modules and set plotting styles

**Key Features**:
```python
# Automatic font handling for Hindi display
def find_and_register_devanagari_font():
    """
    Find and load the best available font for Devanagari script.

    Priority:
    1. Custom fonts from project's fonts/ directory
    2. System-installed Devanagari fonts
    3. Fallback to DejaVu Sans (limited support)
    """
    # Tries multiple font sources
    # Registers fonts with matplotlib
    # Returns FontProperties for consistent rendering
```

#### Section 1: Data Loading

- Load processed corpus splits (train/val/test)
- Display basic statistics (size, word count, character count)
- Source distribution visualization

**Example Output**:
```
Corpus Statistics:
  Train: 10,234,567 words | 58,123,456 characters
  Val:   1,023,456 words  | 5,812,345 characters
  Test:  1,023,456 words  | 5,812,345 characters

Source Distribution:
  IndicCorp: 60%
  Wikipedia: 25%
  Children's Books: 10%
  Dialogue: 5%
```

#### Section 2: Enhanced Basic Statistics

- Vocabulary size and coverage
- Word length distribution
- Sentence length distribution
- Character frequency analysis
- Token statistics

**Visualizations**:
- Histogram of word lengths
- Sentence length distribution
- Top N most frequent words (with Hindi font support)
- Character frequency bar chart

#### Section 3: Advanced Distribution Analysis

- Zipf's law analysis (word frequency vs rank)
- Vocabulary growth curve
- Type-token ratio (TTR) analysis
- Lexical diversity metrics
- Hapax legomena (words appearing once)

**Key Insights**:
- Zipf's law compliance indicates natural language distribution
- Vocabulary growth shows corpus coverage
- TTR measures lexical richness
- Hapax ratio indicates vocabulary diversity

#### Section 4: Deep Character & Script Analysis

- Devanagari character distribution
- Consonant vs vowel ratio
- Diacritic usage analysis
- Conjunct character frequency
- Script purity (% Devanagari vs other scripts)

**Example Analysis**:
```python
# Devanagari character categories
consonants = set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
vowels = set('अआइईउऊऋएऐओऔ')
vowel_signs = set('ािीुूृेैोौ')
special = set('ंःँ्')

# Analyze distribution
consonant_freq = count_chars(corpus, consonants)
vowel_freq = count_chars(corpus, vowels)
# Visualize with matplotlib
```

#### Section 5: Advanced Word-Level Analysis

- Part-of-speech distribution (if tagged)
- Named entity frequency
- Word form variation
- Morphological complexity
- Compound word analysis

**Morphological Metrics**:
- Average morphemes per word
- Inflection frequency
- Derivation patterns
- Compounding rate

#### Section 6: Morphological Analysis

- Case marker distribution (ergative, dative, locative, etc.)
- Verbal morphology (aspect, tense, mood)
- Gender agreement patterns
- Number marking frequency
- Honorific usage

**Example Patterns**:
```python
# Case markers
case_markers = {
    'Ergative (-ने)': count_suffix(corpus, 'ने'),
    'Dative (-को)': count_suffix(corpus, 'को'),
    'Locative (-में)': count_suffix(corpus, 'में'),
    # ...
}

# Visualize distribution
plot_case_distribution(case_markers)
```

#### Section 7: Linguistic Phenomena Detection

- Subject-verb agreement patterns
- Word order variations (SOV, OVS, etc.)
- Complex predicate constructions
- Reduplication occurrences
- Loan word detection

**Phenomena Metrics**:
- SOV sentence ratio
- Agreement violations (for testing)
- Compound verb frequency
- Reduplication types

#### Section 8: Data Quality Assessment

- Language detection scores
- Encoding issues detection
- Duplicate text identification
- Noise ratio (special characters, numbers)
- Sentence completeness

**Quality Checks**:
```python
quality_metrics = {
    'devanagari_ratio': 0.956,      # % Devanagari characters
    'complete_sentences': 0.892,    # % ending with punctuation
    'avg_sentence_words': 12.4,     # Average words per sentence
    'duplicate_ratio': 0.023,       # % duplicate texts
    'noise_ratio': 0.012           # % non-linguistic content
}
```

#### Section 9: Cross-Source Comparative Analysis

- Compare statistics across data sources
- Source-specific vocabulary
- Domain-specific characteristics
- Quality variations by source
- Source balance in splits

**Comparative Metrics**:
- Vocabulary overlap between sources
- Average sentence length by source
- Morphological complexity by domain
- Quality scores by source

#### Section 10: Summary Statistics & Export

- Generate executive summary
- Create comparison tables
- Export statistics to CSV/JSON
- Generate visualizations for thesis
- Save analysis report

**Outputs**:
- `data_exploration_summary.md`: Markdown report
- `corpus_statistics.csv`: Detailed statistics
- `figures/data_*.png`: Visualization figures
- `data_quality_report.json`: Quality metrics

### Usage

```bash
# Start Jupyter
cd notebooks/
jupyter notebook 01_data_exploration.ipynb

# Or use JupyterLab
jupyter lab 01_data_exploration.ipynb
```

**Running on Cluster (LRZ)**:
```bash
# Start Jupyter on compute node
jupyter notebook --no-browser --port=8888

# On local machine, create SSH tunnel:
ssh -L 8888:localhost:8888 username@lrz_cluster
```

## 2. Results Analysis Notebook

**Location**: `notebooks/02_results_analysis.ipynb`

**Purpose**: Comprehensive experimental results analysis with statistical testing

### Key Features

- **Hindi Font Support**: Automatic Devanagari font detection and registration
- **Multi-Experiment Loading**: Analyze all experiments in results directory
- **Statistical Rigor**: Paired t-tests, Wilcoxon tests, effect sizes, bootstrap CIs
- **Publication Outputs**: LaTeX tables, high-resolution figures (300 DPI)
- **Flexible Configuration**: Easy experiment selection for comparisons

### Notebook Structure

#### Cell 0: Font Configuration

```python
# Finds and registers Devanagari fonts
def find_and_register_devanagari_font():
    """
    Priority:
    1. Custom fonts from project's fonts/ directory (Noto Sans Devanagari)
    2. System-installed Devanagari fonts
    3. Fallback to DejaVu Sans

    Returns:
        tuple: (font_name, FontProperties object)
    """
```

**Supported Fonts** (in priority order):
1. Noto Sans Devanagari (best)
2. Kohinoor Devanagari
3. Arial Unicode MS
4. Mangal
5. Lohit Devanagari
6. DejaVu Sans (fallback)

#### Cell 1: YAML Config Loading Fix

Registers custom YAML constructors for loading experiment configs:

```python
# Register constructors for model configs
yaml.add_constructor(
    'tag:yaml.org,2002:python/object:src.utils.experiment_config.GPTModelConfig',
    gpt_config_constructor,
    Loader=yaml.SafeLoader
)
```

This fixes issues when loading experiment configs that contain custom Python objects.

#### Cell 2: Load Experimental Results

```python
from src.analysis.results_analyzer import analyze_experiments

# Load all experiments
analyzer = analyze_experiments(results_dir='results')

print(f"Loaded {len(analyzer.experiments)} experiments:")
for exp_name in analyzer.experiments.keys():
    print(f"  - {exp_name}")
```

**Automatic Experiment Discovery**:
- Scans results directory
- Loads all experiments with evaluation results
- Displays basic info (epochs, final loss, available evaluations)

**Experiment Configuration**:
```python
# Configure experiments for binary comparisons
COMPARISON_EXP1 = 'baseline_experiment'
COMPARISON_EXP2 = 'enhanced_experiment'

# Or use auto-selection (alphabetically sorted)
ALL_EXPERIMENTS = sorted(analyzer.experiments.keys())
COMPARISON_EXP1 = ALL_EXPERIMENTS[0]
COMPARISON_EXP2 = ALL_EXPERIMENTS[1]
```

#### Cell 3: Comprehensive Results Overview Table

Multi-experiment comparison table with color-coded best scores:

```python
# Creates styled DataFrame
results_df = create_comprehensive_results_table(analyzer)
styled_results = style_results_table(results_df)

# Display with highlighting:
# - Best score per task/phenomenon highlighted in green
# - Sorted by model type, dataset size, vocab size
# - Includes IndicGLUE tasks + average
# - Includes MultiBLiMP phenomena + average
```

**Table Columns**:
- Experiment name
- IndicGLUE tasks (BBC, Wiki, COPA, Movie, Product, Discourse, CSQA)
- IndicGLUE average
- MultiBLiMP phenomena (SV-#, SV-G, SV-P, SP-#, SP-G)
- MultiBLiMP average

#### Cell 4: Training Curves Comparison

Plot training progression for all experiments:

```python
fig = analyzer.plot_training_curves(
    experiment_names=ALL_EXPERIMENTS,
    metrics=['train_loss', 'val_loss', 'perplexity'],
    save_path='figures/training_curves.png'
)
```

**Multi-Panel Figure**:
- One subplot per metric
- All experiments overlaid
- Markers for easy identification
- Grid for precise reading

#### Cell 5: Training Dynamics Analysis

**4-Panel Analysis** (requires 2 experiments for comparison):

1. **Training and Validation Loss**:
   - Dual curves (train/val) for both experiments
   - Vertical lines marking best validation epochs
   - Legend with best epoch identification

2. **Perplexity Over Training**:
   - Validation perplexity curves
   - Lower is better
   - Shows model uncertainty

3. **Generalization Gap**:
   - `val_loss - train_loss` for each epoch
   - Positive values indicate overfitting
   - Zero line as reference

4. **Epoch-to-Epoch Improvement**:
   - `-diff(val_loss)` shows learning progress
   - Positive values = improvement
   - Negative values = degradation

**Key Insights Output**:
```
Key Insights:
  baseline: Best val loss = 0.3822 at epoch 10
  enhanced: Best val loss = 0.3456 at epoch 8
  Final perplexity - Exp1: 12.34, Exp2: 10.23
```

#### Cell 6: Detailed IndicGLUE Analysis

**Per-Task Comparison with Confidence Intervals**:

```python
# Horizontal bar chart with:
# - Two bars per task (one per experiment)
# - Error bars showing 95% CI
# - Tasks sorted by improvement (best first)
# - Improvement values annotated
```

**Features**:
- Safe data extraction (handles missing tasks)
- Confidence interval visualization
- Color coding (improvement in green, decline in red)
- Tasks with missing data are skipped with warnings

#### Cell 7: Confusion Matrices

**Per-Experiment Confusion Matrix Figures**:

```python
# For each experiment:
# - BBC Articles Classification
# - Discourse Mode
# - Movie Review Sentiment

# Each with:
# - Normalized confusion matrix (0-1 scale)
# - Seaborn heatmap visualization
# - Class name labels
# - Color bar for interpretation
```

**Saved Files**:
- `confusion_matrices_{experiment_name}.png` for each experiment
- Multiple tasks in vertical layout per file

#### Cell 8: IndicGLUE Comparison (All Experiments)

Horizontal bar chart comparing all experiments across IndicGLUE tasks:

```python
fig = analyzer.plot_evaluation_comparison(
    experiment_names=ALL_EXPERIMENTS,
    eval_type='indicglue',
    save_path='figures/indicglue_comparison.png'
)
```

**Output**: Wide figure with all tasks and all experiments side-by-side

#### Cell 9: MultiBLiMP Comparison (All Experiments)

Similar to IndicGLUE but for syntactic phenomena:

```python
fig = analyzer.plot_evaluation_comparison(
    experiment_names=ALL_EXPERIMENTS,
    eval_type='multiblimp',
    save_path='figures/multiblimp_comparison.png'
)
```

#### Cell 10: Detailed MultiBLiMP Analysis

**2-Panel Analysis** (requires 2 experiments):

1. **Accuracy Comparison**:
   - Horizontal bars for each phenomenon
   - Two bars per phenomenon (one per experiment)

2. **Loss Difference Analysis**:
   - Mean loss difference (grammatical - ungrammatical)
   - Error bars showing standard deviation
   - Lower/more negative = better grammaticality judgment

**Key Metrics**:
- Accuracy: % of minimal pairs where model prefers grammatical sentence
- Loss difference: How much lower loss for grammatical vs ungrammatical

#### Cell 11: Statistical Significance Testing

**Rigorous Statistical Comparison** (requires exactly 2 experiments):

```python
# IndicGLUE statistical testing
comparison = analyzer.compare_models_statistically(
    COMPARISON_EXP1, COMPARISON_EXP2,
    metric='accuracy',
    eval_type='indicglue'
)
```

**Output**:
```
📊 IndicGLUE Statistical Comparison:

Summary Statistics:
  baseline mean: 0.6834 ± 0.0423
  enhanced mean: 0.7256 ± 0.0312
  Difference: 0.0422
  Number of tasks: 8

Paired t-test:
  p-value: 0.012345
  Significant: Yes ✓

Wilcoxon signed-rank test:
  p-value: 0.015678
  Significant: Yes ✓

Effect Size:
  Cohen's d: 0.6234
  Interpretation: medium

Bootstrap 95% CI for difference:
  [0.0123, 0.0721]
```

Also performs same analysis for MultiBLiMP phenomena.

#### Cell 12: LaTeX Table Generation

**Generate Publication-Ready Tables**:

```python
# IndicGLUE table
latex_table = analyzer.generate_latex_table(
    experiment_names=ALL_EXPERIMENTS,
    eval_type='indicglue',
    metric='accuracy',
    caption='IndicGLUE Benchmark Results',
    label='tab:indicglue_results',
    save_path='tables/indicglue_results.tex'
)

# MultiBLiMP table
latex_table = analyzer.generate_latex_table(
    experiment_names=ALL_EXPERIMENTS,
    eval_type='multiblimp',
    metric='accuracy',
    caption='MultiBLiMP Results',
    label='tab:multiblimp_results',
    save_path='tables/multiblimp_results.tex'
)
```

**Features**:
- Best values bolded
- Average row computed
- Booktabs formatting
- Ready for `\input{}` in LaTeX

#### Cell 13: Comprehensive Comparison Table

All experiments, all key metrics in one table:

```python
comparison_df = pd.DataFrame({
    'Metric': [
        'Final Train Loss',
        'Best Val Loss',
        'Final Perplexity',
        'IndicGLUE Avg',
        'MultiBLiMP Avg'
    ],
    'Exp1': [...],
    'Exp2': [...],
    # ... all experiments
})
```

**Exported To**:
- `tables/experiment_comparison.csv`
- `tables/experiment_comparison.tex`

#### Cell 14: Generate Individual Reports

```python
# For each experiment
for exp_name in ALL_EXPERIMENTS:
    report = analyzer.generate_summary_report(
        exp_name,
        save_path=f'reports/{exp_name}_report.md'
    )
```

**Report Contents**:
- Metadata (timestamp, git commit, device)
- Training summary (epochs, losses, time)
- Evaluation results (all benchmarks)
- Formatted markdown

### Usage

```bash
# Start Jupyter
cd notebooks/
jupyter notebook 02_results_analysis.ipynb

# Or run all cells programmatically
jupyter nbconvert --to notebook --execute 02_results_analysis.ipynb
```

### Output Files Generated

After running the notebook:

```
├── figures/
│   ├── training_curves.png               # All experiments training
│   ├── training_dynamics.png             # 4-panel analysis
│   ├── indicglue_detailed_ci.png         # Per-task with CI
│   ├── confusion_matrices_*.png          # Per experiment
│   ├── indicglue_comparison.png          # All experiments
│   ├── multiblimp_comparison.png         # All experiments
│   └── multiblimp_detailed.png           # Loss difference analysis
├── tables/
│   ├── indicglue_results.tex             # LaTeX table
│   ├── multiblimp_results.tex            # LaTeX table
│   ├── experiment_comparison.csv         # All metrics CSV
│   └── experiment_comparison.tex         # All metrics LaTeX
└── reports/
    ├── {experiment_1}_report.md
    ├── {experiment_2}_report.md
    └── ...
```

## 3. Best Practices

### Font Handling for Hindi

**Always register fonts first**:
```python
# At notebook start
from matplotlib import font_manager as fm
font_path = 'fonts/Noto_Sans_Devanagari/static/NotoSansDevanagari-Regular.ttf'
fm.fontManager.addfont(font_path)

# Set as default
plt.rcParams['font.family'] = 'Noto Sans Devanagari'
```

**Test font rendering**:
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.text(0.5, 0.5, 'हिन्दी परीक्षण', fontsize=20, ha='center')
plt.show()
```

### Experiment Configuration

**Flexible comparison setup**:
```python
# Option 1: Manual selection
COMPARISON_EXP1 = 'specific_experiment_name'
COMPARISON_EXP2 = 'another_experiment_name'

# Option 2: Automatic (first two alphabetically)
ALL_EXPERIMENTS = sorted(analyzer.experiments.keys())
COMPARISON_EXP1 = ALL_EXPERIMENTS[0]
COMPARISON_EXP2 = ALL_EXPERIMENTS[1]

# Option 3: Filter by pattern
GPT_EXPERIMENTS = [e for e in ALL_EXPERIMENTS if 'gpt' in e.lower()]
DEBERTA_EXPERIMENTS = [e for e in ALL_EXPERIMENTS if 'deberta' in e.lower()]
```

### Error Handling

**Graceful degradation**:
```python
# Check if sufficient experiments
if len(ALL_EXPERIMENTS) < 2:
    print("⚠️  Need at least 2 experiments for comparison")
    print(f"   Found: {len(ALL_EXPERIMENTS)}")
else:
    # Run comparison analyses
```

**Handle missing data**:
```python
# Check for task existence
if task not in experiment_results:
    print(f"⚠️  Task {task} not found in {exp_name}")
    continue

# Safe data extraction
accuracy = task_results.get('accuracy')
if accuracy is None and 'metrics_with_ci' in task_results:
    accuracy = task_results['metrics_with_ci'].get('accuracy', {}).get('value')
```

### Output Organization

**Consistent file naming**:
```python
# Include experiment names in filenames
safe_exp_name = exp_name.replace('/', '_').replace(' ', '_')
save_path = f'figures/analysis_{safe_exp_name}.png'

# Include timestamps for versioning
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = f'figures/analysis_{timestamp}.png'
```

### Memory Management

**Large corpus handling**:
```python
# Process in chunks
chunk_size = 10000
for i in range(0, len(corpus), chunk_size):
    chunk = corpus[i:i+chunk_size]
    process_chunk(chunk)

# Clear memory after heavy operations
import gc
del large_dataframe
gc.collect()
```

## 4. Troubleshooting

### Font Issues

**Problem**: Hindi text displays as boxes/squares

**Solutions**:
1. Install Noto Sans Devanagari fonts system-wide
2. Place fonts in `fonts/` directory
3. Use fallback: `plt.rcParams['font.family'] = 'DejaVu Sans'`

### YAML Loading Errors

**Problem**: `ConstructorError: could not determine a constructor for the tag...`

**Solution**: Register custom constructors (Cell 2)
```python
yaml.add_constructor('tag:...', constructor_func, Loader=yaml.SafeLoader)
```

### Missing Experiment Data

**Problem**: Some experiments load but lack evaluation data

**Solution**: Check experiment completion
```python
# Verify experiment structure
exp_dir = Path(f'results/{exp_name}')
required_files = ['metadata.json', 'training_summary.json', 'evaluation_results.json']

for file in required_files:
    if not (exp_dir / file).exists():
        print(f"⚠️  Missing: {file}")
```

### Kernel Crashes

**Problem**: Kernel dies when loading large experiments

**Solution**: Increase memory limits
```python
# Load experiments selectively
experiments_to_load = ['exp1', 'exp2']  # Don't load all
for exp in experiments_to_load:
    analyzer.load_experiment(exp)
```

## 5. Extending the Notebooks

### Adding Custom Analysis

**Example: Add new visualization**:
```python
# Create new cell
def plot_custom_metric(analyzer, metric_name):
    """Custom visualization for specific metric"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name in analyzer.experiments.keys():
        # Extract metric
        metric_data = extract_metric(analyzer, exp_name, metric_name)

        # Plot
        ax.plot(metric_data, label=exp_name)

    ax.set_xlabel('X-axis Label')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'figures/{metric_name}_comparison.png', dpi=300)
    plt.show()

# Use it
plot_custom_metric(analyzer, 'learning_rate')
```

### Adding Statistical Tests

**Example: Add ANOVA for multi-experiment comparison**:
```python
from scipy import stats

def compare_multiple_experiments(analyzer, exp_names, metric='accuracy'):
    """ANOVA for comparing 3+ experiments"""
    groups = []

    for exp_name in exp_names:
        # Extract metric values across tasks
        values = extract_task_metrics(analyzer, exp_name, metric)
        groups.append(values)

    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    print(f"One-way ANOVA Results:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant: {p_value < 0.05}")

    return f_stat, p_value

# Use for 3+ experiments
compare_multiple_experiments(analyzer, ALL_EXPERIMENTS)
```

## Related Documentation

- [Analysis and Visualization](08_ANALYSIS_AND_VISUALIZATION.md) - ResultsAnalyzer and ThesisPlotter API
- [Tokenizer Evaluation](08b_TOKENIZER_EVALUATION.md) - Morphological analysis and tokenizer comparison
- [Evaluation Framework](06_EVALUATION.md) - Understanding evaluation metrics
- [Training Pipeline](05_TRAINING.md) - How results are generated
