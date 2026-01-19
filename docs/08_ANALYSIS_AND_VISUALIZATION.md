# Analysis and Visualization Tools

## Overview

The Hindi BabyLM project provides comprehensive analysis and visualization tools for processing experimental results, generating thesis-ready figures, and performing statistical comparisons. These tools bridge the gap between raw experimental results and publication-quality outputs.

**Key Components**:
1. **ResultsAnalyzer**: Statistical analysis, model comparison, LaTeX table generation (`src/analysis/results_analyzer.py`)
2. **ThesisPlotter**: Publication-ready visualizations with thesis formatting (`src/analysis/visualization_utils.py`)
3. **ComparativeAnalyzer**: Advanced multi-model comparison with HTML/PDF reports (`src/evaluation/comparative_analysis.py`)
4. **Tokenizer Evaluation**: Morphological preservation and compression metrics (`src/tokenization/`)
5. **Jupyter Notebooks**: Interactive data exploration and results analysis

## Architecture

```
Experimental Results
         ↓
┌────────────────────────────────────┐
│    ResultsAnalyzer                 │
│  • Load experiment results         │
│  • Statistical testing             │
│  • Model comparison                │
│  • LaTeX table generation          │
└────────┬───────────────────────────┘
         ↓
┌────────────────────────────────────┐
│    ThesisPlotter                   │
│  • Training curves                 │
│  • Evaluation comparisons          │
│  • Layer-wise probe visualization  │
│  • Publication-quality formatting  │
└────────┬───────────────────────────┘
         ↓
┌────────────────────────────────────┐
│    ComparativeAnalyzer             │
│  • Multi-model comparison          │
│  • Radar plots                     │
│  • Regression analysis             │
│  • HTML/PDF report generation      │
└────────┬───────────────────────────┘
         ↓
    Thesis Outputs
    • Figures (PNG/PDF)
    • LaTeX tables (.tex)
    • HTML reports (.html)
    • PDF reports (.pdf)
    • Markdown reports (.md)
```

## 1. ResultsAnalyzer

**Location**: `src/analysis/results_analyzer.py:42`

**Purpose**: Comprehensive analysis of experimental results with statistical rigor

### Features

- **Multi-experiment loading**: Load and compare results from multiple training runs
- **Statistical testing**: t-tests, Wilcoxon tests, effect size calculations, bootstrap CIs
- **Publication outputs**: LaTeX tables, markdown reports, CSV summaries
- **Training analysis**: Convergence curves, learning rate schedules, loss trajectories
- **Evaluation analysis**: Per-task breakdowns, overall metrics, comparative analysis

### Initialization

```python
from src.analysis.results_analyzer import ResultsAnalyzer, analyze_experiments

# Method 1: Initialize and load experiments manually
analyzer = ResultsAnalyzer(results_dir='results')
analyzer.load_experiment('baseline_experiment')
analyzer.load_experiment('curriculum_experiment')

# Method 2: Convenience function (loads all experiments)
analyzer = analyze_experiments(results_dir='results')

print(f"Loaded {len(analyzer.experiments)} experiments")
```

### Core Methods

#### `load_experiment()` (line 68)

Load a single experiment's results:

```python
results = analyzer.load_experiment('baseline_experiment')

# Results structure:
# {
#   'metadata': {...},           # Timestamp, git commit, device
#   'training': {...},           # Training history, final metrics
#   'evaluation': {...},         # IndicGLUE, MultiBLiMP, Probes results
#   'config': {...}              # Experiment configuration
# }
```

#### `load_all_experiments()` (line 116)

Load all experiments from the results directory:

```python
num_loaded = analyzer.load_all_experiments()
print(f"Loaded {num_loaded} experiments")

# Access experiments
for exp_name, exp_data in analyzer.experiments.items():
    print(f"{exp_name}: {exp_data.keys()}")
```

#### `get_training_history()` (line 138)

Extract training history as a pandas DataFrame:

```python
history_df = analyzer.get_training_history('baseline_experiment')

# DataFrame columns:
# - epoch: Epoch number
# - train_loss: Training loss per epoch
# - val_loss: Validation loss per epoch
# - perplexity: Validation perplexity per epoch
# Additional metrics if available in training logs
```

**Note**: This method handles both old and new training log formats:
- New format: `metrics_history` dict with metric names as keys
- Old format: `history` list of dicts with per-epoch metrics

#### `plot_training_curves()` (line 188)

Visualize training progression:

```python
# Plot specific metrics for specific experiments
fig = analyzer.plot_training_curves(
    experiment_names=['baseline', 'curriculum'],
    metrics=['train_loss', 'val_loss', 'perplexity'],
    save_path='figures/training_curves.png'
)

# Plot all experiments, all metrics
fig = analyzer.plot_training_curves()
```

**Output**: Multi-panel figure with one subplot per metric, comparing all specified experiments.

#### `plot_evaluation_comparison()` (line 236)

Compare evaluation results across experiments:

```python
# IndicGLUE comparison
fig = analyzer.plot_evaluation_comparison(
    experiment_names=['baseline', 'curriculum', 'enhanced'],
    eval_type='indicglue',
    save_path='figures/indicglue_comparison.png'
)

# MultiBLiMP comparison
fig = analyzer.plot_evaluation_comparison(
    eval_type='multiblimp',
    save_path='figures/multiblimp_comparison.png'
)
```

**Output**: Horizontal bar chart with tasks/phenomena on y-axis, accuracy on x-axis, one bar per experiment.

**Supported eval_types**:
- `'indicglue'`: IndicGLUE benchmark tasks
- `'multiblimp'`: MultiBLiMP syntactic phenomena

#### `compare_models_statistically()` (line 298)

Rigorous statistical comparison between two models:

```python
comparison = analyzer.compare_models_statistically(
    exp1='baseline',
    exp2='curriculum',
    metric='accuracy',
    eval_type='indicglue'
)

print("Statistical Comparison Results:")
print(f"Mean difference: {comparison['summary']['difference']:.4f}")
print(f"t-test p-value: {comparison['t_test']['p_value']:.4f}")
print(f"Significant: {comparison['t_test']['significant']}")
print(f"Effect size: {comparison['effect_size']['cohens_d']:.4f} ({comparison['effect_size']['interpretation']})")
print(f"Bootstrap 95% CI: [{comparison['bootstrap_ci']['lower']:.4f}, {comparison['bootstrap_ci']['upper']:.4f}]")
```

**Statistical Tests Performed**:
1. **Paired t-test**: Tests if mean difference is significantly different from zero
2. **Wilcoxon signed-rank test**: Non-parametric alternative to t-test
3. **Cohen's d**: Effect size measure (negligible: <0.2, small: 0.2-0.5, medium: 0.5-0.8, large: >0.8)
4. **Bootstrap confidence intervals**: 10,000 resamples, 95% CI by default

**Example Output**:
```python
{
  "t_test": {
    "statistic": 2.456,
    "p_value": 0.0234,
    "significant": True
  },
  "wilcoxon": {
    "statistic": 34.0,
    "p_value": 0.0312,
    "significant": True
  },
  "effect_size": {
    "cohens_d": 0.68,
    "interpretation": "medium"
  },
  "bootstrap_ci": {
    "mean": 0.0423,
    "lower": 0.0089,
    "upper": 0.0751,
    "ci_level": 0.95
  },
  "summary": {
    "exp1_mean": 0.7621,
    "exp1_std": 0.0423,
    "exp2_mean": 0.7198,
    "exp2_std": 0.0512,
    "difference": 0.0423,
    "num_tasks": 8
  }
}
```

#### `generate_latex_table()` (line 419)

Generate LaTeX tables for thesis:

```python
latex_table = analyzer.generate_latex_table(
    experiment_names=['Baseline', 'Curriculum', 'Enhanced'],
    eval_type='indicglue',
    metric='accuracy',
    caption='IndicGLUE Benchmark Results',
    label='tab:indicglue',
    save_path='tables/indicglue_results.tex'
)

print(latex_table)
```

**Example Output**:
```latex
\begin{table}[htbp]
\centering
\caption{IndicGLUE Benchmark Results}
\label{tab:indicglue}
\begin{tabular}{lccc}
\toprule
Task & Baseline & Curriculum & Enhanced \\
\midrule
headlines_classification & 0.712 & 0.734 & \textbf{0.758} \\
bbc_hindi & 0.689 & 0.701 & \textbf{0.723} \\
movie_reviews & 0.745 & \textbf{0.768} & 0.762 \\
product_reviews & 0.721 & 0.739 & \textbf{0.751} \\
\midrule
Average & 0.717 & 0.736 & 0.749 \\
\bottomrule
\end{tabular}
\end{table}
```

**Features**:
- Best value in each row is **bolded**
- Average row computed automatically
- Uses professional `booktabs` package formatting
- Ready for direct inclusion in LaTeX thesis

#### `generate_summary_report()` (line 502)

Generate comprehensive markdown report:

```python
report = analyzer.generate_summary_report(
    experiment_name='curriculum_experiment',
    save_path='reports/curriculum_report.md'
)

print(report)
```

### Convenience Functions

#### `analyze_experiments()` (line 569)

Quick setup for analysis:

```python
from src.analysis.results_analyzer import analyze_experiments

# One-liner to load all experiments
analyzer = analyze_experiments('results')

# Start analyzing immediately
fig = analyzer.plot_training_curves(metrics=['train_loss', 'val_loss', 'perplexity'])
```

#### `quick_comparison()` (line 584)

Fast statistical comparison:

```python
from src.analysis.results_analyzer import quick_comparison

comparison = quick_comparison(
    exp1='baseline',
    exp2='curriculum',
    results_dir='results'
)

print(f"Significant difference: {comparison['t_test']['significant']}")
```

## 2. ThesisPlotter

**Location**: `src/analysis/visualization_utils.py:30`

**Purpose**: Create publication-quality visualizations with consistent thesis formatting

### Features

- **Consistent styling**: All plots follow thesis formatting guidelines
- **High resolution**: 300 DPI output for publication
- **Customizable themes**: 'thesis', 'presentation', 'paper' styles
- **Automatic layout**: Smart spacing and sizing
- **Export options**: PNG, PDF, SVG formats

### Initialization

```python
from src.analysis.visualization_utils import ThesisPlotter

# Initialize with thesis style
plotter = ThesisPlotter(style='thesis')

# Alternative styles
plotter_pres = ThesisPlotter(style='presentation')  # Larger fonts, bolder lines
plotter_paper = ThesisPlotter(style='paper')        # Compact for journals
```

### Plotting Methods

#### `plot_learning_rate_schedule()` (line 63)

```python
fig = plotter.plot_learning_rate_schedule(
    steps=list(range(0, 10000, 100)),
    lrs=[1e-5, 5e-5, 1e-4, 5e-4, 1e-4, 5e-5, 1e-5],
    title='Learning Rate Schedule: Linear Warmup + Cosine Decay',
    save_path='figures/lr_schedule.png'
)
```

**Features**:
- Logarithmic y-axis for LR visualization
- Grid for easy reading
- Professional styling

#### `plot_gradient_norms()` (line 95)

Visualize gradient norms during training:

```python
fig = plotter.plot_gradient_norms(
    epochs=list(range(1, 11)),
    grad_norms=[2.1, 1.8, 1.5, 1.2, 0.9, 0.8, 0.7, 0.7, 0.6, 0.6],
    title='Gradient Norms During Training',
    save_path='figures/gradient_norms.png'
)
```

**Features**:
- Marks clipping threshold (1.0) with dashed line
- Shows gradient behavior over training
- Helps diagnose training stability

#### `plot_multi_run_comparison()` (line 128)

Compare multiple runs with error bars:

```python
# Data structure: {experiment: {run_name: [values]}}
data = {
    'Baseline': {
        'Run 1': [0.71, 0.72, 0.73],
        'Run 2': [0.70, 0.71, 0.72],
        'Run 3': [0.72, 0.73, 0.74]
    },
    'Enhanced': {
        'Run 1': [0.75, 0.76, 0.77],
        'Run 2': [0.74, 0.75, 0.76],
        'Run 3': [0.76, 0.77, 0.78]
    }
}

fig = plotter.plot_multi_run_comparison(
    data,
    metric_name='Accuracy',
    title='Multi-Seed Run Comparison',
    save_path='figures/multi_run_comparison.png'
)
```

**Features**:
- Automatic mean and standard deviation calculation
- Error bars for uncertainty visualization
- Grouped bar charts for easy comparison

#### `plot_token_distribution()` (line 227)

Visualize token frequency distribution:

```python
token_counts = {
    'और': 15234,
    'में': 12456,
    'के': 11234,
    'है': 10987,
    # ... more tokens
}

fig = plotter.plot_token_distribution(
    token_counts,
    top_n=20,
    title='Top 20 Most Frequent Tokens',
    save_path='figures/token_distribution.png'
)
```

**Features**:
- Horizontal bar chart (easier to read token text)
- Frequency counts displayed on bars
- Colorful gradient for visual appeal

#### `plot_confusion_matrix()` (line 271)

Plot confusion matrices with normalization:

```python
import numpy as np

cm = np.array([
    [85, 10, 5],
    [8, 92, 0],
    [7, 3, 90]
])
class_names = ['Class A', 'Class B', 'Class C']

fig = plotter.plot_confusion_matrix(
    cm,
    class_names,
    title='Classification Results',
    normalize=True,
    save_path='figures/confusion_matrix.png'
)
```

**Features**:
- Optional normalization (row-wise)
- Color-coded cells with values
- Automatic threshold for text color (white on dark, black on light)

#### `create_figure_grid()` (line 370)

Create multi-panel figures:

```python
plots = [
    ('Training Loss', plot_func1, {'data': data1}),
    ('Validation Accuracy', plot_func2, {'data': data2}),
    ('Confusion Matrix', plot_func3, {'data': data3}),
    ('Token Distribution', plot_func4, {'data': data4})
]

fig = plotter.create_figure_grid(
    plots,
    ncols=2,
    figsize=(14, 10),
    title='Experiment Analysis Summary',
    save_path='figures/analysis_grid.png'
)
```

**Features**:
- Automatic grid layout
- Consistent subplot styling
- Overall title and individual subplot titles

### Style Customization

```python
# Thesis style (default)
plotter = ThesisPlotter(style='thesis')
# Font size: 11pt
# Figure size: (8, 5)
# DPI: 300

# Presentation style
plotter = ThesisPlotter(style='presentation')
# Font size: 14pt
# Figure size: (12, 7)
# DPI: 300

# Paper style (for journals)
plotter = ThesisPlotter(style='paper')
# Font size: 9pt
# Figure size: (6, 4)
# DPI: 300
```

### Export Formats

```python
# PNG (default)
plotter.plot_learning_rate_schedule(..., save_path='figure.png')

# PDF (vector, recommended for LaTeX)
plotter.plot_learning_rate_schedule(..., save_path='figure.pdf')

# SVG (editable vector format)
plotter.plot_learning_rate_schedule(..., save_path='figure.svg')
```

### Convenience Function

#### `quick_training_plot()` (line 412)

Quick training visualization:

```python
from src.analysis.visualization_utils import quick_training_plot

fig = quick_training_plot(
    train_losses=[2.5, 2.3, 2.1, 2.0, 1.9],
    val_losses=[2.6, 2.4, 2.2, 2.1, 2.0],
    save_path='figures/training.png'
)
```

## 3. ComparativeAnalyzer

**Location**: `src/evaluation/comparative_analysis.py:27`

**Purpose**: Advanced multi-model comparison with interactive visualizations and reports

### Features

- **Cross-model comparison**: Compare multiple experiments systematically
- **Comparison tables**: Side-by-side performance metrics
- **Radar plots**: Multi-dimensional task performance visualization
- **Regression analysis**: Training progression across checkpoints
- **Report generation**: HTML and PDF reports with Bootstrap styling

### Initialization

```python
from src.evaluation.comparative_analysis import ComparativeAnalyzer, compare_results

# Initialize analyzer
analyzer = ComparativeAnalyzer(
    results_dir='results',
    output_dir='comparative_analysis'
)

# Load specific experiments
result_paths = {
    'Baseline': 'results/baseline_experiment',
    'Curriculum': 'results/curriculum_experiment',
    'Enhanced': 'results/enhanced_experiment'
}
analyzer.load_results(result_paths)
```

### Core Methods

#### `load_results()` (line 105)

Load multiple evaluation results:

```python
result_paths = {
    'GPT-10M': 'results/gpt_10M_baseline',
    'GPT-100M': 'results/gpt_100M_large',
    'DeBERTa-10M': 'results/deberta_10M_baseline'
}

analyzer.load_results(result_paths)
print(f"Loaded {len(analyzer.loaded_results)} experiments")
```

#### `create_comparison_table()` (line 123)

Create side-by-side comparison table:

```python
comparison_df = analyzer.create_comparison_table(
    result_names=['GPT-10M', 'GPT-100M', 'DeBERTa-10M'],
    tasks=None,  # All tasks
    metrics=['accuracy', 'f1_macro']
)

print(comparison_df)
# Outputs pivoted DataFrame with tasks as rows and model_metric as columns
```

**Output Structure**:
```
Task                          GPT-10M_accuracy  GPT-100M_accuracy  DeBERTa-10M_accuracy
BBCArticlesClassification             0.7123            0.7456                 0.7234
MovieReviewSentiment                  0.6834            0.7123                 0.7012
...
```

#### `create_radar_plot()` (line 204)

Create multi-dimensional radar plot:

```python
fig = analyzer.create_radar_plot(
    result_names=['GPT-10M', 'GPT-100M', 'DeBERTa-10M'],
    tasks=None,  # All tasks
    metric='accuracy',
    save_path='comparative_analysis/radar_plot.html'
)

# Interactive HTML plot with plotly
```

**Requirements**: Requires `plotly` package
**Output**: Interactive HTML file with radar/spider plot

#### `regression_analysis()` (line 296)

Analyze metric progression across checkpoints:

```python
# Dictionary mapping checkpoint steps to results
checkpoint_results = {
    1000: results_1000,
    2000: results_2000,
    3000: results_3000,
    4000: results_4000,
    5000: results_5000
}

analysis = analyzer.regression_analysis(
    checkpoint_results,
    task='BBCArticlesClassification',
    metric='accuracy'
)

print(f"Slope: {analysis['slope']:.6f}")
print(f"R²: {analysis['r_squared']:.4f}")
print(f"Best checkpoint: {analysis['best_checkpoint']} ({analysis['best_value']:.4f})")
print(f"Improvement: {analysis['improvement']:.4f}")
```

**Output**:
```python
{
  'task': 'BBCArticlesClassification',
  'metric': 'accuracy',
  'num_checkpoints': 5,
  'slope': 0.000032,
  'intercept': 0.6834,
  'r_squared': 0.9234,
  'best_checkpoint': 5000,
  'best_value': 0.7456,
  'initial_value': 0.6834,
  'final_value': 0.7456,
  'improvement': 0.0622,
  'checkpoints': [1000, 2000, 3000, 4000, 5000],
  'values': [0.6834, 0.7012, 0.7234, 0.7345, 0.7456]
}
```

#### `generate_html_report()` (line 382)

Generate interactive HTML report:

```python
report_path = analyzer.generate_html_report(
    result_names=['GPT-10M', 'GPT-100M', 'DeBERTa-10M'],
    include_radar=True,
    include_comparison=True
)

print(f"HTML report saved to: {report_path}")
```

**Features**:
- Bootstrap 5 styling
- Responsive design
- Embedded comparison tables
- Interactive radar plots (iframe)
- Professional appearance
- Timestamp and metadata

#### `generate_pdf_report()` (line 513)

Generate publication-ready PDF report:

```python
report_path = analyzer.generate_pdf_report(
    result_names=['GPT-10M', 'GPT-100M', 'DeBERTa-10M']
)

print(f"PDF report saved to: {report_path}")
```

**Requirements**: Requires `matplotlib`
**Features**:
- Multi-page PDF with title page
- Comparison tables formatted for print
- Professional styling
- Suitable for thesis appendix

### Convenience Function

#### `compare_results()` (line 595)

Quick comparison workflow:

```python
from src.evaluation.comparative_analysis import compare_results

# One-liner for complete comparison with reports
analyzer = compare_results(
    result_paths={
        'Baseline': 'results/baseline',
        'Enhanced': 'results/enhanced'
    },
    output_dir='comparative_analysis',
    generate_html=True,
    generate_pdf=True
)

# Reports automatically generated in output_dir
```

## 4. Complete Analysis Workflow

### Step 1: Load and Analyze Results

```python
from src.analysis.results_analyzer import analyze_experiments
from src.analysis.visualization_utils import ThesisPlotter

# Load all experiments
analyzer = analyze_experiments('results')

# Check what was loaded
print(f"Loaded experiments: {list(analyzer.experiments.keys())}")
```

### Step 2: Training Analysis

```python
# Plot training curves
fig = analyzer.plot_training_curves(
    experiment_names=['baseline', 'curriculum', 'enhanced'],
    metrics=['train_loss', 'val_loss', 'perplexity'],
    save_path='figures/training_comparison.png'
)
```

### Step 3: Evaluation Comparison

```python
# IndicGLUE comparison
fig = analyzer.plot_evaluation_comparison(
    eval_type='indicglue',
    save_path='figures/indicglue_comparison.png'
)

# MultiBLiMP comparison
fig = analyzer.plot_evaluation_comparison(
    eval_type='multiblimp',
    save_path='figures/multiblimp_comparison.png'
)
```

### Step 4: Statistical Testing

```python
# Compare two best models
comparison = analyzer.compare_models_statistically(
    exp1='curriculum',
    exp2='enhanced',
    metric='accuracy',
    eval_type='indicglue'
)

print("📊 Statistical Comparison:")
print(f"  Curriculum: {comparison['summary']['exp1_mean']:.4f} ± {comparison['summary']['exp1_std']:.4f}")
print(f"  Enhanced: {comparison['summary']['exp2_mean']:.4f} ± {comparison['summary']['exp2_std']:.4f}")
print(f"  Difference: {comparison['summary']['difference']:.4f}")
print(f"  p-value: {comparison['t_test']['p_value']:.4f}")
print(f"  Significant: {comparison['t_test']['significant']}")
print(f"  Effect size: {comparison['effect_size']['interpretation']}")
```

### Step 5: Generate LaTeX Tables

```python
# IndicGLUE table
latex = analyzer.generate_latex_table(
    eval_type='indicglue',
    caption='IndicGLUE Benchmark Results',
    label='tab:indicglue',
    save_path='tables/indicglue_results.tex'
)

# MultiBLiMP table
latex = analyzer.generate_latex_table(
    eval_type='multiblimp',
    caption='MultiBLiMP Syntactic Phenomena Results',
    label='tab:multiblimp',
    save_path='tables/multiblimp_results.tex'
)
```

### Step 6: Generate Reports

```python
# Generate report for each experiment
for exp_name in analyzer.experiments.keys():
    report = analyzer.generate_summary_report(
        exp_name,
        save_path=f'reports/{exp_name}_report.md'
    )
```

### Step 7: Advanced Comparative Analysis

```python
from src.evaluation.comparative_analysis import compare_results

# Generate comprehensive comparison with HTML/PDF reports
analyzer = compare_results(
    result_paths={
        'Baseline': 'results/baseline',
        'Curriculum': 'results/curriculum',
        'Enhanced': 'results/enhanced'
    },
    output_dir='comparative_analysis',
    generate_html=True,
    generate_pdf=False  # Requires matplotlib
)
```

## Output Directory Structure

After running a complete analysis workflow:

```
results/
├── figures/
│   ├── training_comparison.png
│   ├── indicglue_comparison.png
│   ├── multiblimp_comparison.png
│   ├── lr_schedule.png
│   ├── gradient_norms.png
│   └── ...
├── tables/
│   ├── indicglue_results.tex
│   ├── multiblimp_results.tex
│   └── experiment_comparison.csv
├── reports/
│   ├── baseline_report.md
│   ├── curriculum_report.md
│   └── enhanced_report.md
└── comparative_analysis/
    ├── comparative_report_*.html
    ├── comparative_report_*.pdf
    └── radar_plot.html
```

## Best Practices

### Statistical Testing

1. **Always use paired tests**: Experiments are evaluated on same tasks
2. **Report multiple tests**: Include both parametric (t-test) and non-parametric (Wilcoxon)
3. **Calculate effect size**: p-value alone doesn't indicate practical significance
4. **Use bootstrap CIs**: Provides robust uncertainty estimates

### Visualization

1. **Consistent styling**: Use ThesisPlotter for all figures
2. **High resolution**: Save at 300+ DPI for print quality
3. **Vector formats**: Use PDF for LaTeX inclusion when possible
4. **Clear labels**: Include units, legend, and descriptive titles

### LaTeX Integration

1. **Use `\input{}`**: Include generated .tex files directly in thesis
2. **Consistent labels**: Follow naming convention (tab:category_metric)
3. **Update captions**: Customize captions for thesis context
4. **Check formatting**: Ensure tables compile correctly in thesis template

### Reproducibility

1. **Save configurations**: Include experiment configs in reports
2. **Document versions**: Record git commits in metadata
3. **Archive results**: Keep raw results separate from processed outputs
4. **Version figures**: Include date or experiment ID in filenames

## Related Documentation

- [Training Pipeline Documentation](05_TRAINING.md) - Generates the results being analyzed
- [Evaluation Framework Documentation](06_EVALUATION.md) - Produces evaluation metrics
- [Jupyter Notebooks Documentation](08c_JUPYTER_NOTEBOOKS.md) - Interactive analysis workflows
- [Configuration Reference](07_CONFIGURATION.md) - Experiment configuration options
