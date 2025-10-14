# Analysis and Visualization Tools

## Overview

Phase 2 of the Hindi BabyLM project provides comprehensive analysis and visualization tools for processing experimental results, generating thesis-ready figures, and performing statistical comparisons. These tools bridge the gap between raw experimental results and publication-quality outputs.

**Key Components**:
1. **ResultsAnalyzer**: Statistical analysis, model comparison, LaTeX table generation
2. **ThesisPlotter**: Publication-ready visualizations with thesis formatting
3. **Jupyter Notebooks**: Interactive data exploration and results analysis

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
    Thesis Outputs
    • Figures (PNG/PDF)
    • LaTeX tables (.tex)
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

#### `plot_training_curves()` (line 159)

Visualize training progression:

```python
# Plot specific metrics for specific experiments
fig = analyzer.plot_training_curves(
    experiment_names=['baseline', 'curriculum'],
    metrics=['loss', 'perplexity'],
    save_path='figures/training_curves.png'
)

# Plot all experiments, all metrics
fig = analyzer.plot_training_curves()
```

**Output**: Multi-panel figure with one subplot per metric, comparing all specified experiments.

#### `plot_evaluation_comparison()` (line 207)

Compare evaluation results across experiments:

```python
# IndicGLUE comparison
fig = analyzer.plot_evaluation_comparison(
    experiment_names=['baseline', 'curriculum', 'enhanced'],
    eval_type='indicglue',
    save_path='figures/indicglue_comparison.png'
)

# MultiBLiMP comparison (14 phenomena)
fig = analyzer.plot_evaluation_comparison(
    eval_type='multiblimp',
    save_path='figures/multiblimp_comparison.png'
)

# Morphological probes comparison (10 tasks)
fig = analyzer.plot_evaluation_comparison(
    eval_type='probes',
    save_path='figures/probes_comparison.png'
)
```

**Output**: Horizontal bar chart with tasks/phenomena on y-axis, accuracy on x-axis, one bar per experiment.

#### `compare_models_statistically()` (line 276)

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
3. **Cohen's d**: Effect size measure (small: 0.2-0.5, medium: 0.5-0.8, large: >0.8)
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

#### `generate_latex_table()` (line 397)

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
soham_ner & 0.654 & 0.672 & \textbf{0.689} \\
wikiann_ner & 0.678 & 0.691 & \textbf{0.704} \\
\midrule
Average & 0.700 & 0.718 & 0.731 \\
\bottomrule
\end{tabular}
\end{table}
```

**Features**:
- Best value in each row is **bolded**
- Average row computed automatically
- Uses professional `booktabs` package formatting
- Ready for direct inclusion in LaTeX thesis

#### `generate_summary_report()` (line 480)

Generate comprehensive markdown report:

```python
report = analyzer.generate_summary_report(
    experiment_name='curriculum_experiment',
    save_path='reports/curriculum_report.md'
)

print(report)
```

**Example Output**:
```markdown
# Experiment Report: curriculum_experiment

## Metadata

- **Timestamp**: 2025-01-15T14:32:11
- **Git Commit**: abc123def
- **Device**: cuda:0

## Training Summary

- **Epochs**: 50
- **Final Loss**: 2.3456
- **Best Val Loss**: 2.2891
- **Training Time**: 18234.56s

## Evaluation Results

### INDICGLUE

- **headlines_classification**: 0.7342
- **bbc_hindi**: 0.7012
- **movie_reviews**: 0.7681
...

### MULTIBLIMP

- **subject_verb_agreement_number**: 0.8800
- **case_marking_ergative**: 0.7600
...

### MORPHOLOGICAL_PROBES

- **case_detection**: 0.8400 (best layer: 8)
- **number_detection**: 0.9100 (best layer: 6)
...
```

### Convenience Functions

#### `analyze_experiments()` (line 540)

Quick setup for analysis:

```python
from src.analysis.results_analyzer import analyze_experiments

# One-liner to load all experiments
analyzer = analyze_experiments('results')

# Start analyzing immediately
fig = analyzer.plot_training_curves(metrics=['loss', 'perplexity'])
```

#### `quick_comparison()` (line 555)

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

**Location**: `src/analysis/visualization_utils.py:25`

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
plotter_paper = ThesisPlotter(style='paper')        # Nature/Science formatting
```

### Plotting Methods

#### `plot_training_curves()` (line 85)

```python
# Plot training and validation curves
fig = plotter.plot_training_curves(
    train_losses=[2.8, 2.5, 2.3, 2.1, 2.0],
    val_losses=[2.9, 2.6, 2.4, 2.2, 2.1],
    epochs=list(range(1, 6)),
    title='Training Convergence',
    save_path='figures/convergence.png'
)
```

#### `plot_evaluation_heatmap()` (line 132)

```python
# Heatmap of evaluation results
import pandas as pd

results = pd.DataFrame({
    'Task 1': [0.72, 0.75, 0.78],
    'Task 2': [0.68, 0.71, 0.74],
    'Task 3': [0.81, 0.83, 0.85]
}, index=['Baseline', 'Curriculum', 'Enhanced'])

fig = plotter.plot_evaluation_heatmap(
    results,
    title='Evaluation Results Heatmap',
    save_path='figures/eval_heatmap.png'
)
```

#### `plot_layer_wise_probe_results()` (line 189)

Visualize morphological probe results across layers:

```python
# Layer-wise probe accuracies
layer_results = {
    0: 0.48, 1: 0.57, 2: 0.65, 3: 0.72,
    4: 0.77, 5: 0.81, 6: 0.83, 7: 0.84,
    8: 0.85, 9: 0.84, 10: 0.82, 11: 0.79, 12: 0.76
}

fig = plotter.plot_layer_wise_probe_results(
    layer_results,
    probe_name='Case Detection',
    title='Layer-wise Case Detection Accuracy',
    save_path='figures/case_probe_layers.png'
)
```

**Features**:
- Highlights best-performing layer
- Shows layer-by-layer progression
- Annotates peak performance

#### `plot_multiple_probes_comparison()` (line 245)

Compare multiple probes on the same plot:

```python
probes_data = {
    'Case Detection': layer_results_case,
    'Number Detection': layer_results_number,
    'Gender Detection': layer_results_gender
}

fig = plotter.plot_multiple_probes_comparison(
    probes_data,
    title='Morphological Probe Comparison Across Layers',
    save_path='figures/probes_comparison.png'
)
```

#### `plot_phenomenon_breakdown()` (line 302)

Detailed breakdown of MultiBLiMP phenomena:

```python
phenomena_accuracies = {
    'Subject-Verb Agr. (Num)': 0.88,
    'Subject-Verb Agr. (Gender)': 0.82,
    'Case Marking (Erg)': 0.76,
    'Case Marking (Acc)': 0.72,
    'Word Order': 0.68,
    'Honorific Agreement': 0.79,
    # ... all 14 phenomena
}

fig = plotter.plot_phenomenon_breakdown(
    phenomena_accuracies,
    title='MultiBLiMP Syntactic Phenomena Results',
    save_path='figures/phenomena_breakdown.png'
)
```

#### `plot_learning_rate_schedule()` (line 358)

Visualize LR scheduling:

```python
fig = plotter.plot_learning_rate_schedule(
    learning_rates=[1e-5, 5e-5, 1e-4, 5e-4, 1e-4, 5e-5, 1e-5],
    epochs=list(range(1, 8)),
    schedule_name='Linear Warmup + Cosine Decay',
    save_path='figures/lr_schedule.png'
)
```

#### `plot_curriculum_difficulty()` (line 412)

Visualize curriculum learning progression:

```python
fig = plotter.plot_curriculum_difficulty(
    epochs=list(range(1, 11)),
    difficulty_scores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    num_samples=[1000, 2000, 3500, 5000, 6500, 8000, 9200, 10000, 10000, 10000],
    title='Curriculum Learning Progression',
    save_path='figures/curriculum_progression.png'
)
```

#### `plot_performance_vs_model_size()` (line 468)

Compare performance across model sizes:

```python
model_sizes = [50, 110, 350]  # Million parameters
accuracies = [0.72, 0.78, 0.82]
model_names = ['Tiny', 'Small', 'Medium']

fig = plotter.plot_performance_vs_model_size(
    model_sizes,
    accuracies,
    model_names,
    title='Model Size vs IndicGLUE Performance',
    xlabel='Parameters (Millions)',
    ylabel='IndicGLUE Accuracy',
    save_path='figures/model_size_comparison.png'
)
```

### Style Customization

```python
# Thesis style (default)
plotter = ThesisPlotter(style='thesis')
# Font size: 10-12pt
# Figure size: (10, 6)
# Line width: 2
# DPI: 300

# Presentation style
plotter = ThesisPlotter(style='presentation')
# Font size: 14-16pt
# Figure size: (12, 7)
# Line width: 3
# DPI: 150

# Paper style (Nature/Science)
plotter = ThesisPlotter(style='paper')
# Font size: 8-10pt
# Figure size: (8, 5)
# Line width: 1.5
# DPI: 600
```

### Export Formats

```python
# PNG (default)
plotter.plot_training_curves(..., save_path='figure.png')

# PDF (vector, recommended for LaTeX)
plotter.plot_training_curves(..., save_path='figure.pdf')

# SVG (editable vector format)
plotter.plot_training_curves(..., save_path='figure.svg')

# High-resolution PNG
plotter.set_dpi(600)
plotter.plot_training_curves(..., save_path='figure_hires.png')
```

## Complete Analysis Workflow

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
    metrics=['loss', 'perplexity'],
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

# Morphological probes comparison
fig = analyzer.plot_evaluation_comparison(
    eval_type='probes',
    save_path='figures/probes_comparison.png'
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

# Morphological probes table
latex = analyzer.generate_latex_table(
    eval_type='probes',
    caption='Morphological Probe Results',
    label='tab:probes',
    save_path='tables/probes_results.tex'
)
```

### Step 6: Detailed Visualizations

```python
# Initialize plotter
plotter = ThesisPlotter(style='thesis')

# Layer-wise probe visualization for each probe
probe_tasks = ['case_detection', 'number_detection', 'gender_detection']

for probe in probe_tasks:
    # Extract layer-wise results from analyzer
    layer_results = {}  # ... extract from experiment results

    fig = plotter.plot_layer_wise_probe_results(
        layer_results,
        probe_name=probe.replace('_', ' ').title(),
        save_path=f'figures/probe_{probe}_layers.png'
    )
```

### Step 7: Generate Reports

```python
# Generate report for each experiment
for exp_name in analyzer.experiments.keys():
    report = analyzer.generate_summary_report(
        exp_name,
        save_path=f'reports/{exp_name}_report.md'
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
│   ├── probes_comparison.png
│   ├── probe_case_detection_layers.png
│   ├── probe_number_detection_layers.png
│   └── ...
├── tables/
│   ├── indicglue_results.tex
│   ├── multiblimp_results.tex
│   └── probes_results.tex
└── reports/
    ├── baseline_report.md
    ├── curriculum_report.md
    └── enhanced_report.md
```

## Integration with Jupyter Notebooks

The analysis tools are designed to work seamlessly in Jupyter notebooks:

```python
# In notebooks/02_results_analysis.ipynb

import sys
sys.path.append('..')

from src.analysis.results_analyzer import analyze_experiments
from src.analysis.visualization_utils import ThesisPlotter

# Load experiments
analyzer = analyze_experiments('../results')

# Interactive plotting
%matplotlib inline
fig = analyzer.plot_training_curves(metrics=['loss', 'perplexity'])
plt.show()

# Generate all thesis outputs
analyzer.generate_latex_table(eval_type='indicglue',
                             save_path='../tables/indicglue.tex')
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
- [Jupyter Notebooks Documentation](09_JUPYTER_NOTEBOOKS.md) - Interactive analysis workflows
- [Thesis Integration Guide](10_THESIS_INTEGRATION.md) - Incorporating outputs into thesis
