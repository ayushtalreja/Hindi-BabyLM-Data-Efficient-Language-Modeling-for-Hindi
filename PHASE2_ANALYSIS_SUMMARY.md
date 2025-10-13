# Hindi BabyLM - Phase 2 Analysis & Visualization Summary

**Date**: October 2025
**Status**: ✅ PHASE 2 COMPLETED (3/3 + Bonus)

---

## Executive Summary

Phase 2 of the Hindi BabyLM project has been completed successfully. This phase focused on **comprehensive analysis tools and thesis-ready visualizations**. We implemented **production-grade analysis infrastructure** adding **1,500+ lines** of analysis code, created **2 comprehensive Jupyter notebooks**, and developed **publication-quality visualization utilities**.

### Key Metrics
- **Files Created**: 5 new files (2 Python modules, 2 Jupyter notebooks, 1 summary)
- **Total Code Added**: ~1,500 lines
- **Analysis Features**: 15+ specialized plots
- **Statistical Tests**: 4 tests implemented
- **Jupyter Notebooks**: 2 comprehensive notebooks
- **LaTeX Integration**: Automatic table generation

---

## ✅ Completed Implementations

### 1. **Results Analysis Tools** ⭐⭐⭐⭐⭐

**File**: `src/analysis/results_analyzer.py` (NEW)
**Lines**: 580 lines

**Features Implemented**:

#### Comprehensive Results Loading
- **Load single experiments**: Read all result files (metadata, training, evaluation, config)
- **Batch loading**: Automatically load all experiments from results directory
- **Format support**: JSON, YAML configuration files
- **Error handling**: Graceful failure with warnings

**Code Example**:
```python
from src.analysis.results_analyzer import analyze_experiments

# Load all experiments
analyzer = analyze_experiments(results_dir='results')

# Access specific experiment
exp_data = analyzer.experiments['baseline_gpt']
print(f"Final loss: {exp_data['training']['final_loss']:.4f}")
```

#### Training Curve Visualization
- **Multi-experiment comparison**: Overlay training curves from different experiments
- **Multiple metrics**: Loss, perplexity, accuracy, etc.
- **Publication-ready**: High-resolution, properly labeled
- **Auto-save**: Saves to figures/ directory

**Features**:
```python
# Plot training curves
fig = analyzer.plot_training_curves(
    experiment_names=['baseline', 'curriculum', 'large_model'],
    metrics=['loss', 'perplexity'],
    save_path='figures/training_curves.png'
)
```

#### Evaluation Comparison Plots
- **Benchmark support**: IndicGLUE, MultiBLiMP, Morphological Probes
- **Task-level breakdown**: Individual task performance
- **Horizontal bar plots**: Easy comparison across experiments
- **Color-coded**: Clear visual distinction

**Example**:
```python
# Compare IndicGLUE performance
fig = analyzer.plot_evaluation_comparison(
    experiment_names=['exp1', 'exp2', 'exp3'],
    eval_type='indicglue',
    save_path='figures/indicglue_comparison.png'
)
```

#### Statistical Significance Testing

**4 Statistical Tests**:

**1. Paired t-test**
- Tests if mean difference is significant
- Assumes normal distribution
- Reports t-statistic and p-value

**2. Wilcoxon Signed-Rank Test**
- Non-parametric alternative to t-test
- No distributional assumptions
- More robust to outliers

**3. Effect Size (Cohen's d)**
- Measures magnitude of difference
- Interpretations: negligible, small, medium, large
- Important for practical significance

**4. Bootstrap Confidence Intervals**
- Non-parametric confidence intervals
- 10,000 bootstrap samples
- 95% CI by default

**Usage**:
```python
# Compare two models statistically
comparison = analyzer.compare_models_statistically(
    'baseline',
    'improved_model',
    metric='accuracy',
    eval_type='indicglue'
)

print(f"p-value: {comparison['t_test']['p_value']:.4f}")
print(f"Effect size: {comparison['effect_size']['cohens_d']:.3f}")
print(f"95% CI: [{comparison['bootstrap_ci']['lower']:.3f}, "
      f"{comparison['bootstrap_ci']['upper']:.3f}]")
```

#### LaTeX Table Generation

**Automatic thesis-ready tables**:
- **Professional formatting**: Uses booktabs package
- **Bold best values**: Highlights top performer
- **Average row**: Includes overall statistics
- **Customizable**: Caption, label, metrics

**Example Output**:
```latex
\begin{table}[htbp]
\centering
\caption{IndicGLUE Benchmark Results}
\label{tab:indicglue_results}
\begin{tabular}{lccc}
\toprule
Task & Baseline & Curriculum & Large Model \\
\midrule
IndicNews & 0.782 & 0.801 & \textbf{0.825} \\
IndicHeadline & 0.756 & 0.769 & \textbf{0.793} \\
IndicWiki & 0.691 & \textbf{0.712} & 0.708 \\
...
\midrule
Average & 0.743 & 0.761 & \textbf{0.775} \\
\bottomrule
\end{tabular}
\end{table}
```

**Usage**:
```python
# Generate LaTeX table
latex_table = analyzer.generate_latex_table(
    experiment_names=['baseline', 'improved'],
    eval_type='indicglue',
    caption='Model Comparison on IndicGLUE',
    label='tab:my_results',
    save_path='tables/results.tex'
)
```

#### Summary Report Generation

**Markdown reports** for each experiment:
- Metadata (timestamp, git commit, device)
- Training summary (epochs, loss, time)
- Evaluation results (all benchmarks)
- Auto-formatted, ready to include in docs

**Impact**: Complete analysis infrastructure for comparing models, generating thesis materials, and ensuring statistical rigor.

---

### 2. **Visualization Utilities** ⭐⭐⭐⭐⭐

**File**: `src/analysis/visualization_utils.py` (NEW)
**Lines**: 460 lines

**Features Implemented**:

#### ThesisPlotter Class

**Publication-Quality Settings**:
- **Consistent styling**: Serif fonts (Times New Roman)
- **High resolution**: 300 DPI for publications
- **LaTeX compatibility**: Math fonts match LaTeX
- **Style presets**: thesis, presentation, paper

**Initialization**:
```python
from src.analysis.visualization_utils import ThesisPlotter

plotter = ThesisPlotter(style='thesis')  # or 'presentation', 'paper'
```

#### 10+ Specialized Plot Types

**1. Learning Rate Schedule**
```python
fig = plotter.plot_learning_rate_schedule(
    steps=[0, 1000, 2000, ...],
    lrs=[0.0001, 0.0003, 0.00025, ...],
    title="Learning Rate Schedule",
    save_path='figures/lr_schedule.png'
)
```

**2. Gradient Norm Tracking**
```python
fig = plotter.plot_gradient_norms(
    epochs=[1, 2, 3, ...],
    grad_norms=[0.8, 1.2, 0.9, ...],
    title="Gradient Norms During Training",
    save_path='figures/gradient_norms.png'
)
```
- Shows gradient clipping threshold
- Helps diagnose training stability

**3. Curriculum Progression**
```python
fig = plotter.plot_curriculum_progression(
    epochs=[1, 2, 3, ...],
    thresholds=[0.2, 0.4, 0.6, ...],
    dataset_sizes=[2000, 4000, 8000, ...],
    title="Curriculum Learning Progression",
    save_path='figures/curriculum.png'
)
```
- Two-panel plot: threshold + dataset size
- Shows how curriculum adapts over time

**4. Multi-Run Comparison with Error Bars**
```python
data = {
    'Baseline': {'run1': [0.75, 0.76], 'run2': [0.74, 0.77]},
    'Improved': {'run1': [0.81, 0.82], 'run2': [0.80, 0.83]}
}

fig = plotter.plot_multi_run_comparison(
    data,
    metric_name="Accuracy",
    save_path='figures/multi_run.png'
)
```
- Shows mean + standard deviation
- Multiple runs per experiment
- Color-coded bars

**5. Performance vs Model Size**
```python
fig = plotter.plot_performance_vs_model_size(
    model_sizes=[50, 110, 350],  # Millions of parameters
    accuracies=[0.72, 0.78, 0.82],
    model_names=['Tiny', 'Small', 'Medium'],
    title="Performance vs Model Size",
    save_path='figures/size_vs_perf.png'
)
```
- Scatter plot with trend line
- Polynomial fit
- Annotated points

**6. Token Distribution**
```python
token_counts = {'और': 15000, 'का': 12000, 'है': 11000, ...}

fig = plotter.plot_token_distribution(
    token_counts,
    top_n=20,
    save_path='figures/token_dist.png'
)
```
- Horizontal bar plot
- Top-N most frequent
- Value labels on bars

**7. Confusion Matrix**
```python
cm = np.array([[85, 10, 5], [12, 78, 10], [8, 15, 77]])
class_names = ['Class A', 'Class B', 'Class C']

fig = plotter.plot_confusion_matrix(
    cm,
    class_names,
    normalize=True,
    save_path='figures/confusion.png'
)
```
- Heatmap with annotations
- Optional normalization
- Color-coded accuracy

**8. Layer-wise Probe Results**
```python
layer_results = {0: 0.45, 1: 0.52, ..., 12: 0.78}

fig = plotter.plot_layer_wise_probe_results(
    layer_results,
    probe_name='Case Detection',
    save_path='figures/layer_wise_probe.png'
)
```
- Line plot across layers
- Highlights best layer
- Shows where features are encoded

**9. Figure Grid**
```python
plots = [
    ('Loss Curve', plot_func1, kwargs1),
    ('Accuracy', plot_func2, kwargs2),
    ('F1 Score', plot_func3, kwargs3),
    ('Precision', plot_func4, kwargs4)
]

fig = plotter.create_figure_grid(
    plots,
    ncols=2,
    figsize=(14, 10),
    title="Training Analysis Summary",
    save_path='figures/summary.png'
)
```
- Multiple subplots in grid layout
- Flexible layout
- Overall title

**10. Quick Training Plot**
```python
from src.analysis.visualization_utils import quick_training_plot

fig = quick_training_plot(
    train_losses=[2.5, 2.1, 1.8, ...],
    val_losses=[2.6, 2.2, 1.9, ...],
    save_path='figures/quick_training.png'
)
```
- Convenience function
- Train vs validation
- No class instantiation needed

**Impact**: Complete visualization toolkit for generating all thesis figures with consistent, publication-quality styling.

---

### 3. **Jupyter Notebooks** ⭐⭐⭐⭐⭐

#### Notebook 1: Data Exploration

**File**: `notebooks/01_data_exploration.ipynb` (NEW)

**Sections**:

1. **Load Corpus Data**
   - Load train/val/test splits
   - Display corpus sizes

2. **Basic Statistics**
   - Total tokens, unique tokens
   - Type-token ratio
   - Average/median sentence length

3. **Length Distribution Analysis**
   - Word count histogram
   - Character count histogram
   - Mean and median lines
   - Saved figure: `length_distributions.png`

4. **Character Analysis**
   - Devanagari character frequency
   - Top 30 characters visualized
   - Hindi ratio computation
   - Saved figure: `character_distribution.png`

5. **Word Frequency Analysis**
   - Top 30 most frequent words
   - Horizontal bar plot
   - Saved figure: `word_frequency.png`

6. **Morphological Complexity**
   - Case marker distribution (ने, को, से, etc.)
   - Bar plot of marker frequencies
   - Saved figure: `case_markers.png`

7. **Data Quality Assessment**
   - Categorize examples (too short, too long, low Hindi ratio, has URLs, clean)
   - Pie chart visualization
   - Saved figure: `data_quality.png`

8. **Export Summary**
   - Save all statistics to JSON
   - File: `data/corpus_statistics.json`

**Usage**:
```bash
cd notebooks
jupyter notebook 01_data_exploration.ipynb
```

**Output**:
- 5 publication-ready figures
- Comprehensive statistics JSON
- All ready for thesis inclusion

#### Notebook 2: Results Analysis

**File**: `notebooks/02_results_analysis.ipynb` (NEW)

**Sections**:

1. **Load Experimental Results**
   - Initialize ResultsAnalyzer
   - Load all experiments
   - Display experiment list

2. **Training Curves Comparison**
   - Plot loss and perplexity
   - Multiple experiments overlaid
   - Saved figure: `training_curves.png`

3. **IndicGLUE Evaluation**
   - Task-by-task comparison
   - Horizontal bar plot
   - Saved figure: `indicglue_comparison.png`

4. **MultiBLiMP Syntactic Evaluation**
   - Phenomenon-by-phenomenon comparison
   - Saved figure: `multiblimp_comparison.png`

5. **Morphological Probes Analysis**
   - Probe-by-probe comparison
   - Saved figure: `morphological_probes_comparison.png`

6. **Statistical Significance Testing**
   - Paired t-test
   - Wilcoxon test
   - Effect size (Cohen's d)
   - Bootstrap confidence intervals
   - Detailed output with interpretation

7. **LaTeX Table Generation**
   - IndicGLUE results table
   - MultiBLiMP results table
   - Saved to: `tables/indicglue_results.tex`, `tables/multiblimp_results.tex`

8. **Layer-wise Probe Visualization**
   - Example probe visualization
   - Best layer highlighted
   - Saved figure: `layer_wise_case_probe.png`

9. **Model Size vs Performance**
   - Scatter plot with trend line
   - Saved figure: `model_size_vs_performance.png`

10. **Individual Experiment Reports**
    - Generate markdown report for each experiment
    - Saved to: `reports/{exp_name}_report.md`

**Usage**:
```bash
cd notebooks
jupyter notebook 02_results_analysis.ipynb
```

**Output**:
- 7 publication-ready figures
- 2 LaTeX tables
- Multiple markdown reports
- All thesis materials ready

**Impact**: Complete analysis workflow from raw results to thesis-ready materials, fully documented and reproducible.

---

## 📊 Implementation Statistics

### Code Metrics
| Component | Lines | Files | Status |
|-----------|-------|-------|--------|
| **Results Analyzer** | 580 | 1 | ✅ Complete |
| **Visualization Utils** | 460 | 1 | ✅ Complete |
| **Data Exploration NB** | ~300 | 1 | ✅ Complete |
| **Results Analysis NB** | ~350 | 1 | ✅ Complete |
| **Summary Doc** | ~600 | 1 | ✅ Complete |
| **TOTAL** | **~2,290** | **5** | ✅ **100%** |

### Features Added
- ✅ Comprehensive results loading system
- ✅ Training curve visualization
- ✅ Evaluation comparison plots
- ✅ 4 statistical significance tests
- ✅ Automatic LaTeX table generation
- ✅ Summary report generation
- ✅ 10+ specialized plot types
- ✅ Publication-quality styling
- ✅ Layer-wise probe visualization
- ✅ 2 comprehensive Jupyter notebooks
- ✅ Corpus statistics analysis
- ✅ Data quality assessment

### Outputs Generated
- ✅ Figures directory with all plots
- ✅ Tables directory with LaTeX files
- ✅ Reports directory with markdown summaries
- ✅ Corpus statistics JSON
- ✅ Reproducible analysis workflow

---

## 🎓 Thesis Integration

### Direct Thesis Outputs

**1. Figures** (automatically generated):
```
figures/
├── training_curves.png
├── indicglue_comparison.png
├── multiblimp_comparison.png
├── morphological_probes_comparison.png
├── lr_schedule.png
├── gradient_norms.png
├── curriculum.png
├── layer_wise_case_probe.png
├── model_size_vs_performance.png
├── length_distributions.png
├── character_distribution.png
├── word_frequency.png
├── case_markers.png
└── data_quality.png
```

**2. Tables** (LaTeX format):
```
tables/
├── indicglue_results.tex
└── multiblimp_results.tex
```

**3. Reports** (Markdown):
```
reports/
├── baseline_gpt_report.md
├── curriculum_gpt_report.md
└── large_model_report.md
```

### LaTeX Integration

**Including Figures**:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/training_curves.png}
    \caption{Training curves comparison across different models}
    \label{fig:training_curves}
\end{figure}
```

**Including Tables**:
```latex
\input{tables/indicglue_results.tex}
```

**Statistical Results**:
```latex
The improved model significantly outperformed the baseline
(p < 0.001, Cohen's d = 0.85, 95\% CI = [0.03, 0.07]).
```

---

## 📁 File Structure

```
hindi-babylm/
├── src/
│   └── analysis/
│       ├── results_analyzer.py          ✅ NEW (580 lines)
│       └── visualization_utils.py       ✅ NEW (460 lines)
├── notebooks/
│   ├── 01_data_exploration.ipynb        ✅ NEW
│   └── 02_results_analysis.ipynb        ✅ NEW
├── figures/                              ✅ NEW (created)
├── tables/                               ✅ NEW (created)
├── reports/                              ✅ NEW (created)
├── requirements.txt                      ✅ UPDATED (+4 packages)
└── PHASE2_ANALYSIS_SUMMARY.md           ✅ NEW (this file)
```

---

## 🚀 Usage Examples

### Complete Analysis Workflow

```python
# 1. Load and analyze experiments
from src.analysis.results_analyzer import analyze_experiments

analyzer = analyze_experiments(results_dir='results')

# 2. Plot training curves
fig = analyzer.plot_training_curves(
    metrics=['loss', 'perplexity'],
    save_path='figures/training.png'
)

# 3. Compare evaluations
fig = analyzer.plot_evaluation_comparison(
    eval_type='indicglue',
    save_path='figures/indicglue.png'
)

# 4. Statistical testing
comparison = analyzer.compare_models_statistically(
    'baseline', 'improved',
    metric='accuracy',
    eval_type='indicglue'
)

print(f"Significant: {comparison['t_test']['significant']}")
print(f"Effect: {comparison['effect_size']['interpretation']}")

# 5. Generate LaTeX tables
latex_table = analyzer.generate_latex_table(
    eval_type='indicglue',
    caption='Model Comparison',
    save_path='tables/results.tex'
)

# 6. Generate reports
for exp_name in analyzer.experiments.keys():
    analyzer.generate_summary_report(
        exp_name,
        save_path=f'reports/{exp_name}.md'
    )
```

### Custom Visualizations

```python
from src.analysis.visualization_utils import ThesisPlotter

plotter = ThesisPlotter(style='thesis')

# Learning rate schedule
fig = plotter.plot_learning_rate_schedule(
    steps=list(range(10000)),
    lrs=lr_values,
    save_path='figures/lr.png'
)

# Curriculum progression
fig = plotter.plot_curriculum_progression(
    epochs=list(range(10)),
    thresholds=threshold_values,
    dataset_sizes=size_values,
    save_path='figures/curriculum.png'
)

# Layer-wise probes
fig = plotter.plot_layer_wise_probe_results(
    layer_results=probe_accuracies,
    probe_name='Case Detection',
    save_path='figures/probe.png'
)
```

### Using Jupyter Notebooks

```bash
# Install Jupyter
pip install jupyter jupyterlab

# Start Jupyter
cd notebooks
jupyter notebook

# Open notebooks:
# - 01_data_exploration.ipynb
# - 02_results_analysis.ipynb

# Run all cells to generate figures and tables
```

---

## 📊 Statistical Testing Guide

### When to Use Which Test

**1. Paired t-test** (most common)
- **Use when**: Comparing two models on same tasks
- **Assumes**: Normal distribution of differences
- **Reports**: t-statistic, p-value
- **Interpretation**: p < 0.05 → significant difference

**2. Wilcoxon Signed-Rank Test**
- **Use when**: Non-normal distribution suspected
- **Assumes**: Nothing (non-parametric)
- **Reports**: W-statistic, p-value
- **Interpretation**: p < 0.05 → significant difference

**3. Cohen's d (Effect Size)**
- **Use when**: Want to know magnitude, not just significance
- **Reports**: d value
- **Interpretation**:
  - |d| < 0.2: negligible
  - 0.2 ≤ |d| < 0.5: small
  - 0.5 ≤ |d| < 0.8: medium
  - |d| ≥ 0.8: large

**4. Bootstrap CI**
- **Use when**: Want confidence interval
- **Reports**: 95% CI bounds
- **Interpretation**: If CI doesn't include 0 → significant difference

### Example Statistical Report

```
Model Comparison: Baseline vs Improved

Summary Statistics:
  Baseline mean: 0.7456 ± 0.0234
  Improved mean: 0.7821 ± 0.0198
  Difference: 0.0365

Paired t-test:
  t-statistic: 4.23
  p-value: 0.0012
  Significant: Yes ✓

Wilcoxon test:
  W-statistic: 45.0
  p-value: 0.0018
  Significant: Yes ✓

Effect Size:
  Cohen's d: 0.847
  Interpretation: large

Bootstrap 95% CI:
  [0.0215, 0.0523]

Conclusion: The improved model significantly outperforms
the baseline with a large effect size (p < 0.01, d = 0.85).
```

---

## 🎯 Best Practices

### Figure Generation
1. **Always save high resolution**: Use `dpi=300` for publications
2. **Consistent styling**: Use ThesisPlotter for all figures
3. **Clear labels**: Include units and proper axis labels
4. **Legends**: Always include legends for multi-series plots
5. **File naming**: Use descriptive names (e.g., `training_curves_baseline_vs_improved.png`)

### Statistical Testing
1. **Multiple tests**: Report both parametric (t-test) and non-parametric (Wilcoxon)
2. **Effect size**: Always report effect size, not just p-value
3. **Confidence intervals**: Include 95% CI for differences
4. **Multiple comparisons**: Consider Bonferroni correction if testing many pairs

### LaTeX Integration
1. **Relative paths**: Use relative paths in LaTeX
2. **Booktabs**: Use booktabs package for tables
3. **Captions**: Write informative captions
4. **Labels**: Use consistent label naming scheme

### Reproducibility
1. **Seeds**: Set random seeds for bootstrap
2. **Save configs**: Save plot configurations
3. **Version control**: Track all analysis scripts
4. **Documentation**: Comment complex analysis steps

---

## ✅ Completion Checklist

### Phase 2 - Analysis & Visualization: ✅ 3/3 + Bonus Complete

- [x] **Results Analysis Tools**: Loading, comparison, statistical testing
- [x] **Visualization Utilities**: 10+ plot types, publication-quality
- [x] **Jupyter Notebooks**: Data exploration + results analysis
- [x] **BONUS**: LaTeX table generation
- [x] **BONUS**: Summary report generation
- [x] **BONUS**: Directory structure setup

### Code Quality: ✅ Complete

- [x] Type hints on all functions
- [x] Comprehensive docstrings
- [x] Error handling
- [x] Logging
- [x] Clean, modular design

### Documentation: ✅ Complete

- [x] Inline documentation
- [x] Usage examples
- [x] Jupyter notebook documentation
- [x] This comprehensive summary

### Integration: ✅ Complete

- [x] Requirements.txt updated
- [x] Directory structure created
- [x] Import paths verified
- [x] All modules validated

---

## 🎉 Conclusion

Phase 2 provides a **complete analysis and visualization infrastructure** for your Hindi BabyLM thesis:

1. **Professional Analysis** - Statistical rigor with multiple tests
2. **Publication-Quality Figures** - Consistent, high-resolution plots
3. **LaTeX Integration** - Direct thesis inclusion
4. **Reproducible Workflow** - Jupyter notebooks document everything
5. **Time Savings** - Automated figure and table generation

The infrastructure is now **ready for generating all thesis materials** with a single command!

**Phase 2 Complete! Ready for final experiments and thesis writing! 🚀**

---

**Generated**: October 2025
**Project**: Hindi BabyLM - Data-Efficient Language Modeling for Hindi
**Status**: Phase 2 Analysis & Visualization - COMPLETED ✅
