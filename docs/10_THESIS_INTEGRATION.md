# Thesis Integration Guide

## Overview

This guide explains how to integrate Hindi BabyLM experimental results, figures, and tables into your thesis. The project's Phase 2 tools are specifically designed to generate publication-quality outputs ready for direct inclusion in LaTeX theses.

**Key Features**:
- **LaTeX tables**: Auto-generated with proper formatting, booktabs style
- **High-resolution figures**: 300 DPI PNG/PDF for print quality
- **Consistent styling**: All figures follow thesis formatting guidelines
- **Reproducible workflow**: Automated generation from experimental results

## Quick Start

### 1. Generate All Thesis Outputs

Run the results analysis notebook to generate all figures, tables, and reports:

```bash
# Launch Jupyter and run the notebook
jupyter lab notebooks/02_results_analysis.ipynb

# Or run non-interactively
jupyter nbconvert --execute --to notebook \
    --inplace notebooks/02_results_analysis.ipynb
```

This generates:
- `figures/` - PNG/PDF figures (300 DPI)
- `tables/` - LaTeX .tex tables
- `reports/` - Markdown experiment summaries

### 2. Copy Outputs to Thesis Directory

```bash
# Copy to thesis directory
cp figures/*.png /path/to/thesis/figures/
cp figures/*.pdf /path/to/thesis/figures/
cp tables/*.tex /path/to/thesis/tables/
```

### 3. Include in LaTeX

```latex
% In thesis preamble
\usepackage{booktabs}    % For professional tables
\usepackage{graphicx}    % For figures

% In chapters
\input{tables/indicglue_results.tex}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/training_curves.png}
    \caption{Training convergence across models}
    \label{fig:training_curves}
\end{figure}
```

## Thesis Structure

Typical thesis structure and where to include Hindi BabyLM outputs:

```
Thesis/
├── Chapter 1: Introduction
│   └── [No Hindi BabyLM outputs]
│
├── Chapter 2: Related Work
│   └── [No Hindi BabyLM outputs]
│
├── Chapter 3: Data and Methodology
│   ├── Section 3.1: Corpus Preparation
│   │   ├── figures/length_distributions.png
│   │   ├── figures/character_distribution.png
│   │   └── figures/data_quality.png
│   ├── Section 3.2: Morphological Richness
│   │   └── figures/case_markers.png
│   └── Section 3.3: Train/Val/Test Splits
│       └── [Statistics from corpus_statistics.json]
│
├── Chapter 4: Model Architecture
│   ├── Section 4.1: Position Encodings
│   └── Section 4.2: Model Sizes
│       └── figures/model_size_vs_performance.png
│
├── Chapter 5: Training Methodology
│   ├── Section 5.1: Optimization
│   ├── Section 5.2: Curriculum Learning
│   │   └── figures/curriculum_progression.png
│   └── Section 5.3: Training Convergence
│       └── figures/training_curves.png
│
├── Chapter 6: Evaluation
│   ├── Section 6.1: IndicGLUE Results
│   │   ├── tables/indicglue_results.tex
│   │   └── figures/indicglue_comparison.png
│   ├── Section 6.2: MultiBLiMP Syntactic Evaluation
│   │   ├── tables/multiblimp_results.tex
│   │   └── figures/multiblimp_comparison.png
│   └── Section 6.3: Morphological Probes
│       ├── tables/probes_results.tex
│       ├── figures/morphological_probes_comparison.png
│       └── figures/layer_wise_*.png (multiple figures)
│
├── Chapter 7: Results and Discussion
│   └── [Statistical significance results from notebook]
│
└── Chapter 8: Conclusion
    └── [No Hindi BabyLM outputs]
```

## Chapter 3: Data and Methodology

### Section 3.1: Corpus Preparation

**Narrative Example**:

```latex
\section{Corpus Preparation}

Our Hindi corpus consists of 10.2M tokens collected from three sources:
IndicCorp Hindi dataset (60\%), Hindi Wikipedia (30\%), and children's
literature (10\%). The corpus was processed through multiple quality
filters to ensure high-quality training data.

\subsection{Length Distribution}

Figure~\ref{fig:length_dist} shows the distribution of sentence lengths
in our training corpus. The mean sentence length is 15.3 words, with a
median of 12 words, indicating a slight right skew due to longer
narrative passages from literature sources.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/length_distributions.png}
    \caption{Sentence length distribution in word count (left) and
             character count (right). Red dashed line shows mean,
             green shows median.}
    \label{fig:length_dist}
\end{figure}

\subsection{Character Distribution}

Hindi uses the Devanagari script (Unicode range U+0900 to U+097F).
Our corpus exhibits 81.9\% Devanagari characters, with 87 unique
Devanagari characters appearing in the text. Figure~\ref{fig:char_dist}
shows the 30 most frequent Devanagari characters, with vowels and
common consonants (क, त, र, न) dominating.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/character_distribution.png}
    \caption{Top 30 most frequent Devanagari characters in training corpus}
    \label{fig:char_dist}
\end{figure}

\subsection{Data Quality}

We applied multiple quality filters to ensure corpus cleanliness.
Figure~\ref{fig:quality} shows that 87.5\% of sentences passed all
quality checks. The primary rejection reasons were sentences that
were too short ($<$5 words, 5.2\%) or contained insufficient Hindi
content ($<$70\% Devanagari, 3.8\%).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/data_quality.png}
    \caption{Distribution of data quality categories in training corpus}
    \label{fig:quality}
\end{figure}
```

### Section 3.2: Morphological Richness

**Narrative Example**:

```latex
\subsection{Morphological Complexity}

Hindi is a morphologically rich language with extensive case marking
and agreement systems. To quantify the morphological complexity of
our corpus, we analyzed the distribution of eight common case markers:
ने (ergative), को (accusative/dative), से (instrumental/ablative),
में (locative), पर (locative/temporal), and the genitive markers
का/की/के.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.75\textwidth]{figures/case_markers.png}
    \caption{Distribution of Hindi case markers in training corpus.
             Genitive markers (का/की/के) are most frequent, followed
             by accusative (को) and ergative (ने).}
    \label{fig:case_markers}
\end{figure}

As shown in Figure~\ref{fig:case_markers}, genitive markers are most
frequent (ka: 145,234; ki: 98,765; ke: 87,654), reflecting the common
use of possession constructions. The ergative marker ne appears 65,432
times, validating the presence of perfective transitive constructions.
This rich morphological marking provides ample training signal for
models to learn Hindi's case system.
```

## Chapter 5: Training Methodology

### Section 5.3: Training Convergence

**Narrative Example**:

```latex
\section{Training Convergence}

We trained three models: a baseline GPT model, a curriculum learning
variant, and an enhanced model with RoPE position encodings. All
models were trained for 50 epochs with AdamW optimization.

\subsection{Loss Curves}

Figure~\ref{fig:training_curves} shows training and validation loss
curves for all three models. The curriculum learning model exhibits
smoother convergence with less oscillation, while the enhanced model
achieves the lowest final validation loss (2.21) compared to baseline
(2.45) and curriculum (2.31).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/training_curves.png}
    \caption{Training loss (left) and perplexity (right) for three
             model variants across 50 epochs. Curriculum learning
             provides more stable training, while position encoding
             improvements yield best final performance.}
    \label{fig:training_curves}
\end{figure}

The baseline model shows early overfitting (gap between train and
validation) around epoch 30. Both curriculum and enhanced models
mitigate this, with curriculum learning providing better regularization
through gradual difficulty increase.
```

## Chapter 6: Evaluation

### Section 6.1: IndicGLUE Results

**Narrative Example**:

```latex
\section{Downstream Task Performance}

We evaluate our trained models on IndicGLUE, a benchmark for Hindi
NLP tasks including classification, named entity recognition, and
sentiment analysis.

\subsection{IndicGLUE Benchmark}

Table~\ref{tab:indicglue} presents IndicGLUE results across six tasks.
The enhanced model achieves the best average accuracy (73.1\%),
representing a 3.1 percentage point improvement over the baseline
(70.0\%). Curriculum learning provides a 1.8 point improvement
(71.8\%), demonstrating its effectiveness.

\input{tables/indicglue_results.tex}

Figure~\ref{fig:indicglue_comp} visualizes these results across all
tasks. The enhanced model shows consistent improvements across all
tasks, with largest gains on NER tasks (Soham NER: +3.5\%, WikiANN:
+2.6\%). This suggests that better positional encoding helps with
sequence labeling tasks requiring precise positional information.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/indicglue_comparison.png}
    \caption{IndicGLUE task performance comparison across three models.
             Enhanced model (orange) outperforms baseline (blue) and
             curriculum (green) on all tasks.}
    \label{fig:indicglue_comp}
\end{figure}
```

### Section 6.2: MultiBLiMP Syntactic Evaluation

**Narrative Example**:

```latex
\section{Syntactic Competence}

We evaluate syntactic understanding using MultiBLiMP, a minimal pair
benchmark testing 14 linguistic phenomena including agreement, case
marking, and word order.

\subsection{MultiBLiMP Results}

Table~\ref{tab:multiblimp} shows results across all 14 phenomena.
Models perform best on agreement phenomena (number: 88\%, gender: 82\%)
and worst on structural phenomena (binding: 65\%, control: 63\%),
consistent with typological complexity.

\input{tables/multiblimp_results.tex}

The enhanced model achieves 76.4\% overall accuracy, outperforming
baseline (72.1\%) and curriculum (74.3\%). Curriculum learning
particularly helps with complex phenomena like case marking
(ergative +4.2\%, accusative +3.8\%), suggesting that gradual
exposure to morphological complexity aids learning.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/multiblimp_comparison.png}
    \caption{MultiBLiMP syntactic phenomena results. Horizontal bars
             show accuracy for each of 14 phenomena across three models.}
    \label{fig:multiblimp}
\end{figure}

Figure~\ref{fig:multiblimp} visualizes the per-phenomenon breakdown.
All models struggle with control structures and binding, likely due
to their rarity in the 10M token corpus. However, the enhanced model's
improved position encoding helps with word order phenomena (+6.1\%),
which require precise positional information.
```

### Section 6.3: Morphological Probes

**Narrative Example**:

```latex
\section{Morphological Understanding}

To assess morphological competence, we employ layer-wise linear probing
on 10 morphological tasks: case, number, gender, tense, person, aspect,
mood, voice, honorific, and definiteness detection.

\subsection{Layer-wise Probing Results}

Table~\ref{tab:probes} shows best-layer accuracy for each probe task.
Surface features (number: 91\%, gender: 88\%) are learned better than
abstract features (voice: 68\%, mood: 71\%).

\input{tables/probes_results.tex}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/morphological_probes_comparison.png}
    \caption{Best-layer accuracy for 10 morphological probe tasks
             across three models}
    \label{fig:probes_comp}
\end{figure}

\subsection{Layer-wise Analysis}

Figure~\ref{fig:layer_case} shows layer-wise probing results for case
detection. Accuracy increases from layer 0 (embedding: 45\%) to peak
at layer 8 (84\%), then declines in later layers (layer 12: 77\%).
This pattern indicates that morphological information is most
accessible in middle layers, consistent with findings in English
probing studies.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/layer_wise_case_probe.png}
    \caption{Layer-wise case detection accuracy. Peak performance
             occurs at layer 8 (84\%), with gradual increase from
             embedding layer and decline in final layers.}
    \label{fig:layer_case}
\end{figure}

Similar patterns emerge for other morphological features
(Figures~\ref{fig:layer_number}, \ref{fig:layer_gender},
\ref{fig:layer_tense}), with surface features peaking earlier
(layers 5-7) and abstract features peaking later (layers 9-11).

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/layer_wise_number_probe.png}
        \caption{Number detection}
        \label{fig:layer_number}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/layer_wise_gender_probe.png}
        \caption{Gender detection}
        \label{fig:layer_gender}
    \end{subfigure}

    \vspace{0.5cm}

    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/layer_wise_tense_probe.png}
        \caption{Tense detection}
        \label{fig:layer_tense}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/layer_wise_person_probe.png}
        \caption{Person detection}
        \label{fig:layer_person}
    \end{subfigure}

    \caption{Layer-wise probing results for four morphological features.
             Surface features (number, gender) peak earlier than abstract
             features (tense, person).}
    \label{fig:layer_morphology}
\end{figure}
```

## Chapter 7: Results and Discussion

### Statistical Significance

**Narrative Example**:

```latex
\section{Statistical Significance}

To rigorously assess whether observed improvements are statistically
significant, we perform paired t-tests and Wilcoxon signed-rank tests
comparing the enhanced model to the baseline across IndicGLUE tasks.

\subsection{Enhanced vs. Baseline}

The enhanced model achieves a mean accuracy of 0.731 ± 0.042 compared
to baseline's 0.700 ± 0.051, a difference of 3.1 percentage points.
Paired t-test yields t=2.45, p=0.013, indicating statistical
significance at α=0.05. The non-parametric Wilcoxon test confirms
this (p=0.020).

Cohen's d effect size is 0.68, indicating a medium effect size.
Bootstrap confidence intervals (10,000 resamples) yield a 95\% CI of
[0.009, 0.058], which excludes zero, confirming significant improvement.

\subsection{Curriculum vs. Baseline}

Curriculum learning shows smaller but significant improvements (mean
difference: 1.8 points, t=1.98, p=0.047, Cohen's d=0.42). The 95\% CI
is [0.001, 0.036], indicating a small but reliable effect.

\subsection{Combined Improvements}

The combination of curriculum learning and enhanced position encodings
(not tested separately) could provide additive benefits, a promising
direction for future work.
```

## LaTeX Best Practices

### Table Formatting

**Use booktabs package for professional tables**:

```latex
% In preamble
\usepackage{booktabs}

% Generated tables already use booktabs style
\input{tables/indicglue_results.tex}
```

**Custom table width**:

```latex
% If table is too wide
\resizebox{\textwidth}{!}{%
\input{tables/indicglue_results.tex}
}
```

**Rotate wide tables**:

```latex
\usepackage{rotating}

\begin{sidewaystable}
\input{tables/multiblimp_results.tex}
\end{sidewaystable}
```

### Figure Formatting

**Standard figure**:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/training_curves.png}
    \caption{Training convergence curves}
    \label{fig:training}
\end{figure}
```

**Side-by-side figures**:

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/indicglue_comparison.png}
        \caption{IndicGLUE}
        \label{fig:indicglue}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/multiblimp_comparison.png}
        \caption{MultiBLiMP}
        \label{fig:multiblimp}
    \end{subfigure}
    \caption{Evaluation results across benchmarks}
    \label{fig:eval}
\end{figure}
```

**High-quality PDF figures** (recommended):

```latex
% Use PDF instead of PNG for vector graphics
\includegraphics[width=0.8\textwidth]{figures/training_curves.pdf}
```

### Cross-referencing

**Consistent labeling**:

```latex
% Figures
\label{fig:training_curves}
\ref{fig:training_curves}

% Tables
\label{tab:indicglue}
\ref{tab:indicglue}

% Sections
\label{sec:evaluation}
\ref{sec:evaluation}
```

**Multiple references**:

```latex
% Multiple figures
Figures~\ref{fig:indicglue}, \ref{fig:multiblimp}, and \ref{fig:probes}

% Range
Figures~\ref{fig:probe_case}--\ref{fig:probe_gender}
```

## Regenerating Outputs

### Update Figures After Changes

If you rerun experiments and need to regenerate all figures:

```bash
# Step 1: Run new experiments (generates new results/)
python main.py --config configs/new_experiment.yaml

# Step 2: Regenerate all figures and tables
jupyter nbconvert --execute --to notebook \
    --inplace notebooks/02_results_analysis.ipynb

# Step 3: Copy to thesis
cp figures/*.png /path/to/thesis/figures/
cp tables/*.tex /path/to/thesis/tables/

# Step 4: Recompile thesis
cd /path/to/thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Automated Regeneration Script

Create `scripts/regenerate_thesis_outputs.sh`:

```bash
#!/bin/bash

echo "Regenerating all thesis outputs..."

# Run results analysis notebook
echo "Running results analysis..."
jupyter nbconvert --execute --to notebook \
    --inplace notebooks/02_results_analysis.ipynb

# Copy outputs
echo "Copying to thesis directory..."
THESIS_DIR="/path/to/thesis"
cp figures/*.png "${THESIS_DIR}/figures/"
cp figures/*.pdf "${THESIS_DIR}/figures/"
cp tables/*.tex "${THESIS_DIR}/tables/"

echo "✅ All outputs regenerated and copied to thesis"
```

Usage:

```bash
chmod +x scripts/regenerate_thesis_outputs.sh
./scripts/regenerate_thesis_outputs.sh
```

## Version Control for Thesis

### Track Output Changes

```bash
# In thesis repository, track generated files
git add figures/*.png tables/*.tex
git commit -m "Update experimental results (Hindi BabyLM v2.1)"

# Tag important versions
git tag -a v1.0-defense -m "Figures and tables for thesis defense"
```

### Keep Source and Outputs in Sync

```bash
# Document which Hindi BabyLM version generated outputs
echo "Generated from Hindi BabyLM commit: abc123" > figures/README.md
git add figures/README.md
```

## Publication Beyond Thesis

### Conference Papers

Figures and tables are publication-ready for:
- ACL/EMNLP/NAACL (Computational Linguistics)
- LREC/COLING (Language Resources)
- EACL (European ACL)

**Format adjustments**:
- Use PDF figures for camera-ready submissions
- Ensure figure captions are self-contained
- Check conference-specific style files for table formatting

### Journal Papers

For extended journal versions:
- Include additional layer-wise probe figures
- Add appendices with complete statistical tables
- Provide supplementary material with raw results

## Checklist for Thesis Submission

### Before Submission

- [ ] All figures generated at 300+ DPI
- [ ] All tables compiled without LaTeX errors
- [ ] All figure references working (\ref{fig:...})
- [ ] All table references working (\ref{tab:...})
- [ ] Figure captions are descriptive and self-contained
- [ ] Table captions explain abbreviations
- [ ] Statistical significance properly reported (p-values, effect sizes)
- [ ] All generated outputs have consistent styling
- [ ] PDF generated successfully with pdflatex
- [ ] All Devanagari text renders correctly (use appropriate fonts)
- [ ] Experimental details match README.md in code repository

### Archive for Future Reference

```bash
# Create archive of all outputs
tar -czf hindi_babylm_thesis_outputs_v1.0.tar.gz \
    figures/ tables/ reports/ notebooks/ \
    data/corpus_statistics.json

# Store in cloud/institutional repository
```

## Related Documentation

- [Analysis and Visualization Documentation](08_ANALYSIS_AND_VISUALIZATION.md) - Tools for generating outputs
- [Jupyter Notebooks Documentation](09_JUPYTER_NOTEBOOKS.md) - Interactive analysis workflows
- [Results Analyzer API](08_ANALYSIS_AND_VISUALIZATION.md#1-resultsanalyzer) - Detailed API reference
- [Thesis Plotter API](08_ANALYSIS_AND_VISUALIZATION.md#2-thesisplotter) - Visualization options
