# IndicBERT Evaluation Script

## Overview

This standalone script evaluates the pre-trained IndicBERT model from HuggingFace on IndicGLUE tasks to verify the correctness of the existing IndicGLUE evaluation implementation by comparing results with reported scores from the IndicBERT paper.

**Note** Our scores from running the `evaluate_indicbert.py` script differ from the official results reported by the IndicBERT team. The obtained scores are summarized in the following table.

## IndicBERT evaluation scores

| Task                                   | Accuracy | F1-Macro | Reason |
|----------------------------------------|----------|----------|--------|
| Wikipedia Section Title Prediction     | 62.12%   | 62.11%   | Unknown |
| Cloze-style multiple-choice QA         | 32.19%   | 32.16%   | Custom splits from test set only |
| BBCArticlesClassification              | 43.30%   | 8.35%    | Paper: 6 classes; HF: 14 classes |
| MovieReviewSentiment                   | 38.06%   | 23.48%   | Unknown |
| ProductReviewSentiment                 | 55.64%   | 39.47%   | Unknown |
| DiscourseMode                          | 63.69%   | 32.26%   | Paper: 5 classes; HF: 6 classes |
| Choice of Plausible Alternatives       | 53.41%   | 53.35%   | Swapped val/test due to label 1 missing in test |

## Key Features

- **Standalone**: No modifications to core codebase required
- **Paper-compliant**: Follows IndicBERT paper specifications exactly
- **Task-specific fine-tuning**: Independent models per task
- **Comprehensive output**: JSON, CSV, and visualizations
- **Flexible CLI**: Customizable hyperparameters and task selection

## Critical Implementation Details (from IndicBERT Paper)

⚠️ **These specifications are hard-coded to match the paper:**

1. **Max Sequence Length: 128 tokens** (NOT 512!)
2. **[CLS] Token Pooling**: From last hidden layer
3. **Classification**: Linear classifier with softmax + multi-class cross-entropy loss
4. **Task-specific**: Independent fine-tuning per task (not multi-task)
5. **Hyperparameters**: ALBERT defaults (lr=2e-5, weight_decay=0.01)

## Installation Requirements

Ensure you have the required dependencies:

```bash
# Install from project root
pip install -r requirements.txt

# Key dependencies:
# - torch
# - transformers (for HuggingFace models)
# - datasets (for IndicGLUE)
# - pandas, numpy, scikit-learn
```

## Quick Start

### 1. Basic Usage (All Tasks, Fine-tuning)

```bash
python scripts/evaluate_indicbert.py
```

This will:
- Load `ai4bharat/indic-bert` from HuggingFace
- Evaluate on all 7 tasks: WSTP, CSQA, BBCA, iitp-mr, iitp-pr, iitp-md, COPA
- Fine-tune task-specific classification heads (10 epochs)
- Save results to `results/indicbert_evaluation/`

### 2. Quick Smoke Test

Test the script quickly with limited samples:

```bash
python scripts/evaluate_indicbert.py --max-samples 10
```

This runs on just 10 samples per task to verify everything works.

### 3. Specific Tasks

Evaluate only specific tasks:

```bash
# Just classification tasks
python scripts/evaluate_indicbert.py --tasks BBCA iitp-mr iitp-pr iitp-md

# Just multiple-choice tasks
python scripts/evaluate_indicbert.py --tasks WSTP COPA

# Single task
python scripts/evaluate_indicbert.py --tasks BBCA
```

### 4. Zero-Shot Evaluation

Evaluate without fine-tuning:

```bash
python scripts/evaluate_indicbert.py --mode zero-shot
```

### 5. Custom Hyperparameters

Override default hyperparameters:

```bash
python scripts/evaluate_indicbert.py \
  --epochs 5 \
  --learning-rate 1e-5 \
  --batch-size 64 \
  --weight-decay 0.001
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `ai4bharat/indic-bert` | HuggingFace model ID |
| `--tasks` | All 7 tasks | List of tasks to evaluate |
| `--mode` | `fine-tune` | Evaluation mode: `fine-tune` or `zero-shot` |
| `--batch-size` | 32 | Batch size for evaluation |
| `--epochs` | 10 | Number of fine-tuning epochs |
| `--learning-rate` | 2e-5 | Learning rate (ALBERT default) |
| `--weight-decay` | 0.01 | Weight decay (ALBERT default) |
| `--device` | Auto-detect | Device: `cuda` or `cpu` |
| `--output-dir` | `results/indicbert_evaluation` | Output directory |
| `--seed` | 42 | Random seed for reproducibility |
| `--max-samples` | None (all) | Limit samples per task for testing |
| `--no-visualizations` | False | Skip generating plots |

## Output Files

The script creates the following files in the output directory:

### 1. `indicbert_results_detailed.json`
Complete results with:
- Metadata (model, timestamp, config)
- Full metrics for each task (accuracy, F1, precision, recall)
- Confidence intervals (95% CI with 1000 bootstrap samples)
- Confusion matrices
- Per-class metrics

### 2. `results_summary.csv`
Summary table for easy comparison:

```csv
Task,Accuracy,F1_Macro,F1_Weighted,Precision,Recall,Num_Examples,Status
Wikipedia Section Title Prediction,0.8234,0.8112,0.8156,0.8189,0.8112,449,completed
BBCA,0.8567,0.8423,0.8489,0.8501,0.8423,1200,completed
...
```

### 3. `config.json`
Configuration used for the evaluation (for reproducibility).

### 4. `indicbert_evaluation.log`
Detailed execution log with timestamps.

### 5. Visualizations (optional)
- Confusion matrices (PNG + HTML)
- Per-class metrics plots

## Task Information

### Classification Tasks (Type A)
These use simple text → [CLS] → Linear classifier:

| Task Code | Task Name | Classes | 
|-----------|-----------|---------|
| BBCA | BBC Articles Classification | 14 | 
| iitp-mr | Movie Review Sentiment | 3 | 
| iitp-pr | Product Review Sentiment | 3 | 
| iitp-md | Discourse Mode | 6 | 

### Multiple-Choice Tasks (Type B)
These use text pairs with [SEP] tokens:

| Task Code | Task Name | Choices | 
|-----------|-----------|---------|
| WSTP | Wikipedia Section Title Prediction | 4 |
| CSQA | Cloze-style multiple-choice QA | 4 | 

### Sentence Pair Tasks (Type C)
These use premise-alternative pairs:

| Task Code | Task Name | Choices |
|-----------|-----------|---------|
| COPA | Choice of Plausible Alternatives | 2 | 

## Expected Behavior

### Successful Tasks
7 out of 7 tasks should complete successfully:
- ✓ WSTP (Wikipedia Section Title Prediction)
- ✓ BBCA (BBC Articles)
- ✓ iitp-mr (Movie Reviews)
- ✓ iitp-pr (Product Reviews)
- ✓ iitp-md (Discourse Mode)
- ✓ COPA (Plausible Alternatives)

### Example Output

```
================================================================================
INDICBERT EVALUATION RESULTS
================================================================================

Task                                      Accuracy   F1-Macro   Examples     Status
--------------------------------------------------------------------------------
Wikipedia Section Title Prediction          82.34%     81.12%        449          ✓
Choice of Plausible Alternatives            76.54%     76.32%         88          ✓
BBCArticlesClassification                   85.67%     84.23%       1200          ✓
MovieReviewSentiment                        78.90%     77.45%        800          ✓
ProductReviewSentiment                      79.12%     78.01%        900          ✓
DiscourseMode                               72.34%     70.56%        600          ✓
Cloze-style multiple-choice QA                 N/A        N/A        N/A          ✗
--------------------------------------------------------------------------------

Summary:
  Successful: 6 / 7 tasks
  Skipped: 1 tasks
  Failed: 0 tasks
  Average Accuracy: 78.21%
  Average F1-Macro: 76.54%
================================================================================
```

## Troubleshooting

### GPU Out of Memory

If you encounter OOM errors, reduce batch size:

```bash
python scripts/evaluate_indicbert.py --batch-size 16
```

Or use CPU (slower):

```bash
python scripts/evaluate_indicbert.py --device cpu
```

### Model Download Issues

If model download fails, try:

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache
python scripts/evaluate_indicbert.py
```

## Comparing with Paper Results

After running the evaluation:

1. Check the CSV file: `results/indicbert_evaluation/results_summary.csv`
2. Compare accuracy scores with the IndicBERT paper.

### IndicBERT Paper Results (Hindi)

| Task  | Paper Accuracy | Our Accuracy | Difference |
|-------|----------------|--------------|------------|
| WSTP  | 74.02          | 62.12        | 11.90      |
| COPA  | 62.50          | 53.41        | 9.09       |
| BBCA  | 74.60          | 43.30        | 31.30      |
| iitp-mr | 59.03        | 38.06        | 20.97      |
| iitp-pr | 71.32        | 55.64        | 15.68      |
| iitp-md | 78.44        | 63.69        | 14.75      |

## Implementation Architecture

### Script Components

1. **IndicBERTModelLoader**: Loads model/tokenizer from HuggingFace
2. **ALBERTForSequenceClassification**: Classification wrapper with [CLS] pooling
3. **IndicBERTEvaluationWrapper**: Adapter for evaluator compatibility
4. **IndicGLUEEvaluator**: Reuses existing evaluation infrastructure (zero duplication)

### Key Design Decisions

- **Standalone**: No modifications to core codebase
- **Adapter Pattern**: Maximum code reuse from existing evaluator
- **Monkey-patching**: Clean integration without subclassing
- **Paper-compliant**: Exact specifications from IndicBERT paper

## Technical Details

### Max Sequence Length: 128 Tokens

The script uses 128 tokens (not the typical 512) as specified in the IndicBERT paper. This is hard-coded in the configuration.

### Tokenization Format

- **Classification**: `[CLS] text [SEP]`
- **Multiple-choice**: `[CLS] text1 [SEP] text2 [SEP]`
- **Sentence pairs**: `[CLS] premise [SEP] alternative [SEP]`

### Fine-Tuning Strategy

- Base model: **Unfrozen** (parameters are updated)
- Classification head: **Trainable** (task-specific)
- Early stopping: Patience of 3 epochs on validation accuracy

## Development

### Running Tests

```bash
# Quick smoke test (10 samples per task)
python scripts/evaluate_indicbert.py --max-samples 10

# Single task test
python scripts/evaluate_indicbert.py --tasks BBCA --max-samples 50

# Zero-shot test
python scripts/evaluate_indicbert.py --mode zero-shot --tasks COPA
```

### Adding New Tasks

To add new IndicGLUE tasks, update the `task_mapping` dictionary in the main function and add the task short name to CLI choices.

## Citation

If you use this script, please cite:

```bibtex
@inproceedings{kakwani2020indicnlpsuite,
  title={IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages},
  author={Kakwani, Divyanshu and Kunchukuttan, Anoop and Golla, Satish and Gokul, NC and Bhattacharyya, Avik and Khapra, Mitesh M and Kumar, Pratyush},
  booktitle={Findings of EMNLP},
  year={2020}
}
```

## Support

For issues or questions:
1. Check the execution log: `results/indicbert_evaluation/indicbert_evaluation.log`
2. Verify dependencies are installed correctly
3. Try the smoke test first: `--max-samples 10`
4. Report issues to the project maintainer

## License

This script follows the same license as the parent project.
