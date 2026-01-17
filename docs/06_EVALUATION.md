# Evaluation Framework
<!-- Updated: 2026-01-11 -->

## Overview

The evaluation framework provides comprehensive assessment of trained Hindi language models across multiple dimensions: NLP tasks (IndicGLUE with 8 tasks) and syntactic competence (MultiBLiMP with 5 phenomena, 1,447 minimal pairs from HuggingFace dataset `jumelet/multiblimp`).

## Evaluation Philosophy

**Multi-Dimensional Assessment**:
1. **Task Performance**: How well does the model perform on downstream NLP tasks?
2. **Linguistic Competence**: Does the model understand Hindi grammar and syntax?
3. **Morphological Knowledge**: Can the model handle Hindi's rich morphology?

**Why Multiple Evaluations?**
- Single metric doesn't capture full competence
- Different tasks reveal different capabilities
- Hindi-specific phenomena need targeted evaluation

## Architecture

```
Trained Model + Tokenizer
         ↓
┌─────────────────────────────────────────────────────┐
│         Evaluation Manager                          │
│         (evaluation_manager.py)                     │
├─────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐ │
│  │   IndicGLUE Evaluator (indicglue_evaluator.py)│ │
│  │                                               │ │
│  │  Components:                                  │ │
│  │   • TaskRegistry - 8 task configs             │ │
│  │   • DataLoaderFactory - batch processing      │ │
│  │   • EvaluationStrategies (Classification/MC)  │ │
│  │   • FineTuningManager - optional fine-tuning  │ │
│  │   • ResultVisualizer - metrics & plots        │ │
│  │   • EvaluationCache - 30-day TTL cache        │ │
│  │                                               │ │
│  │  Tasks (8):                                   │ │
│  │   • BBCA (14 classes)                         │ │
│  │   • WSTP (4 choices)                          │ │
│  │   • CSQA (4 choices)                          │ │
│  │   • WinogradNLI (3 classes) - SKIPPED         │ │
│  │   • COPA (2 choices)                          │ │
│  │   • Movie Sentiment (3 classes)               │ │
│  │   • Product Sentiment (3 classes)             │ │
│  │   • Discourse Mode (6 classes)                │ │ 
│  │   Source: HuggingFace ai4bharat/indicglue     │ │
│  └────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────┐ │
│  │   MultiBLiMP Evaluator (multiblimp_evaluator) │ │
│  │                                                 │ │
│  │  Phenomena (5):                                │ │
│  │   • SV-# (407 pairs) - Number agreement       │ │
│  │   • SV-G (419 pairs) - Gender agreement       │ │
│  │   • SV-P (412 pairs) - Person agreement       │ │
│  │   • SP-# (100 pairs) - Predicate number       │ │
│  │   • SP-G (109 pairs) - Predicate gender       │ │
│  │                                                 │ │
│  │  Total: 1,447 minimal pairs                   │ │
│  │  Source: HuggingFace jumelet/multiblimp       │ │
│  └────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────┐ │
│  │   Supporting Components                        │ │
│  │                                                 │ │
│  │   • MetricsAggregator - Bootstrap CI (1000x)  │ │
│  │   • EvaluationCache - Hash-based caching      │ │
│  │   • ComparativeAnalyzer - Cross-model compare │ │
│  └────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────┘
                     ↓
            Comprehensive Results
             • JSON Report (evaluation_results.json)
             • CSV Summary (evaluation_summary.csv)
             • Cached Predictions (.eval_cache/)
             • Comparative Analysis (HTML/PDF reports)
```

## Evaluation Manager

**Location**: `src/evaluation/evaluation_manager.py:30`

**Purpose**: Orchestrates all evaluation tasks and compiles results.

### Implementation

```python
class EvaluationManager:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Initialize evaluators (no morphological probes in current implementation)
        self.indicglue_evaluator = IndicGLUEEvaluator(model, tokenizer, config)
        self.multiblimp_evaluator = MultiBLiMPEvaluator(model, tokenizer, config)

        # Results storage
        self.results = {}
```

**Key Features**:
- Manages both IndicGLUE and MultiBLiMP evaluation
- Handles result serialization with custom JSON encoder for dataclasses
- Saves results to experiment directory (not timestamp-based)
- Generates overall summary statistics across all evaluations

### Key Methods

#### `run_comprehensive_evaluation()` (line 76)

**Purpose**: Run all evaluation tasks and compile results

```python
def run_comprehensive_evaluation(self) -> Dict:
    """Run all evaluation tasks and compile results"""
    print("Starting comprehensive evaluation...")

    # 1. IndicGLUE Evaluation
    print("\n1. Running IndicGLUE evaluation...")
    indicglue_results = self.indicglue_evaluator.evaluate_all_tasks()
    self.results['indicglue'] = indicglue_results

    # 2. MultiBLiMP Evaluation
    print("\n2. Running MultiBLiMP evaluation...")
    multiblimp_results = self.multiblimp_evaluator.evaluate_all_phenomena()
    self.results['multiblimp'] = multiblimp_results

    # 3. Generate Summary
    summary = self.generate_summary()
    self.results['summary'] = summary

    # 4. Save Results
    self.save_results()

    return self.results
```

#### `generate_summary()` (line 99)

**Purpose**: Compile overall evaluation statistics

```python
def generate_summary(self) -> Dict:
    """Generate evaluation summary"""
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'model_config': self._make_serializable(self.config),
        'overall_scores': {}
    }

    # IndicGLUE average (defensive handling of missing results)
    if 'indicglue' in self.results:
        indicglue_scores = [v.get('accuracy', 0) for v in self.results['indicglue'].values()
                           if isinstance(v, dict) and 'accuracy' in v]
        if indicglue_scores:
            summary['overall_scores']['indicglue_avg'] = sum(indicglue_scores) / len(indicglue_scores)

    # MultiBLiMP overall (use correct key: 'average_accuracy' or 'overall_accuracy')
    if 'multiblimp' in self.results and 'overall' in self.results['multiblimp']:
        multiblimp_overall = self.results['multiblimp']['overall']
        # Try multiple possible keys for robustness
        summary['overall_scores']['multiblimp_accuracy'] = (
            multiblimp_overall.get('average_accuracy') or
            multiblimp_overall.get('overall_accuracy') or
            multiblimp_overall.get('accuracy', 0.0)
        )

    return summary
```

**Key Changes**:
- Uses `_make_serializable()` helper for config serialization (handles dataclasses)
- Defensive handling of missing results
- Multiple key fallbacks for MultiBLiMP (robustness)

#### `save_results()` (line 126)

**Purpose**: Save results to disk

**Implementation**:
```python
def save_results(self):
    """Save evaluation results to files"""
    # Save results to experiment directory (not timestamp-based)
    experiment_name = self.config.get('experiment_name', 'default_experiment')
    results_dir = os.path.join(self.config.get('results_dir', 'results'), experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    # Make results serializable before saving
    serializable_results = self._make_serializable(self.results)

    # Save comprehensive results as JSON with custom encoder
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False, cls=DataclassJSONEncoder)

    # Save summary as CSV for easy analysis
    summary_df = pd.DataFrame([self.results['summary']['overall_scores']])
    summary_file = os.path.join(results_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    print(f"Results saved to: {results_dir}")
    return results_dir
```

**Outputs**:
- `results/<experiment_name>/evaluation_results.json` - Full results with dataclass support
- `results/<experiment_name>/evaluation_summary.csv` - Summary table
- Saved to experiment directory

## 1. IndicGLUE Evaluation

**Location**: `src/evaluation/indicglue_evaluator.py:179`

**Purpose**: Evaluate on Hindi NLP benchmarks with optional fine-tuning

### Architecture

The IndicGLUE evaluator uses modular components for better testability and maintainability:

**Core Components** (in `src/evaluation/indicglue/`):
- **TaskRegistry** (`task_registry.py:38`) - Centralized task configurations with 8 tasks
- **TaskDataExtractor** (`data_extractor.py`) - Extracts text and labels from datasets
- **DataLoaderFactory** (`dataloader_factory.py`) - Creates task-specific dataloaders
- **EvaluationStrategies** (`evaluation_strategies.py`) - Classification, MC, and Perplexity strategies
- **FineTuningManager** (`fine_tuning_manager.py`) - Handles optional fine-tuning with early stopping
- **ResultVisualizer** (`result_visualizer.py`) - Computes metrics and creates visualizations

### IndicGLUE Tasks
<!-- Updated: 2026-01-11 -->

**8 Hindi NLP Tasks Implemented**:

| Task | HF Config | Type | Classes/Choices | Metric | Description |
|------|-----------|------|-----------------|--------|-------------|
| **BBCArticlesClassification** | bbca.hi | Classification | 14 classes | Accuracy | News category classification (business, china, entertainment, etc.) |
| **Wikipedia Section Title Prediction** | wstp.hi | Multiple Choice | 4 choices | Accuracy | Section title prediction from text |
| **Cloze-style multiple-choice QA** | csqa.hi | Multiple Choice | 4 choices | Accuracy | Commonsense question answering |
| **WinogradNLI** | wnli.hi | NLI | 3 classes | Accuracy | Natural language inference (premise-hypothesis) |
| **Choice of Plausible Alternatives** | copa.hi | Multiple Choice | 2 choices | Accuracy | Causal reasoning (cause/effect selection) |
| **MovieReviewSentiment** | iitp-mr.hi | Sentiment | 3 classes | Accuracy/F1 | Movie review sentiment (Negative/Neutral/Positive) |
| **ProductReviewSentiment** | iitp-pr.hi | Sentiment | 3 classes | Accuracy/F1 | Product review sentiment (Negative/Neutral/Positive) |
| **DiscourseMode** | md.hi | Classification | 6 classes | Accuracy | Discourse type (Narrative/Descriptive/Dialogue/Informative/Argumentative/Other) |

**Dataset Source**: HuggingFace `ai4bharat/indic_glue` with task-specific configs

**Implementation Location**: `src/evaluation/indicglue_evaluator.py:179`

**Task Registry**: All task configurations are centralized in `TaskRegistry` (`src/evaluation/indicglue/task_registry.py:38`)

### Implementation (Actual)

```python
class IndicGLUEEvaluator:
    """
    Comprehensive evaluator for IndicGLUE benchmark tasks

    Features:
    - All 8 IndicGLUE tasks supported
    - Multiple evaluation metrics per task
    - Batch processing for efficiency
    - Optional fine-tuning with early stopping
    - Evaluation caching (30-day TTL)
    - Bootstrap confidence intervals (1000 samples)
    """

    def __init__(self, model, tokenizer, config: Optional[Dict] = None,
                 model_provider: Optional[callable] = None):
        """
        Initialize IndicGLUE evaluator

        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer for the model
            config: Optional configuration dictionary
            model_provider: Optional callable for creating task-specific models
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = next(model.parameters()).device
        self.max_length = int(self.config.get('max_length', 128))  # IndicBERT uses 128

        # Initialize refactored components
        self.task_registry = TaskRegistry()  # 8 tasks
        self.data_extractor = TaskDataExtractor(self.task_registry)
        self.dataloader_factory = DataLoaderFactory(self.task_registry, self.data_extractor,
                                                     self.tokenizer, self.max_length, self.device)

        # Evaluation strategies
        self.classification_strategy = ClassificationStrategy(self.device, self.task_registry.get_task_config)
        self.mc_strategy = MultipleChoiceStrategy(self.device)
        self.perplexity_strategy = PerplexityStrategy(self.device, self.tokenizer,
                                                       self.max_length, self.data_extractor)

        # Fine-tuning manager (if enabled)
        self.fine_tuning_manager = FineTuningManager(self.config, self.dataloader_factory,
                                                      model_provider or self._get_model_for_task)

        # Metrics with bootstrap CI
        self.metrics_aggregator = MetricsAggregator(
            bootstrap_samples=1000,  # Default: 1000 bootstrap samples
            confidence_level=0.95    # 95% confidence intervals
        )

        # Evaluation cache (30-day TTL)
        eval_config = self.config.get('evaluation', {})
        self.cache_manager = EvaluationCache(
            cache_dir=eval_config.get('cache_dir', '.eval_cache'),
            max_cache_age_days=int(eval_config.get('max_cache_age_days', 30)),
            enable_cache=bool(eval_config.get('use_eval_cache', True))
        )

        # Result visualizer
        self.result_visualizer = ResultVisualizer(
            save_visualizations=eval_config.get('save_visualizations', True),
            visualization_formats=eval_config.get('visualization_format', ['png', 'html'])
        )

    def evaluate_all_tasks(self) -> Dict[str, Dict]:
        """
        Evaluate model on all IndicGLUE tasks

        Returns:
            Dictionary mapping task names to results
        """
        logger.info("Starting IndicGLUE evaluation on all tasks...")
        results = {}

        for task_name in self.task_registry.get_all_task_names():
            logger.info(f"\nEvaluating {task_name}...")

            try:
                task_results = self.evaluate_task(task_name)
                results[task_name] = task_results

                # Log results
                logger.info(f"{task_name} Results:")
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {task_name}: {str(e)}")
                results[task_name] = {'error': str(e), 'status': 'failed'}

        # Compute overall statistics
        results['overall'] = self.result_visualizer.compute_overall_metrics(results, self.task_registry)

        logger.info("\n" + "="*60)
        logger.info("IndicGLUE Evaluation Complete")
        logger.info(f"Overall Accuracy: {results['overall'].get('average_accuracy', 0):.4f}")
        logger.info("="*60)

        return results
```

### Evaluation Modes

The evaluator supports two main evaluation modes:

**1. Zero-Shot Evaluation**:
- No fine-tuning on task data
- Uses base language model directly
- Multiple-choice tasks use **perplexity-based scoring** (compares P(option1), P(option2), etc.)
- Classification tasks use **wrapped classification heads**
- Enabled when `evaluation.benchmarks.indicglue.fine_tuning.enabled: false`

**2. Fine-Tuned Evaluation**:
- Fine-tunes classification/MC heads on train split
- Uses trained heads for test evaluation
- Multiple-choice tasks use **trained MC wrapper** (matches official IndicBERT approach)
- Enabled when `evaluation.benchmarks.indicglue.fine_tuning.enabled: true`

**Fine-Tuning Configuration** (from `configs/base_config.yaml`):
```yaml
evaluation:
  benchmarks:
    indicglue:
      fine_tuning:
        enabled: true                    # Enable fine-tuning
        use_auto_models: false          # Use custom wrappers (better performance)
        freeze_base_model: false        # End-to-end training (recommended)
        num_epochs: 10                  # 10 epochs with early stopping
        learning_rate: 2e-5             # Standard for fine-tuning
        weight_decay: 0.01              # L2 regularization
        warmup_ratio: 0.1               # 10% warmup
        early_stopping:
          patience: 3                   # Stop after 3 epochs without improvement
          monitor_metric: 'eval_loss'   # Monitor validation loss
        dropout: 0.1                    # Dropout for classification heads
        label_smoothing: 0.0            # Label smoothing (0.0 = disabled)
```

**Fine-Tuning Features**:
- **Early Stopping**: Patience-based with best model restoration
- **Learning Rate Warmup**: Linear warmup for first 10% of steps
- **Adaptive Learning Rate**: Reduces LR on validation plateau
- **Gradient Clipping**: max_grad_norm=1.0
- **Memory Cleanup**: Explicit cleanup after each task

### Example Results

```json
{
  "indicglue": {
    "headlines_classification": {
      "accuracy": 0.72,
      "f1": 0.71
    },
    "ner": {
      "f1": 0.68,
      "precision": 0.71,
      "recall": 0.65
    },
    "sentiment": {
      "accuracy": 0.79,
      "f1": 0.78
    },
    "average": {
      "accuracy": 0.73,
      "f1": 0.72
    }
  }
}
```

## 2. MultiBLiMP Evaluation

**Location**: `src/evaluation/multiblimp_evaluator.py:32`

**Purpose**: Evaluate grammatical competence through minimal pairs

### What is MultiBLiMP?

**MultiBLiMP** (Multilingual BLiMP): Tests whether models assign lower loss (higher probability) to grammatical sentences than ungrammatical ones. Each test consists of a minimal pair: two sentences differing in a single linguistic feature.

**Minimal Pair Example**:
```
Grammatical:   लड़का खाता है।    (The boy eats.)
Ungrammatical: लड़का खाते हैं।   (Agreement error: plural verb with singular subject)
```

**Evaluation Method**:
- Compute loss for both sentences using language model
- Model is **correct** if loss(grammatical) < loss(ungrammatical)
- Accuracy = percentage of minimal pairs where grammatical sentence has lower loss

### Hindi Linguistic Phenomena (5 Categories, 1,447 Pairs)
<!-- Updated: 2026-01-11 -->

The implementation loads all 1,447 minimal pairs from HuggingFace dataset `jumelet/multiblimp` (Hindi split).

**Dataset**: `jumelet/multiblimp` (config: 'hin', split: 'train')

**Phenomena Tested** (5 agreement phenomena):

**Subject-Verb Agreement** (3 phenomena, 1,238 pairs):
1. **SV-#** (Subject-Verb Number Agreement): **407 minimal pairs**
   - Tests singular vs. plural verb agreement
   - Example: `लड़का खाता है` (correct) vs. `लड़का खाते हैं` (wrong)

2. **SV-G** (Subject-Verb Gender Agreement): **419 minimal pairs**
   - Tests masculine vs. feminine verb agreement
   - Example: `लड़का गया` (correct) vs. `लड़का गई` (wrong)

3. **SV-P** (Subject-Verb Person Agreement): **412 minimal pairs**
   - Tests first/second/third person verb agreement
   - Example: `मैं जाता हूँ` (correct) vs. `मैं जाता है` (wrong)

**Subject-Predicate Agreement** (2 phenomena, 209 pairs):
4. **SP-#** (Subject-Predicate Number Agreement): **100 minimal pairs**
   - Tests predicate number marking
   - Example: `लड़के अच्छे हैं` (correct) vs. `लड़के अच्छा है` (wrong)

5. **SP-G** (Subject-Predicate Gender Agreement): **109 minimal pairs**
   - Tests predicate gender marking
   - Example: `लड़की सुंदर है` (correct) vs. `लड़का सुंदर है` (wrong - gender mismatch)

**Total**: 5 phenomena, 1,447 minimal pairs

**Data Loading**: Automatic from HuggingFace with fallback error handling

### Implementation

```python
class MultiBLiMPEvaluator:
    """
    Comprehensive evaluator for Hindi syntactic phenomena using minimal pairs

    Features:
    - 5 linguistic phenomena tested (agreement phenomena)
    - Perplexity-based evaluation (loss comparison)
    - Comprehensive minimal pair database (1,447 pairs from HuggingFace)
    - Statistical analysis with mean loss differences
    - Per-phenomenon metrics
    - Overall syntactic competence score

    Location: src/evaluation/multiblimp_evaluator.py:32
    """

    def __init__(self, model, tokenizer, config: Optional[Dict] = None):
        """
        Initialize MultiBLiMP evaluator

        Args:
            model: Language model to evaluate (will unwrap if classification-wrapped)
            tokenizer: Tokenizer for the model
            config: Optional configuration dictionary

        Note:
            MultiBLiMP evaluation requires language modeling capabilities (per-token logits).
            If you pass a classification-wrapped model, it will be automatically unwrapped
            to access the base language model.
        """
        # Detect and unwrap classification models
        if self._is_classification_wrapper(model):
            logger.warning(
                f"Detected classification-wrapped model. MultiBLiMP requires language "
                f"modeling capabilities. Attempting to unwrap..."
            )
            try:
                model = self._unwrap_classification_model(model)
                logger.info(f"Successfully unwrapped to {model.__class__.__name__}")
            except ValueError as e:
                logger.error(str(e))
                raise

        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}

        # Device setup
        self.device = next(model.parameters()).device
        logger.info(f"MultiBLiMP evaluator initialized on device: {self.device}")

        # Get config parameters
        multiblimp_config = self.config.get('evaluation', {}).get('benchmarks', {}).get('multiblimp', {})
        self.max_examples_per_phenomenon = multiblimp_config.get('n_examples_per_phenomenon', None)

        # 5 phenomena from jumelet/multiblimp dataset
        self.phenomena = [
            'SV-#',   # Subject-Verb Number Agreement (407 pairs)
            'SV-G',   # Subject-Verb Gender Agreement (419 pairs)
            'SV-P',   # Subject-Verb Person Agreement (412 pairs)
            'SP-#',   # Subject-Predicate Number Agreement (100 pairs)
            'SP-G',   # Subject-Predicate Gender Agreement (109 pairs)
        ]

        # Load minimal pairs from HuggingFace
        self.minimal_pairs = self._initialize_minimal_pairs()

        # Log test coverage statistics
        total_pairs = sum(len(pairs) for pairs in self.minimal_pairs.values())
        logger.info(f"MultiBLiMP test coverage: {total_pairs} minimal pairs across {len(self.minimal_pairs)} phenomena")

        # Validate model outputs correct structure - fail fast if incompatible
        if not self.config.get('skip_init_validation', False):
            logger.info("Validating model output structure...")
            # ... validation code ...

    def evaluate_all_phenomena(self) -> Dict[str, Dict]:
        """
        Evaluate model on all syntactic phenomena

        Returns:
            Dictionary mapping phenomenon names to results
        """
        logger.info("Starting MultiBLiMP evaluation on all phenomena...")
        results = {}

        for phenomenon in self.phenomena:
            if phenomenon not in self.minimal_pairs:
                logger.warning(f"No minimal pairs found for {phenomenon}")
                continue

            logger.info(f"\nEvaluating {phenomenon}...")

            try:
                phenomenon_results = self.evaluate_phenomenon(
                    phenomenon,
                    self.minimal_pairs[phenomenon]
                )
                results[phenomenon] = phenomenon_results

                # Log results
                logger.info(f"{phenomenon} Results:")
                logger.info(f"  Accuracy: {phenomenon_results['accuracy']:.4f}")
                logger.info(f"  Correct: {phenomenon_results['correct']}/{phenomenon_results['total']}")

            except Exception as e:
                logger.error(f"Error evaluating {phenomenon}: {str(e)}")
                results[phenomenon] = {'error': str(e), 'status': 'failed'}

        # Compute overall statistics
        results['overall'] = self._compute_overall_metrics(results)

        logger.info("\n" + "="*60)
        logger.info("MultiBLiMP Evaluation Complete")
        logger.info(f"Overall Accuracy: {results['overall']['average_accuracy']:.4f}")
        logger.info("="*60)

        return results

    def evaluate_phenomenon(self, phenomenon: str) -> Dict:
        """
        Evaluate a specific linguistic phenomenon.

        For each minimal pair (good, bad), compute:
        - Perplexity of grammatical sentence
        - Perplexity of ungrammatical sentence
        - Model is correct if PPL(good) < PPL(bad)
        """
        pairs = self.minimal_pairs.get(phenomenon, [])

        if not pairs:
            return {'accuracy': 0.0, 'correct': 0, 'total': 0}

        correct = 0
        perplexity_diffs = []

        for pair in pairs:
            good_sent = pair['good']
            bad_sent = pair['bad']

            # Compute perplexities
            ppl_good = self.compute_perplexity(good_sent)
            ppl_bad = self.compute_perplexity(bad_sent)

            # Model should assign lower perplexity to grammatical sentence
            if ppl_good < ppl_bad:
                correct += 1

            perplexity_diffs.append(ppl_bad - ppl_good)

        accuracy = correct / len(pairs)

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(pairs),
            'avg_perplexity_diff': np.mean(perplexity_diffs),
            'std_perplexity_diff': np.std(perplexity_diffs)
        }

    def compute_perplexity(self, sentence: str) -> float:
        """
        Compute perplexity of a sentence using the language model.

        Lower perplexity = more likely/grammatical
        """
        # Tokenize
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids = inputs['input_ids'].to(self.device)

        # Compute loss (negative log likelihood)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        # Perplexity = exp(loss)
        perplexity = np.exp(loss)

        return perplexity
```

### Example Minimal Pairs by Phenomenon

**Subject-Verb Agreement (Number)**:
```python
{'good': 'लड़का खाता है', 'bad': 'लड़का खाते हैं'}     # Singular subject with singular verb
{'good': 'लड़के खाते हैं', 'bad': 'लड़के खाता है'}    # Plural subject with plural verb
```

**Subject-Verb Agreement (Gender)**:
```python
{'good': 'लड़का गया', 'bad': 'लड़का गई'}              # Masculine subject with masculine participle
{'good': 'लड़की गई', 'bad': 'लड़की गया'}             # Feminine subject with feminine participle
```

### Example Results

The evaluation returns detailed per-phenomenon results:

```json
{
  "multiblimp": {
    "subject_verb_agreement_number": {
      "accuracy": 0.88,
      "correct": 22,
      "total": 25,
      "avg_perplexity_diff": 12.4,
      "std_perplexity_diff": 3.2
    },
    "subject_verb_agreement_person": {
      "accuracy": 0.85,
      "correct": 17,
      "total": 20,
      "avg_perplexity_diff": 10.1,
      "std_perplexity_diff": 2.8
    },
    "subject_verb_agreement_gender": {
      "accuracy": 0.82,
      "correct": 16,
      "total": 20,
      "avg_perplexity_diff": 8.7,
      "std_perplexity_diff": 2.1
    },
    "overall": {
      "accuracy": 0.764,
      "correct": 214,
      "total": 280
    }
  }
}
```

**Key Metrics Explained**:
- **accuracy**: Proportion of minimal pairs where grammatical sentence has lower perplexity
- **avg_perplexity_diff**: Average difference in perplexity (higher = clearer distinction)
- **std_perplexity_diff**: Standard deviation of perplexity differences

## Supporting Components

### 1. Evaluation Cache

**Location**: `src/evaluation/evaluation_cache.py:29`

**Purpose**: Hash-based caching of evaluation predictions to avoid redundant inference runs

**Features**:
- SHA256 hash-based cache keys (model + dataset + config)
- 30-day TTL (configurable)
- Pickle serialization for predictions
- Metadata tracking (timestamps, model info, dataset info)
- Age-based cache cleanup

**Usage**:
```python
# Initialize cache manager
cache_manager = EvaluationCache(
    cache_dir='.eval_cache',
    max_cache_age_days=30,
    enable_cache=True
)

# Cache key generation
cache_key = cache_manager._compute_cache_key(
    model_hash='abc123...',
    dataset_name='BBCArticlesClassification',
    dataset_split='test',
    config={'batch_size': 32, 'max_samples': 1000}
)

# Retrieve cached predictions
cached = cache_manager.get_cached_predictions(cache_key)
if cached:
    predictions = cached['predictions']
    metadata = cached['metadata']

# Save predictions to cache
cache_manager.save_predictions(
    cache_key,
    predictions={'predictions': pred_list, 'labels': label_list},
    metadata={'task_name': 'BBCA', 'num_examples': 1000}
)

# Cache statistics
stats = cache_manager.get_cache_stats()
# Returns: {'enabled': True, 'total_entries': 15, 'expired_entries': 2,
#           'total_size_mb': 45.2, 'max_age_days': 30}
```

### 2. Metrics Aggregator

**Location**: `src/evaluation/metrics_utils.py:113`

**Purpose**: Advanced metrics computation with bootstrap confidence intervals

**Features**:
- Bootstrap resampling (default: 1000 samples, 95% CI)
- Multiple aggregation strategies (macro, micro, weighted)
- Per-class metrics computation
- Standardized Metric dataclass with CI
- McNemar's test for statistical significance

**Usage**:
```python
from src.evaluation.metrics_utils import MetricsAggregator, Metric

# Initialize aggregator
aggregator = MetricsAggregator(
    bootstrap_samples=1000,
    confidence_level=0.95,
    random_seed=42
)

# Compute single metric with CI
accuracy_metric = aggregator.compute_metric(
    y_true=[0, 1, 1, 0, 1],
    y_pred=[0, 1, 0, 0, 1],
    metric_name='accuracy',
    compute_ci=True
)
# Returns: Metric(name='accuracy', value=0.80, mean=0.80,
#                 ci_lower=0.60, ci_upper=1.00, n_samples=5)

# Compute per-class metrics
per_class = aggregator.compute_per_class_metrics(
    y_true=[0, 1, 1, 0, 1, 2],
    y_pred=[0, 1, 0, 0, 1, 2],
    class_names=['Negative', 'Neutral', 'Positive']
)
# Returns: {0: {'precision': Metric(...), 'recall': Metric(...), 'f1': Metric(...)}, ...}

# Statistical significance testing
sig_test = aggregator.compute_statistical_significance(
    y_true=labels,
    y_pred1=model1_predictions,
    y_pred2=model2_predictions,
    test='mcnemar'
)
# Returns: {'test': 'mcnemar', 'statistic': 2.5, 'p_value': 0.11,
#           'significant': False, 'contingency_table': {...}}
```

### 3. Comparative Analyzer

**Location**: `src/evaluation/comparative_analysis.py:27`

**Purpose**: Cross-model comparison and analysis with visualizations

**Features**:
- Side-by-side model comparison tables
- Multi-dimensional radar plots (Plotly)
- Training progression regression analysis
- Interactive HTML reports (Bootstrap styling)
- Publication-ready PDF reports (Matplotlib)

**Usage**:
```python
from src.evaluation.comparative_analysis import ComparativeAnalyzer

# Initialize analyzer
analyzer = ComparativeAnalyzer(
    results_dir='results/',
    output_dir='comparative_analysis'
)

# Load multiple evaluation results
analyzer.load_results({
    'GPT-2 Baseline': 'results/gpt2_baseline/evaluation_results.json',
    'GPT-2 + SentencePiece': 'results/gpt2_sp/evaluation_results.json',
    'DeBERTa Baseline': 'results/deberta_baseline/evaluation_results.json'
})

# Create comparison table
comparison_df = analyzer.create_comparison_table(
    result_names=['GPT-2 Baseline', 'DeBERTa Baseline'],
    metrics=['accuracy', 'f1_macro']
)

# Create radar plot
analyzer.create_radar_plot(
    result_names=['GPT-2 Baseline', 'DeBERTa Baseline'],
    metric='accuracy',
    save_path='comparative_analysis/radar_accuracy.html'
)

# Generate HTML report
analyzer.generate_html_report(
    result_names=['GPT-2 Baseline', 'DeBERTa Baseline'],
    include_radar=True,
    include_comparison=True
)

# Regression analysis for training progression
analysis = analyzer.regression_analysis(
    checkpoint_results={
        1000: results_1000_steps,
        2000: results_2000_steps,
        3000: results_3000_steps
    },
    task='BBCArticlesClassification',
    metric='accuracy'
)
# Returns: {'slope': 0.0001, 'r_squared': 0.95, 'best_checkpoint': 3000, ...}
```

## Complete Evaluation Example

```python
from src.models.model_factory import ModelFactory
from src.tokenization.tokenizer_factory import TokenizerFactory
from src.evaluation.evaluation_manager import EvaluationManager
from src.utils.experiment_config import ExperimentConfig

# 1. Load configuration
config = ExperimentConfig.load_config('configs/base_config.yaml')

# 2. Load trained model
model_factory = ModelFactory(config)
model = model_factory.load_trained_model('hindi_babylm_baseline')

# 3. Load tokenizer
tokenizer = TokenizerFactory.load_tokenizer('hindi_babylm_baseline')

# 4. Create evaluation manager
evaluator = EvaluationManager(model, tokenizer, config.__dict__)

# 5. Run comprehensive evaluation
results = evaluator.run_comprehensive_evaluation()

# 6. Print summary
print("\n=== Evaluation Summary ===")
print(f"IndicGLUE Average: {results['summary']['overall_scores']['indicglue_avg']:.2%}")
print(f"MultiBLiMP Accuracy: {results['summary']['overall_scores']['multiblimp_accuracy']:.2%}")

# Results saved to: results/<experiment_name>/
```

## Interpreting Results

**High IndicGLUE Scores**:
- Model captures semantic knowledge
- Good at practical NLP tasks
- Effective representations

**High MultiBLiMP Scores**:
- Model understands Hindi grammar
- Sensitivity to syntactic violations
- Implicit linguistic knowledge

### Diagnostic Analysis

**If IndicGLUE is low but MultiBLiMP is high**:
- Model has linguistic knowledge but lacks semantic/world knowledge
- May need more diverse training data

**If MultiBLiMP is low but IndicGLUE is high**:
- Model relies on statistical patterns rather than linguistic structure
- May overfit to training distribution

## Best Practices

1. **Multiple Metrics**: Don't rely on single evaluation
2. **Error Analysis**: Inspect failure cases
3. **Baseline Comparison**: Compare to random, majority, and prior work
4. **Significance Testing**: Use statistical tests for comparison
5. **Cross-Validation**: Multiple evaluation runs
6. **Qualitative Analysis**: Manual inspection of outputs

## Related Documentation

- [Training Pipeline Documentation](05_TRAINING.md)
- [Model Architecture Documentation](04_MODELS.md)
- [Configuration Guide](07_CONFIGURATION.md)
