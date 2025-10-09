# Evaluation Framework

## Overview

The evaluation framework provides comprehensive assessment of trained Hindi language models across multiple dimensions: NLP tasks (IndicGLUE), syntactic competence (MultiBLiMP), and morphological understanding (custom probes).

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
┌────────────────────────────────────┐
│    Evaluation Manager              │
├────────────────────────────────────┤
│  ┌──────────────────────────────┐ │
│  │   IndicGLUE Evaluator        │ │
│  │   • Classification           │ │
│  │   • NER                       │ │
│  │   • Question Answering       │ │
│  └──────────────────────────────┘ │
│  ┌──────────────────────────────┐ │
│  │   MultiBLiMP Evaluator       │ │
│  │   • Agreement                │ │
│  │   • Case Marking             │ │
│  │   • Word Order               │ │
│  └──────────────────────────────┘ │
│  ┌──────────────────────────────┐ │
│  │   Morphological Probes       │ │
│  │   • Paradigm Consistency     │ │
│  │   • Case Assignment          │ │
│  │   • Verbal Agreement         │ │
│  └──────────────────────────────┘ │
└────────────────┬───────────────────┘
                 ↓
        Comprehensive Results
         • JSON Report
         • CSV Summary
         • Visualizations
```

## Evaluation Manager

**Location**: `src/evaluation/evaluation_manager.py:7`

**Purpose**: Orchestrates all evaluation tasks and compiles results.

### Implementation

```python
class EvaluationManager:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Initialize evaluators
        self.indicglue_evaluator = IndicGLUEEvaluator(model, tokenizer)
        self.multiblimp_evaluator = MultiBLiMPEvaluator(model, tokenizer)
        self.morphological_probe = MorphologicalProbe(model, tokenizer)

        # Results storage
        self.results = {}
```

### Key Methods

#### `run_comprehensive_evaluation()` (line 21)

**Purpose**: Run all evaluation tasks

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

    # 3. Morphological Probes
    print("\n3. Running morphological probes...")
    probe_results = self.morphological_probe.run_all_probes()
    self.results['morphological_probes'] = probe_results

    # 4. Generate Summary
    summary = self.generate_summary()
    self.results['summary'] = summary

    # 5. Save Results
    self.save_results()

    return self.results
```

#### `generate_summary()` (line 49)

**Purpose**: Compile overall evaluation statistics

```python
def generate_summary(self) -> Dict:
    """Generate evaluation summary"""
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'model_config': self.config,
        'overall_scores': {}
    }

    # IndicGLUE average
    indicglue_scores = [
        v.get('accuracy', 0)
        for v in self.results['indicglue'].values()
        if 'accuracy' in v
    ]
    if indicglue_scores:
        summary['overall_scores']['indicglue_avg'] = sum(indicglue_scores) / len(indicglue_scores)

    # MultiBLiMP overall
    if 'overall' in self.results['multiblimp']:
        summary['overall_scores']['multiblimp_accuracy'] = \
            self.results['multiblimp']['overall']['accuracy']

    # Morphological probes average
    probe_scores = [
        v.get('accuracy', 0)
        for v in self.results['morphological_probes'].values()
        if 'accuracy' in v
    ]
    if probe_scores:
        summary['overall_scores']['morphological_avg'] = sum(probe_scores) / len(probe_scores)

    return summary
```

#### `save_results()` (line 73)

**Purpose**: Save results to disk

**Outputs**:
- `results/evaluation_{timestamp}/evaluation_results.json` - Full results
- `results/evaluation_{timestamp}/evaluation_summary.csv` - Summary table

## 1. IndicGLUE Evaluation

**Location**: `src/evaluation/indicglue_evaluator.py`

**Purpose**: Evaluate on Hindi NLP benchmarks

### IndicGLUE Tasks

| Task | Type | Metric | Description |
|------|------|--------|-------------|
| **INLTKH Headlines** | Classification | Accuracy | News category classification |
| **BBC Hindi** | Classification | Accuracy | Document classification |
| **IITP Movie Reviews** | Sentiment | F1 | Sentiment analysis |
| **IITP Product Reviews** | Sentiment | F1 | Product review sentiment |
| **Soham NER** | NER | F1 | Named entity recognition |
| **WikiANN NER** | NER | F1 | Wikipedia NER |

### Implementation (Conceptual)

```python
class IndicGLUEEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_all_tasks(self) -> Dict[str, Dict[str, float]]:
        """Evaluate on all IndicGLUE tasks"""
        results = {}

        for task_name in self.tasks:
            print(f"Evaluating {task_name}...")
            task_results = self.evaluate_task(task_name)
            results[task_name] = task_results

        # Compute average
        results['average'] = self.compute_average(results)

        return results

    def evaluate_task(self, task_name: str) -> Dict[str, float]:
        """Evaluate on a specific task"""
        # Load task data
        test_data = self.load_task_data(task_name)

        # Fine-tune model (if needed) or use zero-shot
        if self.config.get('fine_tune', True):
            self.fine_tune(task_name, train_data)

        # Evaluate
        predictions = []
        labels = []

        for example in test_data:
            pred = self.predict(example)
            predictions.append(pred)
            labels.append(example['label'])

        # Compute metrics
        metrics = self.compute_metrics(predictions, labels, task_name)

        return metrics
```

### Evaluation Modes

**1. Zero-Shot**:
- No fine-tuning
- Direct inference
- Tests pre-trained knowledge

**2. Few-Shot**:
- Fine-tune on small number of examples
- Tests adaptation ability
- Typical: 16, 32, 64 examples

**3. Full Fine-Tuning**:
- Fine-tune on full training set
- Standard benchmark evaluation
- Tests maximum task performance

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

**Location**: `src/evaluation/multiblimp_evaluator.py`

**Purpose**: Evaluate grammatical competence through minimal pairs

### What is BLiMP?

**BLiMP** (Benchmark of Linguistic Minimal Pairs): Tests whether models assign higher probability to grammatical sentences than ungrammatical ones.

**Minimal Pair Example**:
```
Grammatical:   लड़का घर जाता है।  (The boy goes home.)
Ungrammatical: लड़का घर जाता हैं। (Agreement error: plural verb with singular subject)
```

**Evaluation**: Does P(grammatical) > P(ungrammatical)?

### Hindi Linguistic Phenomena

**1. Agreement**:
- Gender agreement (noun-adjective)
- Number agreement (subject-verb)
- Person agreement

**2. Case Marking**:
- Nominative vs. Ergative
- Accusative/Dative markers
- Genitive constructions

**3. Word Order**:
- SOV (default)
- OSV (focus)
- Scrambling constraints

**4. Morphology**:
- Inflectional paradigms
- Derivational patterns
- Compounding

### Implementation (Conceptual)

```python
class MultiBLiMPEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_all_phenomena(self) -> Dict[str, Dict]:
        """Evaluate all linguistic phenomena"""
        results = {}

        phenomena = [
            'agreement',
            'case_marking',
            'word_order',
            'morphology'
        ]

        for phenomenon in phenomena:
            print(f"Evaluating {phenomenon}...")
            results[phenomenon] = self.evaluate_phenomenon(phenomenon)

        # Overall accuracy
        results['overall'] = self.compute_overall(results)

        return results

    def evaluate_phenomenon(self, phenomenon: str) -> Dict:
        """Evaluate a specific phenomenon"""
        # Load minimal pairs
        pairs = self.load_minimal_pairs(phenomenon)

        correct = 0
        total = len(pairs)

        for gram_sent, ungram_sent in pairs:
            # Compute probabilities
            p_gram = self.compute_probability(gram_sent)
            p_ungram = self.compute_probability(ungram_sent)

            # Check if grammatical > ungrammatical
            if p_gram > p_ungram:
                correct += 1

        accuracy = correct / total

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    def compute_probability(self, sentence: str) -> float:
        """Compute sentence probability"""
        tokens = self.tokenizer.encode(sentence)
        log_prob = 0.0

        self.model.eval()
        with torch.no_grad():
            for i in range(1, len(tokens)):
                # P(token_i | token_1, ..., token_{i-1})
                input_ids = torch.tensor([tokens[:i]])
                output = self.model(input_ids)
                logits = output.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                log_prob += torch.log(probs[tokens[i]]).item()

        return log_prob
```

### Example Phenomena

**Agreement Test**:
```python
minimal_pairs = [
    ("लड़का जाता है।", "लड़का जाता हैं।"),  # Singular vs. plural verb
    ("लड़की गई।", "लड़की गया।"),           # Feminine vs. masculine participle
    ("अच्छा लड़का", "अच्छा लड़की"),       # Gender mismatch
]
```

**Case Marking Test**:
```python
minimal_pairs = [
    ("राम ने खाना खाया।", "राम खाना खाया।"),  # Ergative required
    ("मुझे पानी चाहिए।", "मैं पानी चाहिए।"),   # Dative vs. nominative
]
```

### Example Results

```json
{
  "multiblimp": {
    "agreement": {
      "accuracy": 0.85,
      "correct": 170,
      "total": 200
    },
    "case_marking": {
      "accuracy": 0.78,
      "correct": 156,
      "total": 200
    },
    "word_order": {
      "accuracy": 0.72,
      "correct": 144,
      "total": 200
    },
    "overall": {
      "accuracy": 0.78
    }
  }
}
```

## 3. Morphological Probes

**Location**: `src/evaluation/morphological_probes.py`

**Purpose**: Test understanding of Hindi morphology

### Probing Tasks

**1. Paradigm Completion**:
Given inflected forms, predict missing forms

```
Given: लड़का (boy.NOM), लड़कों (boys.OBL)
Predict: लड़के (boy.OBL)
```

**2. Case Assignment**:
Predict correct case marker for context

```
Sentence: राम ___ किताब दी। (Ram gave the book to ___)
Options: को (ACC/DAT), ने (ERG), का (GEN), से (INS)
Answer: को
```

**3. Verbal Agreement**:
Predict correct verb form for subject

```
Subject: लड़कियाँ (girls)
Verb: जा___ (go___)
Options: ती हैं, ता है, ते हैं, ता हूँ
Answer: ती हैं (feminine plural)
```

**4. Compound Segmentation**:
Identify morpheme boundaries

```
Word: विश्वविद्यालय (university)
Segmentation: विश्व-विद्यालय (world-school)
```

### Implementation (Conceptual)

```python
class MorphologicalProbe:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_all_probes(self) -> Dict[str, Dict]:
        """Run all morphological probing tasks"""
        results = {}

        probes = [
            'paradigm_completion',
            'case_assignment',
            'verbal_agreement',
            'compound_segmentation'
        ]

        for probe_name in probes:
            print(f"Running {probe_name} probe...")
            results[probe_name] = self.run_probe(probe_name)

        return results

    def run_probe(self, probe_name: str) -> Dict:
        """Run a specific probe"""
        test_data = self.load_probe_data(probe_name)

        correct = 0
        total = len(test_data)

        for example in test_data:
            prediction = self.predict(example)
            if prediction == example['answer']:
                correct += 1

        accuracy = correct / total

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
```

### Example Results

```json
{
  "morphological_probes": {
    "paradigm_completion": {
      "accuracy": 0.81,
      "correct": 162,
      "total": 200
    },
    "case_assignment": {
      "accuracy": 0.74,
      "correct": 148,
      "total": 200
    },
    "verbal_agreement": {
      "accuracy": 0.88,
      "correct": 176,
      "total": 200
    },
    "compound_segmentation": {
      "accuracy": 0.67,
      "correct": 134,
      "total": 200
    }
  }
}
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
print(f"Morphological Average: {results['summary']['overall_scores']['morphological_avg']:.2%}")

# Results saved to: results/evaluation_{timestamp}/
```

## Interpreting Results

### Performance Benchmarks

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| IndicGLUE | <60% | 60-70% | 70-80% | >80% |
| MultiBLiMP | <70% | 70-80% | 80-90% | >90% |
| Morphological | <65% | 65-75% | 75-85% | >85% |

### What Good Performance Means

**High IndicGLUE Scores**:
- Model captures semantic knowledge
- Good at practical NLP tasks
- Effective representations

**High MultiBLiMP Scores**:
- Model understands Hindi grammar
- Sensitivity to syntactic violations
- Implicit linguistic knowledge

**High Morphological Scores**:
- Model handles inflectional morphology
- Paradigmatic knowledge
- Word internal structure

### Diagnostic Analysis

**If IndicGLUE is low but MultiBLiMP is high**:
- Model has linguistic knowledge but lacks semantic/world knowledge
- May need more diverse training data

**If MultiBLiMP is low but IndicGLUE is high**:
- Model relies on statistical patterns rather than linguistic structure
- May overfit to training distribution

**If Morphological is low**:
- Tokenization may be suboptimal
- Model not capturing word internal structure
- Need morphology-aware training

## Best Practices

1. **Multiple Metrics**: Don't rely on single evaluation
2. **Error Analysis**: Inspect failure cases
3. **Baseline Comparison**: Compare to random, majority, and prior work
4. **Significance Testing**: Use statistical tests for comparison
5. **Cross-Validation**: Multiple evaluation runs
6. **Qualitative Analysis**: Manual inspection of outputs

## Extending Evaluation

### Adding New Tasks

1. Create evaluator class in `src/evaluation/`
2. Implement `evaluate()` method
3. Add to `EvaluationManager`
4. Update result compilation

### Custom Probes

1. Design linguistic test
2. Create test data
3. Implement probing method
4. Add to `MorphologicalProbe`

## Related Documentation

- [Training Pipeline Documentation](05_TRAINING.md)
- [Model Architecture Documentation](04_MODELS.md)
- [Configuration Guide](07_CONFIGURATION.md)
- [API Reference](08_API_REFERENCE.md)
