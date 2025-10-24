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

### Hindi Linguistic Phenomena (14 Categories)

The actual implementation tests 14 specific syntactic phenomena relevant to Hindi:

**Agreement Phenomena**:
1. **subject_verb_agreement_number**: Singular vs. plural subject-verb agreement
2. **subject_verb_agreement_person**: First, second, third person agreement
3. **subject_verb_agreement_gender**: Masculine vs. feminine agreement with verbs
4. **gender_agreement_adjective**: Adjective-noun gender agreement
5. **number_agreement**: General number agreement across constituents
6. **honorific_agreement**: Honorific forms (आप vs. तुम vs. तू)

**Case Marking Phenomena**:
7. **case_marking_ergative**: Ergative case marker (ने) in perfective transitive contexts
8. **case_marking_accusative**: Accusative/dative marker (को) with specific/animate objects
9. **case_marking_dative**: Dative case marker (को) with experiencer constructions

**Structural Phenomena**:
10. **word_order**: SOV vs. other word orders and scrambling constraints
11. **negation**: Placement and form of negation markers
12. **binding**: Pronominal and reflexive binding
13. **control**: Control structures in infinitival complements

**Additional Morphosyntax**:
14. **gender_agreement_verb**: Gender agreement in past tense verb forms

### Implementation

```python
class MultiBLiMPEvaluator:
    """
    Evaluates Hindi models on 14 syntactic phenomena using minimal pairs.

    Location: src/evaluation/multiblimp_evaluator.py:17
    """

    def __init__(self, model, tokenizer, config: Dict = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}

        # 14 phenomena tested
        self.phenomena = [
            'subject_verb_agreement_number',
            'subject_verb_agreement_person',
            'subject_verb_agreement_gender',
            'case_marking_ergative',
            'case_marking_accusative',
            'case_marking_dative',
            'word_order',
            'gender_agreement_adjective',
            'gender_agreement_verb',
            'number_agreement',
            'honorific_agreement',
            'negation',
            'binding',
            'control'
        ]

        # Create comprehensive minimal pairs (70+ pairs total)
        self.minimal_pairs = self._create_comprehensive_minimal_pairs()

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def evaluate_all_phenomena(self) -> Dict[str, Dict]:
        """
        Evaluate all 14 linguistic phenomena.

        Returns:
            Dict with per-phenomenon results and overall statistics
        """
        print("Starting MultiBLiMP evaluation on 14 phenomena...")

        results = {}
        total_correct = 0
        total_pairs = 0

        for phenomenon in self.phenomena:
            print(f"  Evaluating {phenomenon}...")
            phenomenon_results = self.evaluate_phenomenon(phenomenon)
            results[phenomenon] = phenomenon_results

            total_correct += phenomenon_results['correct']
            total_pairs += phenomenon_results['total']

        # Overall accuracy
        results['overall'] = {
            'accuracy': total_correct / total_pairs if total_pairs > 0 else 0.0,
            'correct': total_correct,
            'total': total_pairs
        }

        print(f"\nOverall MultiBLiMP Accuracy: {results['overall']['accuracy']:.2%}")

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

    def _create_comprehensive_minimal_pairs(self) -> Dict[str, List[Dict]]:
        """
        Create 70+ minimal pairs covering all 14 phenomena.

        Returns:
            Dict mapping phenomenon name to list of {'good': str, 'bad': str} pairs
        """
        pairs = {
            'subject_verb_agreement_number': [
                {'good': 'लड़का खाता है', 'bad': 'लड़का खाते हैं'},
                {'good': 'लड़के खाते हैं', 'bad': 'लड़के खाता है'},
                {'good': 'लड़की जाती है', 'bad': 'लड़की जाती हैं'},
                {'good': 'लड़कियाँ जाती हैं', 'bad': 'लड़कियाँ जाती है'},
                {'good': 'मैं खाता हूँ', 'bad': 'मैं खाता है'},
            ],

            'subject_verb_agreement_gender': [
                {'good': 'लड़का गया', 'bad': 'लड़का गई'},
                {'good': 'लड़की गई', 'bad': 'लड़की गया'},
                {'good': 'औरत आई', 'bad': 'औरत आया'},
                {'good': 'आदमी आया', 'bad': 'आदमी आई'},
            ],

            'case_marking_ergative': [
                {'good': 'राम ने खाना खाया', 'bad': 'राम खाना खाया'},
                {'good': 'सीता ने किताब पढ़ी', 'bad': 'सीता किताब पढ़ी'},
                {'good': 'लड़के ने पानी पिया', 'bad': 'लड़के पानी पिया'},
                {'good': 'मैंने काम किया', 'bad': 'मैं काम किया'},
            ],

            'case_marking_accusative': [
                {'good': 'मैंने राम को देखा', 'bad': 'मैंने राम देखा'},
                {'good': 'उसने मुझको बुलाया', 'bad': 'उसने मैं बुलाया'},
                {'good': 'सीता को बुलाओ', 'bad': 'सीता बुलाओ'},
            ],

            # ... (70+ pairs total across all 14 phenomena)
        }

        return pairs
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

**Ergative Case Marking**:
```python
{'good': 'राम ने खाना खाया', 'bad': 'राम खाना खाया'}  # Ergative required in perfective transitive
{'good': 'मैंने काम किया', 'bad': 'मैं काम किया'}    # First person ergative
```

**Accusative Case Marking**:
```python
{'good': 'मैंने राम को देखा', 'bad': 'मैंने राम देखा'}  # Accusative marker with animate object
{'good': 'सीता को बुलाओ', 'bad': 'सीता बुलाओ'}        # Accusative in imperative
```

**Word Order**:
```python
{'good': 'राम घर जाता है', 'bad': 'जाता है राम घर'}    # SOV vs. VSO
{'good': 'मैं किताब पढ़ता हूँ', 'bad': 'पढ़ता हूँ मैं किताब'}  # SOV vs. VOS
```

**Honorific Agreement**:
```python
{'good': 'आप जाते हैं', 'bad': 'आप जाता है'}          # आप requires plural verb
{'good': 'तुम जाते हो', 'bad': 'तुम जाता है'}         # तुम requires specific form
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
    "case_marking_ergative": {
      "accuracy": 0.76,
      "correct": 19,
      "total": 25,
      "avg_perplexity_diff": 7.3,
      "std_perplexity_diff": 2.5
    },
    "case_marking_accusative": {
      "accuracy": 0.72,
      "correct": 14,
      "total": 20,
      "avg_perplexity_diff": 6.2,
      "std_perplexity_diff": 1.9
    },
    "case_marking_dative": {
      "accuracy": 0.70,
      "correct": 14,
      "total": 20,
      "avg_perplexity_diff": 5.8,
      "std_perplexity_diff": 2.3
    },
    "word_order": {
      "accuracy": 0.68,
      "correct": 13,
      "total": 19,
      "avg_perplexity_diff": 4.5,
      "std_perplexity_diff": 1.7
    },
    "gender_agreement_adjective": {
      "accuracy": 0.84,
      "correct": 16,
      "total": 19,
      "avg_perplexity_diff": 9.2,
      "std_perplexity_diff": 2.4
    },
    "gender_agreement_verb": {
      "accuracy": 0.81,
      "correct": 17,
      "total": 21,
      "avg_perplexity_diff": 8.1,
      "std_perplexity_diff": 2.0
    },
    "number_agreement": {
      "accuracy": 0.86,
      "correct": 18,
      "total": 21,
      "avg_perplexity_diff": 10.3,
      "std_perplexity_diff": 2.6
    },
    "honorific_agreement": {
      "accuracy": 0.79,
      "correct": 15,
      "total": 19,
      "avg_perplexity_diff": 7.8,
      "std_perplexity_diff": 2.2
    },
    "negation": {
      "accuracy": 0.75,
      "correct": 12,
      "total": 16,
      "avg_perplexity_diff": 6.5,
      "std_perplexity_diff": 1.8
    },
    "binding": {
      "accuracy": 0.65,
      "correct": 11,
      "total": 17,
      "avg_perplexity_diff": 3.9,
      "std_perplexity_diff": 1.5
    },
    "control": {
      "accuracy": 0.63,
      "correct": 10,
      "total": 16,
      "avg_perplexity_diff": 3.2,
      "std_perplexity_diff": 1.4
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

# Results saved to: results/evaluation_{timestamp}/
```

## Interpreting Results

### Performance Benchmarks

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| IndicGLUE | <60% | 60-70% | 70-80% | >80% |
| MultiBLiMP | <70% | 70-80% | 80-90% | >90% |

### What Good Performance Means

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
