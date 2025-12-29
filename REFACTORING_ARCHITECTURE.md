# IndicGLUE Evaluator - Refactored Architecture

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IndicGLUEEvaluator (Main)                        │
│                         2,563 lines                                  │
│                                                                       │
│  Responsibilities:                                                   │
│  • Model management (wrapping, device handling)                     │
│  • Dataset loading and split creation                               │
│  • High-level evaluation orchestration                              │
│  • Results aggregation and caching                                  │
│  • API compatibility layer                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Delegates to ▼
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                         │
        ▼                                                         ▼
┌─────────────────┐                                    ┌─────────────────┐
│  TaskRegistry   │◄───────────────────────────────────│ TaskDataExtract │
│    313 lines    │                                    │    354 lines    │
├─────────────────┤                                    ├─────────────────┤
│ • Task configs  │                                    │ • extract_text()│
│ • Label maps    │                                    │ • extract_label │
│ • Split remap   │                                    │ • extract_choice│
│ • Converters    │                                    │ • validate()    │
└─────────────────┘                                    └─────────────────┘
        │                                                         │
        │                                                         │
        └────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │DataLoaderFactory│
                   │    313 lines    │
                   ├─────────────────┤
                   │ • create_std()  │
                   │ • create_mc()   │
                   │ • Tokenization  │
                   │ • Collate fns   │
                   └─────────────────┘
                             │
                             │ Creates DataLoaders for ▼
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌─────────────────┐                    ┌─────────────────┐
│EvaluationStrat  │                    │FineTuningMgr    │
│    432 lines    │                    │    267 lines    │
├─────────────────┤                    ├─────────────────┤
│ • Classification│                    │ • fine_tune()   │
│ • MultipleChoice│                    │ • _train_epoch()│
│ • Perplexity    │                    │ • _validate()   │
│ • BinaryCandidat│                    │ • Early stopping│
└─────────────────┘                    └─────────────────┘
        │
        │ Predictions + Labels ▼
        │
        ▼
┌─────────────────┐
│ResultVisualizer │
│    375 lines    │
├─────────────────┤
│ • Confusion mtx │
│ • Class metrics │
│ • Training plots│
│ • Reports       │
└─────────────────┘
```

## Data Flow

### Evaluation Flow
```
Dataset
   │
   ├─► TaskDataExtractor.extract_text() ───┐
   ├─► TaskDataExtractor.extract_label() ──┤
   └─► TaskDataExtractor.extract_choices() ─┤
                                             │
                                             ▼
                                  DataLoaderFactory.create()
                                             │
                                             ▼
                                        DataLoader
                                             │
                                             ▼
                                   EvaluationStrategy.evaluate()
                                             │
                                             ▼
                                  {predictions, labels}
                                             │
                                             ▼
                                  ResultVisualizer.plot()
                                             │
                                             ▼
                                   Visualizations + Metrics
```

### Fine-Tuning Flow
```
Train Dataset + Val Dataset
         │
         ├─► DataLoaderFactory.create() ─► Train Loader
         └─► DataLoaderFactory.create() ─► Val Loader
                                              │
                                              ▼
                              FineTuningManager.fine_tune()
                                              │
                                              ├─► Training Loop
                                              ├─► Early Stopping
                                              └─► Best Model
                                                      │
                                                      ▼
                                              Fine-tuned Model
```

## Component Responsibilities

### 1. TaskRegistry
**Single Responsibility**: Task metadata management

**What it does**:
- Stores configuration for all 8 IndicGLUE tasks
- Provides label conversion functions
- Manages split remapping for corrupted datasets

**What it doesn't do**:
- ❌ Data extraction
- ❌ Model interaction
- ❌ Evaluation logic

### 2. TaskDataExtractor
**Single Responsibility**: Dataset field extraction

**What it does**:
- Extracts text from task-specific schemas
- Converts labels (string → int)
- Extracts choices for multiple-choice tasks
- Validates example structure

**What it doesn't do**:
- ❌ Tokenization
- ❌ DataLoader creation
- ❌ Batch collation

### 3. DataLoaderFactory
**Single Responsibility**: DataLoader creation

**What it does**:
- Creates standard DataLoaders (classification)
- Creates MC DataLoaders (proper shape [batch, choices, seq_len])
- Tokenizes text appropriately
- Implements collate functions

**What it doesn't do**:
- ❌ Data extraction (delegates to TaskDataExtractor)
- ❌ Model evaluation
- ❌ Training

### 4. EvaluationStrategy
**Single Responsibility**: Model evaluation

**What it does**:
- Runs inference on dataloaders/datasets
- Computes predictions from logits
- Returns {predictions, labels} dicts

**What it doesn't do**:
- ❌ Metric computation (returns raw predictions)
- ❌ Visualization
- ❌ Fine-tuning

### 5. FineTuningManager
**Single Responsibility**: Model fine-tuning

**What it does**:
- Runs training loop
- Implements early stopping
- Manages optimizer with parameter-specific decay
- Validates and tracks best model

**What it doesn't do**:
- ❌ DataLoader creation
- ❌ Model architecture changes
- ❌ Evaluation

### 6. ResultVisualizer
**Single Responsibility**: Result visualization

**What it does**:
- Plots confusion matrices
- Generates per-class metric charts
- Creates training history plots
- Produces text summary reports

**What it doesn't do**:
- ❌ Metric computation
- ❌ Data extraction
- ❌ Model evaluation

## Dependency Graph

```
TaskRegistry (no dependencies)
     │
     ├─────► TaskDataExtractor
     │            │
     │            ├─────► DataLoaderFactory
     │            │            │
     │            │            └─────► FineTuningManager
     │            │
     │            └─────► EvaluationStrategy
     │                         │
     │                         └─────► ResultVisualizer
     │
     └─────► (All used by IndicGLUEEvaluator)
```

**Dependency Direction**: Bottom-up (low-level → high-level)
**No circular dependencies**: Clean, acyclic graph

## Interface Design

### TaskRegistry
```python
class TaskRegistry:
    def get_task_config(task_name: str) -> TaskConfig
    def get_all_task_names() -> List[str]
    def get_label_converter(task_name: str) -> Optional[Callable]
    def get_split_remapping(task_name: str) -> Dict[str, str]
```

### TaskDataExtractor
```python
class TaskDataExtractor:
    def extract_text(example: Dict, task_name: str) -> str
    def extract_label(example: Dict, task_name: str) -> int
    def extract_choices(example: Dict, task_name: str) -> List[str]
    def validate_example(example: Dict, task_name: str) -> bool
```

### DataLoaderFactory
```python
class DataLoaderFactory:
    def create_standard_dataloader(...) -> DataLoader
    def create_multiple_choice_dataloader(...) -> DataLoader
```

### EvaluationStrategy
```python
class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(...) -> Dict[str, Any]

class ClassificationStrategy(EvaluationStrategy): ...
class MultipleChoiceStrategy(EvaluationStrategy): ...
class PerplexityStrategy(EvaluationStrategy): ...
```

### FineTuningManager
```python
class FineTuningManager:
    def fine_tune(model, train_loader, val_loader, task_name) -> Dict
    def _create_optimizer(model) -> Optimizer
    def _train_epoch(...) -> float
    def _validate(...) -> Dict[str, float]
```

### ResultVisualizer
```python
class ResultVisualizer:
    def plot_confusion_matrix(...) -> Figure
    def plot_per_class_metrics(...) -> Figure
    def plot_training_history(...) -> Figure
    def create_summary_report(...) -> str
```

## Testing Strategy

Each component has dedicated unit tests that verify:
1. Correct behavior with valid inputs
2. Proper error handling with invalid inputs
3. Edge cases (empty data, missing fields, etc.)
4. Integration with dependencies

**Test Isolation**: Each component can be tested independently using mocks for dependencies.

**Integration Tests**: Verify components work together correctly in realistic scenarios.

## Extensibility Points

### Adding a New Task
1. Add configuration to `TaskRegistry`
2. (Optional) Add field extraction logic to `TaskDataExtractor` if schema is unusual
3. Done! DataLoaderFactory and strategies handle the rest automatically

### Adding a New Evaluation Method
1. Create new class inheriting from `EvaluationStrategy`
2. Implement `evaluate()` method
3. Instantiate in `IndicGLUEEvaluator.__init__()`

### Adding a New Visualization
1. Add method to `ResultVisualizer`
2. Call from evaluation workflow

## Performance Characteristics

| Component | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| TaskRegistry | O(1) | O(n) | n = number of tasks |
| TaskDataExtractor | O(1) | O(1) | Per-example extraction |
| DataLoaderFactory | O(n) | O(b) | n = dataset size, b = batch size |
| EvaluationStrategy | O(n·m) | O(b) | n = examples, m = model inference |
| FineTuningManager | O(e·n·m) | O(b) | e = epochs |
| ResultVisualizer | O(c²) | O(c²) | c = number of classes |

**Overall**: Same performance characteristics as original monolithic code, but with better maintainability.

---

*Architecture follows SOLID principles with clear separation of concerns*
