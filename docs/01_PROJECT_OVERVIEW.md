# Hindi BabyLM: Project Overview and Architecture

## Project Summary

The Hindi BabyLM project implements a comprehensive framework for training data-efficient language models for Hindi, a morphologically rich language. This project is part of the BabyLM challenge, which focuses on training language models with developmentally plausible amounts of data (approximately 10 and 100 million tokens).

## Motivation

Traditional language models require massive amounts of training data (billions to trillions of tokens). The BabyLM challenge explores whether models can achieve strong linguistic competence with significantly less data, similar to how children learn language. This project extends this concept to Hindi, investigating unique challenges posed by:

- **Morphological complexity**: Hindi has rich inflectional morphology
- **Script characteristics**: Devanagari script with complex orthography
- **Resource constraints**: Limited high-quality Hindi datasets compared to English
- **Evaluation challenges**: Need for Hindi-specific linguistic probes

## Key Research Questions

1. **Tokenization Strategy**: Which tokenization method (Unigram, BPE, Wordpiece, Character-level, Character-Bigram) best preserves morphological information in Hindi?
2. **Model Architecture**: What architecture (GPT-style autoregressive, DeBERTa-style masked LM with disentangled attention) performs best with limited Hindi data?
3. **Data Quality vs Quantity**: What types of text data are most valuable for learning Hindi linguistic competence?
4. **Vocab Size impact**: What's the optimal vocab size for Hindi and does hindi benefits from a larger vocab size, given its rich morpholgy?  

## System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                          │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐  │
│  │IndicCorp │ │Wikipedia │ │IndicDialogue│ │Children's   │  │
│  │ (HF)     │ │  (HF)    │ │(Subtitles)│ │    Books     │  │
│  └────┬─────┘ └────┬─────┘ └─────┬─────┘ └──────┬───────┘  │
│       │            │              │               │          │
└───────┼────────────┼──────────────┼───────────────┼──────────┘
        │            │              │               │
        └────────────┴──────────────┴───────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   DATA PROCESSING                           │
│  • Text Cleaning & Normalization                            │
│  • Language Detection                                       │
│  • Quality Filtering (length, readability)                  │
│  • Deduplication (exact & fuzzy matching)                   │
│  • Token Limiting (~10M tokens)                             │
│  • Train/Val/Test Splitting (80/10/10) / User defined limits│                 
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      TOKENIZATION                            │
│  ┌─────────────────┐ ┌─────────────┐ ┌──────────────┐      │
│  │ SentencePiece   │ │  Character  │ │Char-Bigram   │      │
│  │ (Unigram/BPE)   │ │  (UACT)     │ │   (HCBT)     │      │
│  └─────────────────┘ └─────────────┘ └──────────────┘      │
│  • Vocabulary Size: 128-32K tokens (depends on strategy)   │
│  • Morphological Preservation Analysis                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    MODEL ARCHITECTURE                        │
│  ┌──────────────────┐  ┌─────────────────────────────┐     │
│  │     GPT-2        │  │        DeBERTa V2           │     │
│  │  (Causal LM)     │  │ (Masked LM + Disentangled)  │     │
│  └──────────────────┘  └─────────────────────────────┘     │
│  • Model Sizes: Tiny (50M), Small (110M), Medium (350M)    │
│  • Hidden Size: 768 (Small), 1024 (Medium)                 │
│  • Layers: 12 (Small), 24 (Medium)                         │
│  • Attention: Causal (GPT) / Disentangled (DeBERTa)        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                        TRAINING                              │
│  • AdamW Optimizer (lr: 3e-4)                               │
│  • Batch Size: 32                                           │
│  • Max Epochs: 10                                           │
│  • Gradient Clipping                                        │
│  • Learning Rate Scheduling                                 │
│  • Weights & Biases Integration                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                       EVALUATION                             │
│  ┌────────────────────────┐ ┌───────────────────────────┐  │
│  │     IndicGLUE          │ │      MultiBLiMP           │  │
│  │    (8 NLP Tasks)       │ │ (5 Phenomena, 1,447 Pairs)│  │
│  └────────────────────────┘ └───────────────────────────┘  │
│  • Classification, Sentiment, QA, NLI                       │
│  • Subject-Verb/Predicate Agreement (Number, Gender, Person)│
│  • Perplexity-based Grammatical Acceptability Testing      │
└─────────────────────────────────────────────────────────────┘
```

## Project Directory Structure

```
hindi-babylm/
├── main.py                      # Main entry point for pipeline
├── setup_env.sh                 # Environment setup script
├── requirements.txt             # Python dependencies
├── README.md                    # Quick start guide
│
├── configs/                     # Configuration files
│   └── base_config.yaml         # Base configuration
│
├── src/                         # Source code
│   ├── data_processing/         # Data collection & processing
│   │   ├── corpus_builder.py           # Main corpus building pipeline
│   │   ├── downloaders/
│   │   │   ├── indiccorp_downloader.py # IndicCorp dataset handler
│   │   │   ├── wiki_downloader.py      # Wikipedia downloader (HuggingFace)
│   │   │   ├── indicdialogue_loader.py # IndicDialogue movie subtitles
│   │   │   └── base_downloader.py      # Base class for downloaders
│   │   ├── childrens_books.py          # Children's literature collection
│   │   ├── text_cleaner.py             # Text normalization
│   │   ├── quality_filter.py           # Quality filtering
│   │   ├── deduplicator.py             # Deduplication logic
│   │   └── corpus_analyzer.py          # Corpus statistics
│   │
│   ├── tokenization/            # Tokenization experiments
│   │   ├── tokenizer_factory.py        # Factory for creating tokenizers
│   │   ├── sentencepiece_tokenizer.py  # SentencePiece implementation
│   │   ├── character_tokenizer.py      # Character-level tokenizer (UACT)
│   │   ├── character_bigram_tokenizer.py # Character-bigram tokenizer (HCBT)
│   │   ├── tokenizer_comparison.py     # Tokenizer benchmarking
│   │   └── morphological_eval.py       # Morphological analysis
│   │
│   ├── models/                  # Model architectures
│   │   ├── model_factory.py            # Factory for creating models
│   │   ├── gpt_model.py                # GPT-2 style model
│   │   ├── deberta_model.py            # DeBERTa V2 style model
│   │   └── classification_models.py    # Classification adapters
│   │
│   ├── training/                # Training pipeline
│   │   ├── trainer.py                  # Training loop
│   │   └── data_loader.py              # Data loading utilities
│   │
│   ├── evaluation/              # Evaluation framework
│   │   ├── evaluation_manager.py       # Evaluation orchestration
│   │   ├── indicglue_evaluator.py      # IndicGLUE benchmarks (8 tasks)
│   │   ├── multiblimp_evaluator.py     # MultiBLiMP syntax tests (5 phenomena, 1,447 pairs)
│   │   ├── indicglue/                  # IndicGLUE sub-components
│   │   │   ├── task_registry.py        # Task configurations
│   │   │   ├── fine_tuning_manager.py  # Fine-tuning management
│   │   │   └── ...                     # Other IndicGLUE components
│   │   ├── evaluation_cache.py         # Evaluation caching
│   │   ├── comparative_analysis.py     # Cross-model comparison
│   │   └── metrics_utils.py            # Metrics computation
│   │
│   ├── analysis/                # Results analysis tools (Phase 2)
│   │   ├── results_analyzer.py         # Statistical analysis & LaTeX tables
│   │   └── visualization_utils.py      # Publication-quality plotting
│   │
│   └── utils/                   # Utility functions
│       ├── experiment_config.py        # Configuration management
│       └── logging_utils.py            # Logging utilities
│
├── data/                        # Data storage
│   ├── raw/                     # Raw downloaded data
│   ├── splits/                  # Train/validation/test splits
│   └── corpus_statistics.json  # Corpus analysis results (Phase 2)
│
├── results/                     # Experiment results
│   └── [experiment_name]/
│       ├── tokenizers/                  # Trained tokenizers
│       ├── models/                      # Model checkpoints
│       ├── metadata.json               # Experiment metadata
│       ├── config.yaml                 # Configuration snapshot
│       ├── training_summary.json       # Training metrics
│       ├── evaluation_results.json     # Evaluation results
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb       # Corpus analysis
│   ├── 02_results_analysis.ipynb       # Results visualization
│   └── finetuning.ipynb                # Fine-tuning Nb from indicBert team (Broken / Deprecated Code)
│
├── figures/                     # Generated figures
│
├── tables/                      # LaTeX tables 
│
├── reports/                     # Generated reports
│
└── docs/                        # Documentation (this directory)
    ├── 01_PROJECT_OVERVIEW.md
    ├── 02_DATA_PROCESSING.md
    ├── 03_TOKENIZATION.md
    ├── 04_MODELS.md
    ├── 05_TRAINING.md
    ├── 06_EVALUATION.md
    ├── 07_CONFIGURATION.md
    ├── 08_ANALYSIS_AND_VISUALIZATION.md
    ├── 08b_TOKENIZER_EVALUATION.md
    ├── 08c_JUPYTER_NOTEBOOKS.md
```

## Core Components

### 1. Data Processing Pipeline (`src/data_processing/`)
Handles all data collection, cleaning, filtering, and preparation tasks from 4 data sources.

**Key Classes:**
- `CorpusBuilder`: Orchestrates the entire data pipeline
- `BaseDownloader`: Abstract base class for downloaders
- `IndicCorpDownloader`: HuggingFace IndicCorp dataset handler
- `WikiDownloader`: Wikipedia Hindi articles (HuggingFace)
- `IndicDialogueLoader`: Movie/TV subtitles (IndicDialogue dataset)
- `QualityFilter`: Applies quality checks to text (length, language detection)
- `TextDeduplicator`: Removes duplicate content (MinHash LSH algorithm)
- `CorpusAnalyzer`: Statistical analysis of corpus

### 2. Tokenization Module (`src/tokenization/`)
Implements and compares different tokenization strategies for Hindi.

**Key Classes:**
- `TokenizerFactory`: Creates tokenizers based on configuration
- `HindiSentencePieceTokenizer`: SentencePiece wrapper (implements BPE/Unigram/Wordpiece tokenizers)
- `DevanagariCharacterTokenizer`: Pure character-level tokenizer (UACT)
- `CharacterBigramTokenizer`: Hybrid character-bigram tokenizer (HCBT)
- `TokenizerComparison`: Benchmarking tools
- `MorphologicalEvaluator`: Morphological preservation analysis

### 3. Model Architectures (`src/models/`)
Implements transformer-based model architectures for Hindi language modeling.

**Key Classes:**
- `ModelFactory`: Creates models based on configuration
- `HindiGPTModel`: GPT-2 style autoregressive model
- `HindiDeBERTaModel`: DeBERTa V2 style model with disentangled attention
- `ClassificationModelForSequenceClassification`: Adapter for classification tasks
- `ClassificationModelForMultipleChoice`: Adapter for multiple-choice tasks

**Model Sizes Available**:

**GPT-2 Models** (Causal Language Modeling):
- Tiny: 50M parameters (6 layers, 512 hidden, 8 heads)
- Small: 110M parameters (12 layers, 768 hidden, 12 heads)
- Medium: 350M parameters (24 layers, 1024 hidden, 16 heads)

**DeBERTa Models** (Masked Language Modeling):
- Tiny: 22M parameters (6 layers, 384 hidden)
- Small: 86M parameters (12 layers, 768 hidden)
- Base: 86M parameters (12 layers, 768 hidden)
- Large: 304M parameters (24 layers, 1024 hidden)

### 4. Training Pipeline (`src/training/`)
Advanced training pipeline with mixed precision and optimization features.

**Key Classes:**
- **`HindiLanguageModelTrainer`**: Main training loop with:
  - Multiple optimizer support (AdamW, Adam, SGD, Adafactor)
  - LR schedulers (Cosine with warmup, Linear, Constant)
  - Mixed precision training (FP16/BF16)
  - Gradient accumulation
  - Early stopping
  - Comprehensive checkpointing
  - Weights & Biases integration
  - Memory optimization
- `DataLoader`: Data loading utilities for efficient batching

### 5. Evaluation Framework (`src/evaluation/`)
Comprehensive evaluation with NLP tasks and syntactic competence testing.

**Key Classes:**
- `EvaluationManager`: Orchestrates all evaluations
- `IndicGLUEEvaluator`: NLP task benchmarking (8 tasks)
  - Modular architecture with TaskRegistry, DataLoaderFactory, EvaluationStrategies
  - Optional fine-tuning with FineTuningManager
  - Evaluation caching (30-day TTL)
  - Bootstrap confidence intervals
- **`MultiBLiMPEvaluator`**: Syntactic competence testing
  - **5 linguistic phenomena** from HuggingFace dataset `jumelet/multiblimp`
  - Subject-Verb Agreement: Number (407 pairs), Gender (419 pairs), Person (412 pairs)
  - Subject-Predicate Agreement: Number (100 pairs), Gender (109 pairs)
  - **1,447 total minimal pairs** with perplexity-based evaluation
- `EvaluationCache`: Hash-based caching system
- `MetricsAggregator`: Bootstrap confidence intervals and statistical testing
- `ComparativeAnalyzer`: Cross-model comparison with visualizations

### 6. Analysis and Visualization (`src/analysis/`)
Publication-ready analysis tools.

**Key Classes:**
- **`ResultsAnalyzer`**: Statistical analysis and comparison
  - Multi-experiment loading
  - Statistical testing (t-test, Wilcoxon, effect size, bootstrap CI)
  - Training curve visualization
  - Evaluation comparison plots
  - **LaTeX table generation** for thesis
  - Markdown report generation

### 7. Jupyter Notebooks (`notebooks/`)
Interactive data exploration and results analysis.

**Notebooks:**
- **`01_data_exploration.ipynb`**: Comprehensive corpus analysis
  - Font configuration for Hindi/Devanagari display
  - 10 analysis sections: basic stats, distributions, character analysis, word-level analysis
  - Morphological analysis, linguistic phenomena detection
  - Data quality assessment, cross-source comparison
- **`02_results_analysis.ipynb`**: Experimental results visualization
  - Multi-experiment loading and comparison
  - Training curve and dynamics analysis (4-panel plots)
  - IndicGLUE and MultiBLiMP detailed analysis
  - Statistical significance testing (t-test, Wilcoxon, effect size)
  - LaTeX table generation for thesis
  - Confusion matrices and comparative visualizations
- **`finetuning.ipynb`**: Fine-tuning Notebook from indicbert team to fine tune the provided indicbert model on their HF and github repositories. However, the code is written in **TF 1.15** and is incompatible with current **TF 2.x** versions. 
### 8. Configuration System (`src/utils/`)
Flexible configuration management for experiments.

**Key Classes:**
- `ExperimentConfig`: Dataclass for experiment configuration
- `ExperimentManager`: Creates experiment variations

## Design Principles

### 1. Modularity
Each component is self-contained and can be used independently or as part of the full pipeline.

### 2. Configurability
All hyperparameters and settings are specified in YAML configuration files, enabling easy experimentation.

### 3. Reproducibility
- Random seeds are fixed
- All configurations are saved with results
- Git commits are tracked for each experiment

### 4. Scalability
- Efficient data processing with streaming
- GPU acceleration support
- Checkpoint saving for long training runs

### 5. Observability
- Comprehensive logging at each stage
- Weights & Biases integration for training monitoring
- Detailed evaluation reports

## Technology Stack

### Core Libraries
- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: Pre-built model architectures
- **SentencePiece**: Tokenization
- **Tokenizers (HuggingFace)**: Alternative tokenization methods

### Data Processing
- **Datasets (HuggingFace)**: Dataset management
- **BeautifulSoup**: Web scraping
- **langdetect**: Language identification
- **NumPy/Pandas**: Data manipulation

### Experiment Tracking
- **Weights & Biases**: Experiment tracking and visualization
- **YAML**: Configuration files

### Development Tools
- **Python 3.8+**: Programming language
- **Git**: Version control
- **Virtual Environments**: Dependency isolation

## Workflow: From Data to Evaluation

### Stage 1: Data Collection and Processing
1. Download IndicCorp Hindi dataset (HuggingFace)
2. Download Hindi Wikipedia articles (HuggingFace)
3. Download IndicDialogue movie/TV subtitles
4. Collect children's literature
5. Clean and normalize text (Unicode normalization)
6. Apply quality filters (length, language detection, word count)
7. Deduplicate corpus (MinHash LSH with 0.8 similarity threshold)
8. Create balanced train/val/test splits (maintain source ratios) 
9. Limit to target token counts (configurable per split)

**Output**: Processed splits (train.pkl, val.pkl, test.pkl) ready for tokenization

### Stage 2: Tokenization
1. Load training text
2. Train tokenizer (Unigram/WordPiece/BPE)
3. Evaluate morphological preservation
4. Save trained tokenizer

**Output**: Trained tokenizer model

### Stage 3: Model Training
1. Load processed data and tokenizer
2. Create model architecture
3. Initialize optimizer and scheduler
4. Training loop:
   - Forward pass
   - Compute loss
   - Backward pass
   - Update weights
   - Log metrics
   - Save checkpoints
5. Save final model

**Output**: Trained language model

### Stage 4: Evaluation
1. Load trained model and tokenizer
2. Run IndicGLUE benchmarks
3. Run MultiBLiMP syntax tests
4. Compile results and statistics
5. Generate evaluation report

**Output**: Comprehensive evaluation results

## Running the Complete Pipeline

```bash
# Run all stages
python main.py --config configs/base_config.yaml --stage all --experiment_name hindi_babylm_baseline

# Run individual stages
python main.py --config configs/base_config.yaml --stage data --experiment_name my_experiment
python main.py --config configs/base_config.yaml --stage train --experiment_name my_experiment
python main.py --config configs/base_config.yaml --stage eval --experiment_name my_experiment
```

## Extending the Project

### Adding a New Data Source
1. Create a new module in `src/data_processing/`
2. Implement a collection function
3. Add to `CorpusBuilder.collect_all_data()`

### Adding a New Tokenizer
1. Create a new tokenizer class in `src/tokenization/`
2. Add to `TokenizerFactory`
3. Update configuration options

### Adding a New Model Architecture
1. Create a new model class in `src/models/`
2. Add to `ModelFactory`
3. Update configuration options

### Adding a New Evaluation
1. Create a new evaluator in `src/evaluation/`
2. Add to `EvaluationManager`
3. Update result compilation

## Project Phases

### Phase 1: Core Implementation (Completed)

**Objectives**: Build comprehensive framework for data-efficient Hindi language modeling.

**Key Implementations**:
1. **Data Processing Pipeline**:
   - 4 data sources: IndicCorp (HuggingFace), Wikipedia, IndicDialogue, Children's Books
   - BaseDownloader abstract class for consistent interfaces
   - Quality filtering, deduplication (MinHash LSH), balanced splitting
   - Corpus analysis and statistics

2. **Tokenization Strategies**:
   - SentencePiece (Unigram LM and BPE modes)
   - Pure Character tokenizer (UACT) for maximum morphological preservation
   - Character-Bigram tokenizer (HCBT) for balanced compression
   - Comprehensive morphological evaluation framework
   - Tokenizer comparison tools

3. **Model Architectures**:
   - GPT-2 models (Tiny/Small/Medium: 50M/110M/350M parameters)
   - DeBERTa V2 models (Tiny/Small/Base/Large: 22M/86M/86M/304M parameters)
   - Classification adapters for downstream tasks
   - ModelFactory for easy instantiation

4. **Training Pipeline**:
   - HindiLanguageModelTrainer with multiple optimizers (AdamW, Adam, SGD, Adafactor)
   - LR schedulers with warmup (Cosine, Linear, Constant)
   - Mixed precision training (FP16/BF16)
   - Gradient accumulation, early stopping, checkpointing
   - Weights & Biases integration

5. **Evaluation Framework**:
   - **IndicGLUE**: 8 Hindi NLP tasks with modular architecture
     - TaskRegistry, DataLoaderFactory, EvaluationStrategies
     - Optional fine-tuning with FineTuningManager
     - Zero-shot and fine-tuned evaluation modes
   - **MultiBLiMP**: 5 linguistic phenomena, 1,447 minimal pairs from HuggingFace
     - Perplexity-based grammatical acceptability testing
     - Subject-Verb and Subject-Predicate agreement
   - Evaluation caching (30-day TTL)
   - Bootstrap confidence intervals (1000 samples)
   - Comparative analysis tools

### Phase 2: Analysis and Thesis Integration (Completed)

**Objectives**: Provide comprehensive analysis tools and thesis-ready outputs.

**Key Implementations**:
1. **Results Analysis Framework**:
   - Multi-experiment loading and comparison (ResultsAnalyzer)
   - Statistical significance testing (paired t-test, Wilcoxon, Cohen's d effect size, bootstrap CI)
   - Training curve visualization (loss, perplexity, learning rate)
   - Evaluation comparison plots (IndicGLUE, MultiBLiMP)
   - LaTeX table generation for thesis (booktabs format, bold best values)
   - Markdown report generation (experiment summaries)

2. **Publication-Quality Visualization**:
   - ThesisPlotter with consistent formatting (matplotlib-based)
   - 10+ specialized plot types (training curves, comparison bars, confusion matrices)
   - High-resolution export (300 DPI)
   - Multiple formats (PNG, PDF, SVG)
   - Thesis-ready figures with proper fonts and sizing

3. **Interactive Jupyter Notebooks**:
   - **`01_data_exploration.ipynb`**: 10-section corpus analysis
     - Font configuration for Devanagari display
     - Basic statistics, distributions, character/word analysis
     - Morphological patterns, linguistic phenomena detection
     - Data quality assessment, cross-source comparison
   - **`02_results_analysis.ipynb`**: Comprehensive results visualization
     - Multi-experiment comparison with color-coded tables
     - 4-panel training dynamics analysis
     - IndicGLUE per-task comparison with confidence intervals
     - MultiBLiMP phenomenon-level analysis
     - Statistical significance testing with p-values
     - Confusion matrices for classification tasks
     - LaTeX table generation (`.tex` files)

4. **Thesis Integration Workflow**:
   - Automated LaTeX table generation (IndicGLUE, MultiBLiMP, comparison tables)
   - Figure generation with thesis formatting (consistent fonts, sizes, DPI)
   - Comprehensive experiment reports (markdown format)
   - Direct `.tex` output for thesis `\input{}` commands

**Output Directories Created**:
- `figures/` - PNG/PDF/SVG figures ready for thesis (300 DPI)
- `tables/` - LaTeX `.tex` tables (IndicGLUE, MultiBLiMP, comparisons)
- `reports/` - Markdown experiment reports (one per experiment)

**Key Components**:
- 4 data downloaders with BaseDownloader abstract class
- 5 tokenizer implementations (SentencePiece, Character, Character-Bigram)
- 2 model architectures (GPT-2, DeBERTa V2) with multiple size configs
- 8 IndicGLUE tasks with modular evaluation framework
- 1,447 MultiBLiMP minimal pairs from HuggingFace dataset
- 10+ analysis sections in data exploration notebook
- 14 analysis cells in results notebook

## Next Steps

### For Users

1. **Run Complete Pipeline**: Follow README.md for quickstart
2. **Explore Notebooks**: Use Jupyter notebooks for interactive analysis
3. **Generate Thesis Outputs**: Use ResultsAnalyzer for LaTeX tables and figures

### For Developers

1. **Add New Data Sources**: Create downloader class extending `BaseDownloader`
2. **Add New Tokenizers**: Implement in `src/tokenization/` and register in `TokenizerFactory`
3. **Add New Model Architectures**: Implement in `src/models/` and register in `ModelFactory`
4. **Add New Evaluation Tasks**: Add task config to `TaskRegistry` in IndicGLUE framework

### Detailed Documentation

For detailed information on specific components, see:
- [Data Processing Documentation](02_DATA_PROCESSING.md) - Corpus building and data sources
- [Tokenization Documentation](03_TOKENIZATION.md) - Tokenizer implementations and strategies
- [Model Architecture Documentation](04_MODELS.md) - GPT-2 and DeBERTa model details
- [Training Pipeline Documentation](05_TRAINING.md) - Training loop and optimization
- [Evaluation Framework Documentation](06_EVALUATION.md) - IndicGLUE and MultiBLiMP details
- [Configuration Guide](07_CONFIGURATION.md) - YAML configuration reference
- [Analysis and Visualization Documentation](08_ANALYSIS_AND_VISUALIZATION.md) - ResultsAnalyzer and ThesisPlotter
- [Tokenizer Evaluation Documentation](08b_TOKENIZER_EVALUATION.md) - Morphological evaluation and comparison
- [Jupyter Notebooks Documentation](08c_JUPYTER_NOTEBOOKS.md) - Interactive analysis notebooks

