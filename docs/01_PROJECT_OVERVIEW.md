# Hindi BabyLM: Project Overview and Architecture

## Project Summary

The Hindi BabyLM project implements a comprehensive framework for training data-efficient language models for Hindi, a morphologically rich language. This project is part of the BabyLM challenge, which focuses on training language models with developmentally plausible amounts of data (approximately 10 million tokens).

## Motivation

Traditional language models require massive amounts of training data (billions to trillions of tokens). The BabyLM challenge explores whether models can achieve strong linguistic competence with significantly less data, similar to how children learn language. This project extends this concept to Hindi, investigating unique challenges posed by:

- **Morphological complexity**: Hindi has rich inflectional morphology
- **Script characteristics**: Devanagari script with complex orthography
- **Resource constraints**: Limited high-quality Hindi datasets compared to English
- **Evaluation challenges**: Need for Hindi-specific linguistic probes

## Key Research Questions

1. **Tokenization Strategy**: Which tokenization method (SentencePiece, WordPiece, BPE) best preserves morphological information in Hindi?
2. **Model Architecture**: What architecture (GPT-style autoregressive, BERT-style masked LM, or hybrid) performs best with limited Hindi data?
3. **Data Quality vs Quantity**: What types of text data are most valuable for learning Hindi linguistic competence?
4. **Curriculum Learning**: Can developmental progressions (simpler → complex structures) improve learning efficiency?

## System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  IndicCorp   │  │  Wikipedia   │  │ Children's Books│   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘   │
│         │                 │                    │             │
└─────────┼─────────────────┼────────────────────┼─────────────┘
          │                 │                    │
          └─────────────────┴────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   DATA PROCESSING                            │
│  • Text Cleaning & Normalization                            │
│  • Language Detection                                        │
│  • Quality Filtering (length, readability)                  │
│  • Deduplication (exact & fuzzy matching)                   │
│  • Token Limiting (~10M tokens)                             │
│  • Train/Val/Test Splitting (80/10/10)                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      TOKENIZATION                            │
│  ┌─────────────────┐ ┌─────────────┐ ┌──────────────┐      │
│  │ SentencePiece   │ │ WordPiece   │ │     BPE      │      │
│  │ (Unigram LM)    │ │ (BERT-style)│ │  (Byte-Pair) │      │
│  └─────────────────┘ └─────────────┘ └──────────────┘      │
│  • Vocabulary Size: 32,000 tokens                           │
│  • Morphological Preservation Analysis                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    MODEL ARCHITECTURE                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │   GPT-2     │  │    BERT      │  │  Hybrid Model   │    │
│  │ (Causal LM) │  │ (Masked LM)  │  │ (GPT + BERT)    │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
│  • Hidden Size: 768                                         │
│  • Layers: 12                                               │
│  • Attention Heads: 12                                      │
│  • Parameters: ~110M                                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                        TRAINING                              │
│  • AdamW Optimizer (lr: 3e-4)                               │
│  • Batch Size: 32                                           │
│  • Max Epochs: 10                                           │
│  • Gradient Clipping                                        │
│  • Learning Rate Scheduling                                 │
│  • Optional: Curriculum Learning                            │
│  • Weights & Biases Integration                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                       EVALUATION                             │
│  ┌────────────────┐ ┌───────────────┐ ┌─────────────────┐  │
│  │   IndicGLUE    │ │  MultiBLiMP   │ │ Morphological   │  │
│  │  (NLP Tasks)   │ │  (Syntax)     │ │    Probes       │  │
│  └────────────────┘ └───────────────┘ └─────────────────┘  │
│  • Classification, NER, QA                                  │
│  • Grammatical Acceptability                                │
│  • Case Marking, Agreement, Word Order                      │
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
│   │   ├── indiccorp_downloader.py     # IndicCorp dataset handler
│   │   ├── wiki_scraper.py             # Wikipedia scraper
│   │   ├── childrens_books.py          # Children's literature collection
│   │   ├── text_cleaner.py             # Text normalization
│   │   ├── quality_filter.py           # Quality filtering
│   │   ├── deduplicator.py             # Deduplication logic
│   │   ├── data_mixer.py               # Data mixing strategies
│   │   └── corpus_analyzer.py          # Corpus statistics
│   │
│   ├── tokenization/            # Tokenization experiments
│   │   ├── tokenizer_factory.py        # Factory for creating tokenizers
│   │   ├── sentencepiece_tokenizer.py  # SentencePiece implementation
│   │   ├── tokenizer_comparison.py     # Tokenizer benchmarking
│   │   └── morphological_eval.py       # Morphological analysis
│   │
│   ├── models/                  # Model architectures
│   │   ├── model_factory.py            # Factory for creating models
│   │   ├── gpt_model.py                # GPT-2 style model
│   │   ├── bert_model.py               # BERT style model
│   │   └── hybrid_model.py             # Hybrid architecture
│   │
│   ├── training/                # Training pipeline
│   │   ├── trainer.py                  # Main training loop
│   │   └── data_loader.py              # Data loading utilities
│   │
│   ├── evaluation/              # Evaluation framework
│   │   ├── evaluation_manager.py       # Evaluation orchestration
│   │   ├── indicglue_evaluator.py      # IndicGLUE benchmarks
│   │   ├── multiblimp_evaluator.py     # MultiBLiMP syntax tests
│   │   └── morphological_probes.py     # Morphological probing
│   │
│   └── utils/                   # Utility functions
│       ├── experiment_config.py        # Configuration management
│       └── results_analyzer.py         # Results analysis
│
├── data/                        # Data storage
│   ├── raw/                     # Raw downloaded data
│   └── processed/               # Processed datasets
│
├── tokenizers/                  # Trained tokenizers
├── models/                      # Model checkpoints
├── results/                     # Experiment results
├── notebooks/                   # Jupyter notebooks for analysis
└── docs/                        # Documentation (this directory)
```

## Core Components

### 1. Data Processing Pipeline (`src/data_processing/`)
Handles all data collection, cleaning, filtering, and preparation tasks.

**Key Classes:**
- `CorpusBuilder`: Orchestrates the entire data pipeline
- `QualityFilter`: Applies quality checks to text
- `TextDeduplicator`: Removes duplicate content
- `DataMixer`: Combines multiple data sources

### 2. Tokenization Module (`src/tokenization/`)
Implements and compares different tokenization strategies for Hindi.

**Key Classes:**
- `TokenizerFactory`: Creates tokenizers based on configuration
- `HindiSentencePieceTokenizer`: SentencePiece wrapper
- `TokenizerComparison`: Benchmarking tools

### 3. Model Architectures (`src/models/`)
Implements various transformer-based model architectures.

**Key Classes:**
- `ModelFactory`: Creates models based on configuration
- `HindiGPTModel`: GPT-2 style autoregressive model
- `HindiBERTModel`: BERT style masked language model
- `HybridGPTBERTModel`: Combined architecture

### 4. Training Pipeline (`src/training/`)
Manages model training with various optimization strategies.

**Key Classes:**
- `HindiLanguageModelTrainer`: Main training loop
- PyTorch DataLoader integration
- Wandb logging support

### 5. Evaluation Framework (`src/evaluation/`)
Comprehensive evaluation across multiple dimensions.

**Key Classes:**
- `EvaluationManager`: Orchestrates all evaluations
- `IndicGLUEEvaluator`: NLP task benchmarking
- `MultiBLiMPEvaluator`: Syntactic competence testing
- `MorphologicalProbe`: Morphological understanding tests

### 6. Configuration System (`src/utils/`)
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
1. Download IndicCorp Hindi dataset
2. Scrape Hindi Wikipedia articles
3. Collect children's literature
4. Clean and normalize text
5. Apply quality filters
6. Deduplicate corpus
7. Limit to ~10M tokens
8. Create train/val/test splits

**Output**: Processed text files ready for tokenization

### Stage 2: Tokenization
1. Load training text
2. Train tokenizer (SentencePiece/WordPiece/BPE)
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
4. Run morphological probes
5. Compile results and statistics
6. Generate evaluation report

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

## Next Steps

For detailed information on specific components, see:
- [Data Processing Documentation](02_DATA_PROCESSING.md)
- [Tokenization Documentation](03_TOKENIZATION.md)
- [Model Architecture Documentation](04_MODELS.md)
- [Training Pipeline Documentation](05_TRAINING.md)
- [Evaluation Framework Documentation](06_EVALUATION.md)
- [Configuration Guide](07_CONFIGURATION.md)
- [API Reference](08_API_REFERENCE.md)
- [Setup and Usage Guide](09_SETUP_AND_USAGE.md)
