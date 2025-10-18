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
│   │   └── bert_model.py               # BERT style model
│   │
│   ├── training/                # Training pipeline
│   │   ├── trainer.py                  # Training loop
│   │   └── data_loader.py              # Data loading utilities
│   │
│   ├── evaluation/              # Evaluation framework
│   │   ├── evaluation_manager.py       # Evaluation orchestration
│   │   ├── indicglue_evaluator.py      # IndicGLUE benchmarks
│   │   ├── multiblimp_evaluator.py     # MultiBLiMP syntax tests (14 phenomena)
│   │   └── morphological_probes.py     # Morphological probing (10 tasks, layer-wise)
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
│   ├── processed/               # Processed datasets
│   ├── splits/                  # Train/validation/test splits
│   └── corpus_statistics.json  # Corpus analysis results (Phase 2)
│
├── tokenizers/                  # Trained tokenizers
├── models/                      # Model checkpoints
├── results/                     # Experiment results
│   └── [experiment_name]/
│       ├── metadata.json               # Experiment metadata
│       ├── config.yaml                 # Configuration snapshot
│       ├── training_summary.json       # Training metrics
│       ├── evaluation_results.json     # Evaluation results
│       └── checkpoints/                # Model checkpoints
│
├── notebooks/                   # Jupyter notebooks (Phase 2)
│   ├── 01_data_exploration.ipynb       # Corpus analysis
│   └── 02_results_analysis.ipynb       # Results visualization
│
├── figures/                     # Generated figures (Phase 2)
│   ├── training_curves.png
│   ├── indicglue_comparison.png
│   ├── multiblimp_comparison.png
│   └── morphological_probes_comparison.png
│
├── tables/                      # LaTeX tables (Phase 2)
│   ├── indicglue_results.tex
│   ├── multiblimp_results.tex
│   └── probes_results.tex
│
├── reports/                     # Generated reports (Phase 2)
│   └── [experiment_name]_report.md
│
└── docs/                        # Documentation (this directory)
    ├── 01_PROJECT_OVERVIEW.md
    ├── 02_DATA_PROCESSING.md
    ├── 03_TOKENIZATION.md
    ├── 04_MODELS.md
    ├── 05_TRAINING.md
    ├── 06_EVALUATION.md
    ├── 07_CONFIGURATION.md
    ├── 08_ANALYSIS_AND_VISUALIZATION.md    # Phase 2
    ├── 09_JUPYTER_NOTEBOOKS.md             # Phase 2
    └── 10_THESIS_INTEGRATION.md            # Phase 2
```

## Core Components

### 1. Data Processing Pipeline (`src/data_processing/`)
Handles all data collection, cleaning, filtering, and preparation tasks.

**Key Classes:**
- `CorpusBuilder`: Orchestrates the entire data pipeline
- `QualityFilter`: Applies quality checks to text
- `TextDeduplicator`: Removes duplicate content
- `DataMixer`: Combines multiple data sources
- `CorpusAnalyzer`: Statistical analysis of corpus (Phase 2)

### 2. Tokenization Module (`src/tokenization/`)
Implements and compares different tokenization strategies for Hindi.

**Key Classes:**
- `TokenizerFactory`: Creates tokenizers based on configuration
- `HindiSentencePieceTokenizer`: SentencePiece wrapper
- `TokenizerComparison`: Benchmarking tools
- `MorphologicalEvaluator`: Morphological preservation analysis

### 3. Model Architectures (`src/models/`)
Implements various transformer-based model architectures with advanced features.

**Key Classes:**
- `ModelFactory`: Creates models based on configuration
- `HindiGPTModel`: GPT-2 style autoregressive model
- **`EnhancedGPT`** (Phase 1): Advanced GPT with configurable position encodings
- **`PositionEncodings`** (Phase 1): Multiple position encoding types
  - Sinusoidal (original Transformer)
  - Learned (GPT-2 style)
  - RoPE (Rotary Position Embedding)
  - ALiBi (Attention with Linear Biases)
  - Relative Position Bias (T5-style)
- `HindiBERTModel`: BERT style masked language model
- `HybridGPTBERTModel`: Combined architecture

**Model Sizes Available** (Phase 1):
- Tiny: 50M parameters (6 layers, 512 hidden)
- Small: 110M parameters (12 layers, 768 hidden)
- Medium: 350M parameters (24 layers, 1024 hidden)

### 4. Training Pipeline (`src/training/`)
Enhanced training pipeline with curriculum learning and advanced optimization.

**Key Classes:**
- **`HindiLanguageModelTrainer`** (Enhanced Phase 1): Main training loop with:
  - Multiple optimizer support (AdamW, Adam, SGD)
  - LR schedulers (Linear warmup, Cosine, Constant)
  - Mixed precision training (FP16/BF16)
  - Gradient accumulation
  - Early stopping
  - Comprehensive checkpointing
- **`CurriculumStrategies`** (Phase 1): 5 curriculum learning strategies
  - Morphological complexity-based
  - Length-based
  - Frequency-based
  - Combined multi-factor
  - Dynamic difficulty
- **`CurriculumScheduler`** (Phase 1): 5 progression schedules
  - Linear
  - Square root (sqrt)
  - Exponential
  - Step-wise
  - Performance-based
- `DataLoader`: Data loading utilities with curriculum support

### 5. Evaluation Framework (`src/evaluation/`)
Comprehensive multi-dimensional evaluation with detailed linguistic analysis.

**Key Classes:**
- `EvaluationManager`: Orchestrates all evaluations
- `IndicGLUEEvaluator`: NLP task benchmarking (6 tasks)
- **`MultiBLiMPEvaluator`** (Enhanced): Syntactic competence testing
  - **14 linguistic phenomena** (updated)
  - Agreement: number, person, gender, honorific
  - Case marking: ergative, accusative, dative
  - Structural: word order, negation, binding, control
  - **70+ minimal pairs** with perplexity-based evaluation
- **`MorphologicalProbe`** (Enhanced): Morphological understanding tests
  - **10 probe tasks** (updated)
  - Case, number, gender, tense, person
  - Aspect, mood, voice, honorific, definiteness
  - **Layer-wise probing** (all 12 layers + embedding)
  - Linear classifier methodology for interpretability

### 6. Analysis and Visualization (`src/analysis/`) - **Phase 2**
Publication-ready analysis tools for thesis integration.

**Key Classes:**
- **`ResultsAnalyzer`**: Statistical analysis and comparison
  - Multi-experiment loading
  - Statistical testing (t-test, Wilcoxon, effect size, bootstrap CI)
  - Training curve visualization
  - Evaluation comparison plots
  - **LaTeX table generation** for thesis
  - Markdown report generation
- **`ThesisPlotter`**: Publication-quality visualizations
  - Consistent thesis formatting
  - 10+ specialized plot types
  - Layer-wise probe visualization
  - Curriculum progression plots
  - High-resolution export (300 DPI)
  - Multiple format support (PNG, PDF, SVG)

### 7. Jupyter Notebooks (`notebooks/`) - **Phase 2**
Interactive data exploration and results analysis.

**Notebooks:**
- **`01_data_exploration.ipynb`**: Corpus analysis
  - Length distributions, character analysis
  - Word frequency, morphological markers
  - Data quality assessment
- **`02_results_analysis.ipynb`**: Results visualization
  - Training curve comparison
  - Evaluation metric analysis
  - Statistical significance testing
  - Thesis figure generation

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

## Project Phases

### Phase 1: Advanced Training and Evaluation (Completed)

**Objectives**: Enhance model architectures and training strategies for better Hindi language learning.

**Key Additions**:
1. **Enhanced Model Architectures**:
   - 5 position encoding types (Sinusoidal, Learned, RoPE, ALiBi, Relative)
   - 3 model size configurations (50M, 110M, 350M parameters)
   - Advanced features: Gradient checkpointing, RMS Norm, Flash Attention, SwiGLU

2. **Curriculum Learning Framework**:
   - 5 curriculum strategies (morphological, length, frequency, combined, dynamic)
   - 5 progression schedules (linear, root, exponential, step, performance-based)
   - Automatic difficulty scoring and sample ranking

3. **Enhanced Training Pipeline**:
   - Multiple optimizer options (AdamW, Adam, SGD)
   - 3 LR schedulers with warmup (Linear, Cosine, Constant)
   - Mixed precision training (FP16/BF16)
   - Gradient accumulation for larger effective batch sizes
   - Early stopping with configurable patience

4. **Comprehensive Evaluation**:
   - **MultiBLiMP**: Expanded to 14 linguistic phenomena (70+ minimal pairs)
   - **Morphological Probes**: 10 probe tasks with layer-wise analysis
   - Perplexity-based syntactic evaluation
   - Linear probing for morphological competence

**Files Added**:
- `src/models/enhanced_gpt.py` (526 lines)
- `src/models/position_encodings.py` (440 lines)
- `src/training/curriculum_strategies.py` (502 lines)
- `src/training/curriculum_scheduler.py` (485 lines)
- Enhanced `src/training/trainer.py` (586 lines)
- Enhanced `src/evaluation/multiblimp_evaluator.py` (474 lines)
- Enhanced `src/evaluation/morphological_probes.py` (668 lines)

### Phase 2: Analysis and Thesis Integration (Completed)

**Objectives**: Provide comprehensive analysis tools and thesis-ready outputs.

**Key Additions**:
1. **Results Analysis Framework**:
   - Multi-experiment loading and comparison
   - Statistical significance testing (t-test, Wilcoxon, effect size, bootstrap CI)
   - Training curve visualization
   - Evaluation comparison plots
   - LaTeX table generation for thesis
   - Markdown report generation

2. **Publication-Quality Visualization**:
   - ThesisPlotter with consistent formatting
   - 10+ specialized plot types
   - Layer-wise probe visualization
   - Curriculum progression plots
   - High-resolution export (300 DPI)
   - Multiple formats (PNG, PDF, SVG)

3. **Interactive Jupyter Notebooks**:
   - Data exploration notebook (corpus statistics, distributions, quality)
   - Results analysis notebook (comparisons, significance testing, figure generation)

4. **Thesis Integration Workflow**:
   - Automated LaTeX table generation
   - Figure generation with thesis formatting
   - Comprehensive experiment reports
   - Direct .tex and .pdf output for thesis inclusion

**Files Added**:
- `src/analysis/results_analyzer.py` (571 lines)
- `src/analysis/visualization_utils.py` (487 lines)
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_results_analysis.ipynb`

**Output Directories**:
- `figures/` - PNG/PDF figures ready for thesis
- `tables/` - LaTeX .tex tables
- `reports/` - Markdown experiment reports

### Implementation Statistics

**Total Lines of Code**:
- Phase 1: ~3,500 lines of production code
- Phase 2: ~2,000 lines of analysis code
- Documentation: ~6,000 lines across 10 markdown files

**Testing Coverage**:
- Unit tests for curriculum strategies
- Integration tests for evaluation pipelines
- Manual validation of all probe tasks

## Next Steps

### For Users

1. **Run Complete Pipeline**: Follow README.md for quickstart
2. **Explore Notebooks**: Use Jupyter notebooks for interactive analysis
3. **Generate Thesis Outputs**: Use ResultsAnalyzer for LaTeX tables and figures

### For Developers

1. **Add New Curriculum Strategies**: Extend `CurriculumStrategies` class
2. **Add New Position Encodings**: Implement in `position_encodings.py`
3. **Add New Probe Tasks**: Extend `MorphologicalProbe` class

### Detailed Documentation

For detailed information on specific components, see:
- [Data Processing Documentation](02_DATA_PROCESSING.md)
- [Tokenization Documentation](03_TOKENIZATION.md)
- [Model Architecture Documentation](04_MODELS.md) - **Updated with Phase 1**
- [Training Pipeline Documentation](05_TRAINING.md) - **Updated with Phase 1**
- [Evaluation Framework Documentation](06_EVALUATION.md) - **Updated with Phase 1**
- [Configuration Guide](07_CONFIGURATION.md)
- [Analysis and Visualization Documentation](08_ANALYSIS_AND_VISUALIZATION.md) - **Phase 2**
- [Jupyter Notebooks Documentation](09_JUPYTER_NOTEBOOKS.md) - **Phase 2**
- [Thesis Integration Guide](10_THESIS_INTEGRATION.md) - **Phase 2**

