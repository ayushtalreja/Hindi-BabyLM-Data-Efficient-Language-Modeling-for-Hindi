# Hindi BabyLM: Data-Efficient Language Modeling for Hindi

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Hub-FFD54F?style=flat)](https://huggingface.co/Ayush-Talreja/hindi-babylm)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayush-kumar-935703231/)

A comprehensive implementation of data-efficient language modeling for Hindi, designed as a BabyLM challenge adaptation for morphologically rich languages. This project trains transformer-based language models with developmentally plausible amounts of data (~10M and 100M tokens) and includes extensive evaluation frameworks for linguistic competence assessment.

## 🌟 Key Features

- **Causal and Masked language Model Architectures**: GPT-2 (Causal LM) and DeBERTa V2 (Masked LM with disentangled attention) with multiple size variants
- **Advanced Training Pipeline**: Multiple optimizers (AdamW, Adam, SGD, Adafactor), LR schedulers with warmup, mixed precision (FP16/BF16), gradient clipping, gradient accumulation
- **Comprehensive Tokenization**: 5 strategies including novel character-level approaches (SentencePiece Unigram/BPE, Wordpiece,Pure Character UACT, Character-Bigram HCBT)
- **MultiBLiMP Evaluation**: 5 Hindi syntactic phenomena (Subject-Verb & Subject-Predicate Agreement) with 1,447 minimal pairs from HuggingFace
- **IndicGLUE Benchmark**: 8 Hindi NLP tasks with modular evaluation framework and optional fine-tuning
- **Statistical Analysis**: Paired t-tests, Wilcoxon tests, effect sizes, bootstrap confidence intervals
- **Publication-Ready Figures**: 10+ plot types using ThesisPlotter with consistent styling
- **LaTeX Integration**: Automatic generation of thesis-ready tables and figures
- **Interactive Notebooks**: 2 comprehensive Jupyter notebooks for data exploration and results analysis
- **Experiment Tracking**: Automatic logging with Weights & Biases integration
- **IndicCorp V2 Integration**: Automated download from AI4Bharat/HuggingFace with streaming support
- **Multi-Source Corpus**: IndicCorp V2 (HuggingFace), Hindi Wikipedia (HuggingFace), IndicDialogue (movie/TV subtitles), children's literature
- **Advanced Quality Filtering**: Length-based, language detection (Devanagari ratio), deduplication (MinHash LSH)
- **Intelligent Data Mixing**: Configurable source ratios with token-level precision

## 📁 Project Structure

```
hindi-babylm/
├── data/
│   ├── raw/                          # Raw downloaded datasets
│   │   ├── indicdialogue.pkl         # Indicdialogue data
│   │   ├── indiccorp.pkl             # indiccorp data
│   │   ├── wikipedia.pkl             # wikipedia data
│   │   ├── children_stories.pkl      # children stories data
│   ├── splits/                       # Train/val/test splits (NEW PATH)
│   │   ├── train.pkl / train.txt     # Training data
│   │   ├── val.pkl / val.txt         # Validation data
│   │   ├── test.pkl / test.txt       # Test data
│   │   └── metadata.json             # Split metadata
│
├── src/
│   ├── data_processing/
│   │   ├── downloaders/              # Data source downloaders
│   │   │   ├── indiccorp_downloader.py   # IndicCorp V2 from HuggingFace
│   │   │   ├── wiki_downloader.py        # Wikipedia from HuggingFace
│   │   │   ├── indicdialogue_loader.py   # IndicDialogue movie/TV subtitles
│   │   │   └── base_downloader.py        # Abstract base class
│   │   ├── childrens_books.py        # Children's literature collection
│   │   ├── corpus_builder.py         # Main corpus orchestration
│   │   ├── text_cleaner.py           # Unicode normalization, cleaning
│   │   ├── quality_filter.py         # Length/language filtering
│   │   ├── deduplicator.py           # MinHash LSH deduplication
│   │   ├── data_mixer.py             # Multi-source data mixing
│   │   └── corpus_analyzer.py        # Statistics generation
│   │
│   ├── tokenization/
│   │   ├── tokenizer_factory.py      # Tokenizer creation hub (3 strategies)
│   │   ├── sentencepiece_tokenizer.py    # SentencePiece (Unigram/BPE modes, wordpiece)
│   │   ├── character_tokenizer.py        # Pure character-level (UACT)
│   │   ├── character_bigram_tokenizer.py # Character-bigram (HCBT)
│   │   ├── morphological_eval.py         # Morphological preservation metrics
│   │   └── tokenizer_comparison.py       # Benchmarking tool
│   │
│   ├── models/
│   │   ├── model_factory.py          # Model creation hub
│   │   ├── gpt_model.py              # GPT-2 causal LM (Tiny/Small/Medium)
│   │   ├── deberta_model.py          # DeBERTa V2 with disentangled attention
│   │   └── classification_models.py  # Classification adapters for downstream tasks
│   │
│   ├── training/
│   │   ├── trainer.py                # Enhanced trainer with callbacks
│   │   ├── data_loader.py            # PyTorch DataLoader utilities
│   │   └── (optimizer/scheduler integrated in trainer.py)
│   │
│   ├── evaluation/
│   │   ├── evaluation_manager.py     # Evaluation orchestration
│   │   ├── indicglue_evaluator.py    # 8 Hindi NLP tasks with modular architecture
│   │   ├── indicglue/                # IndicGLUE sub-components
│   │   │   ├── task_registry.py      # Task configurations
│   │   │   ├── fine_tuning_manager.py # Fine-tuning management
│   │   │   └── ...                   # Other modular components
│   │   ├── multiblimp_evaluator.py   # 5 syntactic phenomena (1,447 pairs from HF)
│   │   ├── evaluation_cache.py       # Result caching (30-day TTL)
│   │   ├── metrics_utils.py          # Bootstrap CI & statistical testing
│   │   └── comparative_analysis.py   # Cross-model comparison
│   │
│   ├── analysis/                     # Analysis & Visualization
│   │   ├── results_analyzer.py       # Statistical testing & LaTeX tables
│   │   └── visualization_utils.py    # ThesisPlotter - publication figures
│   │
│   └── utils/
│       ├── experiment_config.py      # Configuration management
│       ├── seed_manager.py           # Reproducibility utilities
│       ├── checkpoint_manager.py     # Model checkpoint handling
│       └── logger.py                 # Logging utilities
│
├── notebooks/                        # Interactive Analysis
│   ├── 01_data_exploration.ipynb     # 10-section corpus analysis
│   ├── 02_results_analysis.ipynb     # Multi-experiment comparison & stats
│   └── finetuning.ipynb              # Fine-tuning notebook (IndicBERT team, TF 1.15)
│
├── experiments/
│   ├── run_experiment.py             # Main experiment orchestrator
│
├── configs/                          # Configuration Files
│   ├── base_config.yaml              # Base configuration
│   ├── ablations                     # ablation studies
│       ├── data_mixing               # data source configs dir
│       ├── vocab_sizes               # different vocab sizes config
│       └── tokenizer                 # tokenizers config dir
│
├── docs/                             # Comprehensive Documentation
│   ├── 01_PROJECT_OVERVIEW.md        # Project architecture & phases
│   ├── 02_DATA_PROCESSING.md         # Data pipeline & IndicCorp
│   ├── 03_TOKENIZATION.md            # Tokenization strategies
│   ├── 04_MODELS.md                  # Model architectures (GPT-2, DeBERTa V2)
│   ├── 05_TRAINING.md                # Training pipeline
│   ├── 06_EVALUATION.md              # Evaluation frameworks (IndicGLUE, MultiBLiMP)
│   ├── 07_CONFIGURATION.md           # Configuration guide
│   ├── 08_ANALYSIS_AND_VISUALIZATION.md  # ResultsAnalyzer & ThesisPlotter
│   ├── 08b_TOKENIZER_EVALUATION.md   # Morphological evaluation & comparison
│   ├── 08c_JUPYTER_NOTEBOOKS.md      # Notebook usage guide
│
├── results/                          # Experiment Results
│   └── [experiment_name]/
│       ├── metadata.json
│       ├── config.yaml
│       ├── training_summary.json
│       ├── experiment.log
│       ├── evaluation_results.json
│       ├── evaluation_summary.csv
│       ├── tokenizer/                # Contains trained tokenizer and its metadata
│       └── models/                   # Contains best/final model/checkpoints as configured   
│
├── figures/                          # Generated Figures
│
├── tables/                           # LaTeX Tables
│
├── reports/                          # Markdown Reports
│
├── slurm_scripts/                    # HPC/LRZ Job Scripts
│   ├── README_LRZ.md                 # Complete LRZ setup guide
│   ├── QUICK_REFERENCE.md            # Command cheatsheet
│   ├── run_complete_pipeline.sh      # Full pipeline 
│   ├── run_data_processing.sh        # Data only 
│   ├── run_training.sh               # Training only
│   ├── run_evaluation.sh             # Evaluation only 
│   └── run_tiny_model.sh             # Quick test 
│
└── logs/                             # SLURM job logs (created automatically)

```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
- 16GB+ RAM recommended
- 50GB+ disk space for data and models

### Running on HPC Systems (LRZ, etc.)

**For LRZ users**, we provide ready-to-use SLURM scripts in `slurm_scripts/`. See the complete [LRZ Setup Guide](slurm_scripts/README_LRZ.md) for detailed instructions.

**Quick Start on LRZ:**

```bash
# 1. Setup (one-time)
module load python/3.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p logs

# 2. Update email in scripts
sed -i 's/your.email@tum.de/YOUR_EMAIL@tum.de/g' slurm_scripts/*.sh

# 3. Submit complete pipeline job
sbatch slurm_scripts/run_complete_pipeline.sh

# 4. Monitor job
squeue -u $USER
tail -f logs/pipeline_*.out
```

Available SLURM scripts:
- `run_complete_pipeline.sh` - Full pipeline (24h, 1 GPU)
- `run_data_processing.sh` - Data only (4h, CPU)
- `run_training.sh` - Training only (48h, 1 GPU)
- `run_evaluation.sh` - Evaluation only (8h, 1 GPU)
- `run_tiny_model.sh` - Quick test (12h, 1 GPU)

**Key Features:**
- ✅ GPU auto-detection and setup
- ✅ Module loading (Python, CUDA, cuDNN)
- ✅ Automatic logging to `logs/` directory
- ✅ Email notifications on completion
- ✅ Resource optimization for LRZ partitions
- ✅ Checkpoint resumption support

See [slurm_scripts/README_LRZ.md](slurm_scripts/README_LRZ.md) for troubleshooting and advanced usage.

### Installation (Local Machine)

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/hindi-babylm.git
cd hindi-babylm
```

2. **Set up environment (automated)**
```bash
chmod +x setup_env.sh
./setup_env.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Install project in development mode**
```bash
pip install -e .
```

### Basic Usage

#### 1. Complete Pipeline (Recommended for First Run)

```bash
# Run complete pipeline: data → training → evaluation
python main.py \
    --config configs/base_config.yaml \
    --experiment_name my_first_experiment
```

This single command will:
- Download and process data from all sources (IndicCorp, Wikipedia, children's books) # comment out --force-reprocess in slurm script to download data while running for the first time
- Create train/val/test splits in `data/splits/`
- Train a language model with your configuration
- Run comprehensive evaluation (IndicGLUE 8 tasks, MultiBLiMP 1,447 minimal pairs)
- Save all results to `results/my_first_experiment/`

#### 2. Stage-by-Stage Execution

**Data Processing Only:**
```bash
# Download IndicCorp Hindi (100K samples ≈ 6M tokens) - standalone
python src/data_processing/indiccorp_downloader.py \
    --output-dir data/raw \
    --num-samples 100000 \
    --format both

# Or build complete corpus with all sources via main pipeline
python main.py \
    --config configs/base_config.yaml \
    --stage data \
    --experiment_name data_processing_only
```

**Training Only** (requires existing data):
```bash
# Train Tiny model (50M params) - good for testing
python main.py \
    --config configs/base_config.yaml \
    --stage train \
    --experiment_name tiny_baseline # change model size parameter in base config
```

**Evaluation Only** (requires trained model):
```bash
# Evaluate on all benchmarks (IndicGLUE, MultiBLiMP)
python main.py \
    --config configs/base_config.yaml \
    --stage eval \
    --experiment_name tiny_baseline
```

#### 3. Advanced Options

**Resume Training from Checkpoint:**
```bash
python main.py \
    --config configs/base_config.yaml \
    --stage train \
    --experiment_name resumed_training \
    --resume results/previous_exp/models/checkpoint_epoch_5.pt
```

**Force Reprocess Data:**
```bash
# Useful when you've updated data sources or filtering parameters
python main.py \
    --config configs/base_config.yaml \
    --stage all \
    --experiment_name fresh_run \
    --force-reprocess
```

**Custom Random Seed:**
```bash
# Override config seed for reproducibility experiments
python main.py \
    --config configs/base_config.yaml \
    --experiment_name seed_experiment \
    --seed 42
```

**Specify Device:**
```bash
# Force CPU usage (useful for debugging)
python main.py \
    --config configs/base_config.yaml \
    --experiment_name cpu_run \
    --device cpu

# Force GPU usage
python main.py \
    --config configs/base_config.yaml \
    --experiment_name gpu_run \
    --device cuda
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [01_PROJECT_OVERVIEW.md](docs/01_PROJECT_OVERVIEW.md) | Project architecture, phases, and statistics |
| [02_DATA_PROCESSING.md](docs/02_DATA_PROCESSING.md) | Data pipeline, IndicCorp downloader, quality filtering |
| [03_TOKENIZATION.md](docs/03_TOKENIZATION.md) | Tokenization strategies and morphological analysis |
| [04_MODELS.md](docs/04_MODELS.md) | Model architectures (GPT-2, DeBERTa V2) |
| [05_TRAINING.md](docs/05_TRAINING.md) | Training pipeline with mixed precision |
| [06_EVALUATION.md](docs/06_EVALUATION.md) | MultiBLiMP (5 phenomena, 1,447 pairs), IndicGLUE (8 tasks) |
| [07_CONFIGURATION.md](docs/07_CONFIGURATION.md) | Complete configuration reference |
| [08_ANALYSIS_AND_VISUALIZATION.md](docs/08_ANALYSIS_AND_VISUALIZATION.md) | ResultsAnalyzer and ThesisPlotter API |
| [08b_TOKENIZER_EVALUATION.md](docs/08b_TOKENIZER_EVALUATION.md) | Morphological evaluation and tokenizer comparison |
| [08c_JUPYTER_NOTEBOOKS.md](docs/08c_JUPYTER_NOTEBOOKS.md) | Interactive analysis workflows (10 sections + 14 cells) |

## 🔬 Key Research Questions

This implementation explores:

1. **Data Efficiency**: Can models learn Hindi grammar with only 10M-100M words (vs billions in typical pretraining)?

2. **Tokenization**: Which tokenization method (Unigram, BPE, Wordpiece, Character-level, Character-Bigram) best preserves morphological information in Hindi?

3. **Model Architecture**: What architecture (GPT-style autoregressive, DeBERTa-style masked LM with disentangled attention) performs best with limited Hindi data?

4. **Vocab Size Impact**: What's the optimal vocab size for Hindi and does Hindi benefit from a larger vocab size, given its rich morphology?

5. **Linguistic Competence**: How well do data-efficient models capture syntactic agreement phenomena in Hindi?

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AI4Bharat** for the IndicCorp V2 dataset
- **BabyLM Challenge organizers** for inspiration
- **Technical University of Munich** for supporting this research
- **PyTorch** and **HuggingFace** teams for excellent frameworks

## 📧 Contact

**Ayush Kumar**
Technical University of Munich
Email: ayush1.kumar@tum.de
GitHub: [@ayushtalreja](https://github.com/ayushtalreja)
LinkedIn: [Ayush Kumar](https://www.linkedin.com/in/ayush-kumar-935703231/)

## 📖 Citation

If you use this code or data for research, please cite:

```bibtex
@mastersthesis{kumar2025hindi_babylm,
  title={Hindi BabyLM: Data-Efficient Language Modeling for Hindi},
  author={Kumar, Ayush},
  year={2026},
  school={Technical University of Munich},
  type={Bachelor's Thesis},
  note={Implementation includes data-efficient language modeling for Hindi with
        MultiBLiMP evaluation (5 phenomena, 1,447 minimal pairs from HuggingFace)
        and IndicGLUE benchmarks (8 tasks)}
}
```
**Built with ❤️ for Hindi NLP research**
