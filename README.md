# Hindi BabyLM: Data-Efficient Language Modeling for Hindi

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A comprehensive implementation of data-efficient language modeling for Hindi, designed as a BabyLM challenge adaptation for morphologically rich languages. This project trains transformer-based language models with developmentally plausible amounts of data (~10M tokens) and includes extensive evaluation frameworks for linguistic competence assessment.

## 🌟 Key Features

- **Enhanced GPT Architecture**: 3 model sizes (Tiny: 50M, Small: 110M, Medium: 350M parameters)
- **Curriculum Learning**: 5 training strategies × 5 scheduling approaches (25 combinations)
- **Advanced Training Pipeline**: Multiple optimizers (AdamW, Adam, SGD), LR schedulers, mixed precision (FP16/BF16)
- **Comprehensive Tokenization**: SentencePiece, WordPiece, BPE with morphological preservation analysis
- **MultiBLiMP Evaluation**: 14 Hindi linguistic phenomena with 70+ minimal pairs
- **Morphological Probes**: 10 probe tasks for layer-wise linguistic feature analysis
- **Statistical Analysis**: Paired t-tests, Wilcoxon tests, effect sizes, bootstrap confidence intervals
- **Publication-Ready Figures**: 10+ plot types using ThesisPlotter with consistent styling
- **LaTeX Integration**: Automatic generation of thesis-ready tables and figures
- **Interactive Notebooks**: 2 comprehensive Jupyter notebooks for data exploration and results analysis
- **Experiment Tracking**: Automatic logging with Weights & Biases integration
- **IndicCorp V2 Integration**: Automated download from AI4Bharat/HuggingFace with streaming support
- **Multi-Source Corpus**: IndicCorp, Hindi Wikipedia, children's literature
- **Advanced Quality Filtering**: Length-based, language detection (Devanagari ratio), deduplication (MinHash LSH)
- **Intelligent Data Mixing**: Configurable source ratios with token-level precision

## 📁 Project Structure

```
hindi-babylm/
├── data/
│   ├── raw/                          # Raw downloaded datasets
│   │   ├── indiccorp_hindi.txt       # IndicCorp Hindi corpus
│   │   ├── indiccorp_hindi.pkl       # Pickled format
│   │   ├── indiccorp_metadata.json   # Dataset metadata
│   │   └── raw_corpus.pkl            # Combined raw data
│   ├── splits/                       # Train/val/test splits (NEW PATH)
│   │   ├── train.pkl / train.txt     # Training data
│   │   ├── val.pkl / val.txt         # Validation data
│   │   ├── test.pkl / test.txt       # Test data
│   │   └── metadata.json             # Split metadata
│   └── tokenized/                    # Tokenized datasets
│
├── src/
│   ├── data_processing/
│   │   ├── indiccorp_downloader.py   # IndicCorp V2 downloader (IMPLEMENTED)
│   │   ├── wiki_scraper.py           # Wikipedia scraper
│   │   ├── childrens_books.py        # Children's literature
│   │   ├── corpus_builder.py         # Main corpus orchestration
│   │   ├── text_cleaner.py           # Unicode normalization, cleaning
│   │   ├── quality_filter.py         # Length/language filtering
│   │   ├── deduplicator.py           # MinHash LSH deduplication
│   │   ├── data_mixer.py             # Multi-source data mixing
│   │   └── corpus_analyzer.py        # Statistics generation
│   │
│   ├── tokenization/
│   │   ├── tokenizer_factory.py      # Tokenizer creation hub
│   │   ├── sentencepiece_trainer.py  # SentencePiece training
│   │   ├── wordpiece_trainer.py      # WordPiece training
│   │   ├── bpe_trainer.py            # BPE training
│   │   └── morphological_analyzer.py # Morphological preservation metrics
│   │
│   ├── models/
│   │   ├── model_factory.py          # Model creation hub
│   │   ├── enhanced_gpt.py           # Enhanced GPT (50M/110M/350M)
│   │   └── bert_model.py
│   │
│   ├── training/
│   │   ├── trainer.py                # Enhanced trainer with curriculum learning
│   │   ├── curriculum_learning.py    # 5 curriculum strategies
│   │   ├── curriculum_schedule.py    # 5 scheduling approaches
│   │   ├── optimizer_factory.py      # Multiple optimizer support
│   │   ├── scheduler_factory.py      # LR scheduling strategies
│   │   └── mixed_precision.py        # FP16/BF16 training
│   │
│   ├── evaluation/
│   │   ├── evaluation_manager.py     # Evaluation orchestration
│   │   ├── indicglue_evaluator.py    # Hindi NLP benchmark
│   │   ├── multiblimp_evaluator.py   # 14 linguistic phenomena
│   │   ├── morphological_probes.py   # 10 morphological probe tasks
│   │   └── perplexity_evaluator.py   # Language modeling metrics
│   │
│   ├── analysis/                     # Phase 2: Analysis & Visualization
│   │   ├── results_analyzer.py       # Statistical testing & reporting
│   │   └── visualization_utils.py    # ThesisPlotter - publication figures
│   │
│   └── utils/
│       ├── experiment_config.py      # Configuration management
│       ├── seed_manager.py           # Reproducibility utilities
│       ├── checkpoint_manager.py     # Model checkpoint handling
│       └── logger.py                 # Logging utilities
│
├── notebooks/                        # Interactive Analysis
│   ├── 01_data_exploration.ipynb     # Corpus statistics & quality
│   └── 02_results_analysis.ipynb     # Experiment results & visualizations
│
├── experiments/
│   ├── run_experiment.py             # Main experiment orchestrator
│   ├── run_tokenization_experiments.py
│   ├── run_architecture_experiments.py
│   └── run_curriculum_experiments.py
│
├── configs/                          # Configuration Files
│   ├── base_config.yaml              # Base configuration
│   ├── tiny_model.yaml               # 50M parameter model
│   ├── small_model.yaml              # 110M parameter model
│   ├── medium_model.yaml             # 350M parameter model
│   ├── curriculum_learning.yaml      # Curriculum learning configs
│   └── position_encodings.yaml       # Position encoding variants
│
├── docs/                             # Comprehensive Documentation
│   ├── 01_PROJECT_OVERVIEW.md        # Project architecture & phases
│   ├── 02_DATA_PROCESSING.md         # Data pipeline & IndicCorp
│   ├── 03_TOKENIZATION.md            # Tokenization strategies
│   ├── 04_MODELS.md                  # Model architectures & position encodings
│   ├── 05_TRAINING.md                # Training & curriculum learning
│   ├── 06_EVALUATION.md              # Evaluation frameworks
│   ├── 07_CONFIGURATION.md           # Configuration guide
│   ├── 08_ANALYSIS_AND_VISUALIZATION.md  # ResultsAnalyzer & ThesisPlotter
│   ├── 09_JUPYTER_NOTEBOOKS.md       # Notebook usage guide
│   └── 10_THESIS_INTEGRATION.md      # LaTeX integration & thesis workflow
│
├── results/                          # Experiment Results
│   └── [experiment_name]/
│       ├── metadata.json
│       ├── config.yaml
│       ├── training_summary.json
│       ├── evaluation_results.json
│       └── checkpoints/
│
├── figures/                          # Generated Figures
│   ├── training_curves.png
│   ├── multiblimp_comparison.png
│   ├── morphological_probes_comparison.png
│   └── ...
│
├── tables/                           # LaTeX Tables
│   ├── indicglue_results.tex
│   ├── multiblimp_results.tex
│   └── probes_results.tex
│
├── reports/                          # Markdown Reports
│   └── [experiment_name]_report.md
│
├── slurm_scripts/                    # HPC/LRZ Job Scripts
│   ├── README_LRZ.md                 # Complete LRZ setup guide
│   ├── QUICK_REFERENCE.md            # Command cheatsheet
│   ├── run_complete_pipeline.sh      # Full pipeline (24h, 1 GPU)
│   ├── run_data_processing.sh        # Data only (4h, CPU)
│   ├── run_training.sh               # Training only (48h, 1 GPU)
│   ├── run_evaluation.sh             # Evaluation only (8h, 1 GPU)
│   ├── run_tiny_model.sh             # Quick test (12h, 1 GPU)
│   └── run_curriculum_learning.sh    # Curriculum learning (48h, 1 GPU)
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
- `run_curriculum_learning.sh` - Curriculum learning (48h, 1 GPU)

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
- Download and process data from all sources (IndicCorp, Wikipedia, children's books)
- Create train/val/test splits in `data/splits/`
- Train a language model with your configuration
- Run comprehensive evaluation (IndicGLUE, MultiBLiMP, Morphological Probes)
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
    --config configs/tiny_model.yaml \
    --stage train \
    --experiment_name tiny_baseline

# Train Small model (110M params) with curriculum learning
python main.py \
    --config configs/curriculum_learning.yaml \
    --stage train \
    --experiment_name small_curriculum
```

**Evaluation Only** (requires trained model):
```bash
# Evaluate on all benchmarks (IndicGLUE, MultiBLiMP, Probes)
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
    --resume results/previous_exp/checkpoints/checkpoint_epoch_5.pt
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

### Advanced Usage

#### Experiment with Position Encodings

```bash
# Train with RoPE (Rotary Position Embeddings)
python main.py \
    --config configs/position_encodings.yaml \
    --experiment_name rope_experiment

# Train with ALiBi (Attention with Linear Biases)
python main.py \
    --config configs/position_encodings.yaml \
    --experiment_name alibi_experiment

# Compare all 5 position encoding types (automated suite)
python experiments/run_architecture_experiments.py
```

#### Curriculum Learning Experiments

```bash
# Specific curriculum strategy with morphological ordering
python main.py \
    --config configs/curriculum_learning.yaml \
    --experiment_name morphological_curriculum

# Length-based curriculum learning
python main.py \
    --config configs/curriculum_learning.yaml \
    --experiment_name length_curriculum

# Run all 25 curriculum learning configurations (automated suite)
python experiments/run_curriculum_experiments.py
```

#### Tokenization Comparison

```bash
# Train with SentencePiece tokenization
python main.py \
    --config configs/sentencepiece_config.yaml \
    --experiment_name sentencepiece_exp

# Compare SentencePiece, WordPiece, BPE (automated suite)
python experiments/run_tokenization_experiments.py
```

#### Model Size Experiments

```bash
# Tiny model (50M parameters) - fast training
python main.py \
    --config configs/tiny_model.yaml \
    --experiment_name tiny_50m

# Small model (110M parameters) - balanced
python main.py \
    --config configs/small_model.yaml \
    --experiment_name small_110m

# Medium model (350M parameters) - best performance
python main.py \
    --config configs/medium_model.yaml \
    --experiment_name medium_350m
```

## 📊 Data Processing Pipeline

### 1. IndicCorp V2 Download

The project includes a fully implemented IndicCorp downloader:

```python
from src.data_processing.indiccorp_downloader import download_indiccorp_hindi

# Simple download
paths = download_indiccorp_hindi(
    output_dir='data/raw',
    num_samples=100000,
    streaming=False,
    save_format='both'
)

# Returns:
# {
#     'text': Path('data/raw/indiccorp_hindi.txt'),
#     'pickle': Path('data/raw/indiccorp_hindi.pkl'),
#     'statistics': Path('data/raw/indiccorp_statistics.json'),
#     'metadata': Path('data/raw/indiccorp_metadata.json')
# }
```

**Features**:
- HuggingFace integration with automatic caching
- Streaming mode for memory efficiency
- Comprehensive statistics and metadata
- Command-line interface
- Progress tracking with tqdm

### 2. Quality Filtering

- **Length Filtering**: Remove too short (<10 chars) or too long (>1000 chars) texts
- **Language Detection**: Ensure ≥80% Devanagari characters
- **Deduplication**: MinHash LSH algorithm for exact and near-duplicate detection
- **Token Limiting**: Precisely limit corpus to ~10M tokens

### 3. Data Splits

Splits are saved to `data/splits/` (updated path):
- **Train**: 80% (~8M tokens)
- **Validation**: 10% (~1M tokens)
- **Test**: 10% (~1M tokens)

## 🔧 Configuration

All experiments are configured via YAML files in `configs/`. Key configuration sections:

```yaml
# configs/base_config.yaml

data:
  sources:
    indiccorp: 0.6      # 60% from IndicCorp
    wikipedia: 0.3      # 30% from Wikipedia
    childrens_books: 0.1  # 10% from children's books
  max_tokens: 10_000_000

tokenization:
  type: sentencepiece    # sentencepiece, wordpiece, bpe
  vocab_size: 32000
  model_type: unigram    # unigram, bpe, char, word

model:
  type: enhanced_gpt
  size: small            # tiny (50M), small (110M), medium (350M)
  position_encoding: sinusoidal  # sinusoidal, learned, rope, alibi, relative_bias
  architecture:
    hidden_size: 768
    num_layers: 12
    num_heads: 12
    max_position_embeddings: 1024

training:
  batch_size: 32
  num_epochs: 10
  learning_rate: 5e-4
  optimizer: adamw       # adamw, adam, sgd
  scheduler: cosine      # linear, cosine, constant
  mixed_precision: fp16  # fp16, bf16, fp32

  curriculum:
    use_curriculum: true
    curriculum_strategy: morphological  # morphological, length, frequency, combined, dynamic
    curriculum_schedule: linear         # linear, root, exponential, step, performance_based

evaluation:
  benchmarks:
    - indicglue          # Hindi NLP tasks
    - multiblimp         # 14 linguistic phenomena
    - morphological_probes  # 10 probe tasks

reproducibility:
  seed: 42
  set_deterministic: true
```

See [Configuration Documentation](docs/07_CONFIGURATION.md) for complete guide.

## 📈 Evaluation Frameworks

### 1. MultiBLiMP (14 Linguistic Phenomena)

Tests grammatical competence with minimal pairs:

**Agreement**:
- `subject_verb_agreement_number`: संख्या सहमति (number)
- `subject_verb_agreement_person`: पुरुष सहमति (person)
- `subject_verb_agreement_gender`: लिंग सहमति (gender)
- `determiner_noun_agreement`: निर्धारक-संज्ञा सहमति

**Case Marking**:
- `ergative_case`: कर्तृकारक (-ने)
- `accusative_case`: कर्मकारक (-को)
- `instrumental_case`: करणकारक (-से)

**And 7 more phenomena** (word order, scrambling, binding, etc.)

### 2. Morphological Probes (10 Tasks)

Layer-wise analysis of morphological features:
- Number detection, Gender detection, Person detection
- Case marking, Tense/aspect, Mood detection
- Voice detection, Definiteness, Postposition attachment
- Compound verb structure

### 3. IndicGLUE

Standard Hindi NLP benchmarks for downstream task evaluation.

## 📊 Results Analysis

### Interactive Jupyter Notebooks

#### 1. Data Exploration (`notebooks/01_data_exploration.ipynb`)

Analyzes corpus characteristics:
- Basic statistics (tokens, TTR, sentence lengths)
- Length distributions with histograms
- Character analysis (Devanagari ratio, frequency)
- Word frequency analysis
- Morphological complexity (case markers: ने, को, से, का, etc.)
- Data quality assessment

**Generated Outputs**:
- `figures/length_distributions.png`
- `figures/character_distribution.png`
- `figures/word_frequency.png`
- `figures/case_markers.png`
- `data/corpus_statistics.json`

#### 2. Results Analysis (`notebooks/02_results_analysis.ipynb`)

Comprehensive experiment analysis:
- Training curves comparison
- IndicGLUE/MultiBLiMP performance comparison
- Statistical significance testing (t-test, Wilcoxon, effect sizes)
- Layer-wise probe visualizations
- LaTeX table generation for thesis

**Generated Outputs**:
- `figures/training_curves.png`
- `figures/multiblimp_comparison.png`
- `figures/morphological_probes_comparison.png`
- `tables/indicglue_results.tex`
- `tables/multiblimp_results.tex`
- `reports/[experiment]_report.md`

### Command-Line Analysis

```bash
# Generate comprehensive analysis for all experiments
python src/analysis/results_analyzer.py \
    --results_dir results/ \
    --output_dir analysis/

# Compare two specific experiments with statistical tests
python src/analysis/results_analyzer.py \
    --compare baseline_exp curriculum_exp \
    --metric accuracy \
    --alpha 0.05
```

### Statistical Testing

The `ResultsAnalyzer` automatically performs:
- **Paired t-test**: Parametric significance testing
- **Wilcoxon signed-rank test**: Non-parametric alternative
- **Cohen's d**: Effect size calculation
- **Bootstrap confidence intervals**: 95% CI with 10,000 iterations

## 🎓 Thesis Integration

This project is designed for academic thesis work with first-class LaTeX support.

### Automatic LaTeX Table Generation

```python
from src.analysis.results_analyzer import analyze_experiments

analyzer = analyze_experiments(results_dir='results/')

# Generate thesis-ready LaTeX table
latex_table = analyzer.generate_latex_table(
    eval_type='multiblimp',
    metric='accuracy',
    caption='MultiBLiMP Syntactic Evaluation Results',
    label='tab:multiblimp',
    save_path='tables/multiblimp_results.tex'
)

# In your thesis .tex file:
# \input{tables/multiblimp_results.tex}
```

### Publication-Quality Figures

```python
from src.analysis.visualization_utils import ThesisPlotter

plotter = ThesisPlotter(style='thesis')  # Consistent IEEE/thesis styling

# Training curves with confidence intervals
fig = plotter.plot_training_curves_with_ci(
    experiments=['baseline', 'curriculum'],
    metrics=['loss', 'perplexity'],
    save_path='figures/training_comparison.png'
)
```

See [Thesis Integration Guide](docs/10_THESIS_INTEGRATION.md) for complete workflow.

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [01_PROJECT_OVERVIEW.md](docs/01_PROJECT_OVERVIEW.md) | Project architecture, phases, and statistics |
| [02_DATA_PROCESSING.md](docs/02_DATA_PROCESSING.md) | Data pipeline, IndicCorp downloader, quality filtering |
| [03_TOKENIZATION.md](docs/03_TOKENIZATION.md) | Tokenization strategies and morphological analysis |
| [04_MODELS.md](docs/04_MODELS.md) | Model architectures and 5 position encoding types |
| [05_TRAINING.md](docs/05_TRAINING.md) | Training pipeline and curriculum learning (5×5 matrix) |
| [06_EVALUATION.md](docs/06_EVALUATION.md) | MultiBLiMP (14 phenomena), morphological probes (10 tasks) |
| [07_CONFIGURATION.md](docs/07_CONFIGURATION.md) | Complete configuration reference |
| [08_ANALYSIS_AND_VISUALIZATION.md](docs/08_ANALYSIS_AND_VISUALIZATION.md) | ResultsAnalyzer and ThesisPlotter API |
| [09_JUPYTER_NOTEBOOKS.md](docs/09_JUPYTER_NOTEBOOKS.md) | Interactive analysis workflows |
| [10_THESIS_INTEGRATION.md](docs/10_THESIS_INTEGRATION.md) | LaTeX integration and thesis workflow |

**Documentation Statistics**: ~55,000 lines covering all aspects of the project.

## 🧪 Running Experiments

### Complete Experiment Suite

```bash
# 1. Tokenization comparison (SentencePiece vs WordPiece vs BPE)
python experiments/run_tokenization_experiments.py

# 2. Position encoding comparison (5 types)
python experiments/run_architecture_experiments.py

# 3. Curriculum learning comparison (5 strategies × 5 schedules)
python experiments/run_curriculum_experiments.py

# 4. Model size comparison (Tiny 50M, Small 110M, Medium 350M)
python experiments/run_model_size_experiments.py
```

### Custom Experiment with main.py

```bash
# Complete pipeline with custom configuration
python main.py \
    --config configs/my_custom_config.yaml \
    --experiment_name my_custom_experiment

# With all advanced options
python main.py \
    --config configs/my_custom_config.yaml \
    --experiment_name my_experiment \
    --seed 42 \
    --device cuda \
    --force-reprocess
```

### Programmatic Usage (Advanced)

For more complex orchestration, use the ExperimentOrchestrator:

```python
from experiments.run_experiment import ExperimentOrchestrator

# Initialize with custom config
orchestrator = ExperimentOrchestrator(
    config_path='configs/my_custom_config.yaml',
    experiment_name='my_experiment'
)

# Run complete pipeline
result = orchestrator.run_full_pipeline()

# Or run specific stages with more control
splits = orchestrator.run_data_processing()
model, tokenizer = orchestrator.run_training(splits)
results = orchestrator.run_evaluation(model, tokenizer, splits)
```

### Resume Training

```bash
# Resume from checkpoint using main.py
python main.py \
    --config configs/base_config.yaml \
    --stage train \
    --experiment_name resumed_training \
    --resume results/previous_exp/checkpoints/checkpoint_epoch_5.pt

# Resume with different config (transfer learning)
python main.py \
    --config configs/fine_tuning_config.yaml \
    --stage train \
    --experiment_name fine_tuned \
    --resume results/base_model/checkpoints/checkpoint_best.pt
```

## 🔬 Key Research Questions

This implementation explores:

1. **Position Encodings**: Which position encoding (RoPE, ALiBi, etc.) works best for Hindi's morphologically rich structure?

2. **Curriculum Learning**: Does morphology-based curriculum learning improve syntactic competence more than length-based or frequency-based curricula?

3. **Data Efficiency**: Can models learn Hindi grammar with only 10M tokens (vs billions in typical pretraining)?

4. **Tokenization**: How do different tokenization strategies (SentencePiece, WordPiece, BPE) preserve Hindi morphological boundaries?

5. **Model Size**: What's the optimal model size for data-efficient Hindi language modeling?

## 📦 Project Statistics

- **Total Lines of Code**: ~12,500 lines
- **Documentation**: ~55,000 lines
- **Python Modules**: 45+ files
- **Configuration Templates**: 5 YAML files
- **Evaluation Tasks**: 24 total (14 MultiBLiMP + 10 Probes)
- **Supported Position Encodings**: 5 types
- **Curriculum Strategies**: 5 strategies × 5 schedules = 25 combinations
- **Model Sizes**: 3 (Tiny 50M, Small 110M, Medium 350M)

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
Email: ayush.kumar@tum.de
GitHub: [@ayushtalreja](https://github.com/ayushtalreja)

## 📖 Citation

If you use this code or data for research, please cite:

```bibtex
@mastersthesis{kumar2025hindi_babylm,
  title={Hindi BabyLM: Data-Efficient Language Modeling for Morphologically Rich Languages},
  author={Kumar, Ayush},
  year={2025},
  school={Technical University of Munich},
  type={Master's Thesis},
  note={Implementation includes 5 position encodings, curriculum learning (25 configurations),
        MultiBLiMP evaluation (14 phenomena), and morphological probes (10 tasks)}
}
```

## 🗺️ Roadmap

- [x] Phase 1: Core Implementation (Models, Training, Evaluation)
- [x] Phase 2: Analysis & Visualization (Statistical testing, LaTeX integration)
- [ ] Phase 3: Extended Experiments (Cross-lingual transfer, multilingual models)
- [ ] Phase 4: Deployment (Model serving, API endpoints)

---

**Built with ❤️ for Hindi NLP research**
