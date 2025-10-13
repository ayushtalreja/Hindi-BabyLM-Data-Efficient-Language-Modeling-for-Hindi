# Hindi BabyLM: Data-Efficient Language Modeling for Hindi

This repository contains the complete implementation for creating a Hindi version of the BabyLM challenge, focusing on training language models with developmentally plausible amounts of data (~10M tokens).

## Project Structure

```
hindi-babylm/
├── data/                       # Dataset storage
│   ├── raw/                    # Raw downloaded datasets
│   ├── processed/              # Cleaned and processed datasets
│   ├── splits/                 # Train/validation/test splits
│   └── tokenized/              # Tokenized datasets
├── src/                        # Source code
│   ├── data_processing/        # Data collection and preprocessing
│   ├── tokenization/          # Tokenization experiments
│   ├── models/                # Model architectures
│   ├── training/              # Training pipelines
│   ├── evaluation/            # Evaluation frameworks
│   └── utils/                 # Utility functions
├── configs/                   # Configuration files
├── notebooks/                 # Analysis notebooks
├── results/                   # Experimental results
└── docs/                      # Documentation
```

## Quick Start

1. **Setup Environment**
```bash
./setup_env.sh
```

2. **Download and Process Data**
```bash
python main.py --config configs/base_config.yaml --stage data --experiment_name hindi_babylm_baseline
```

3. **Train Model**
```bash  
python main.py --config configs/base_config.yaml --stage train --experiment_name hindi_babylm_baseline
```

4. **Run Evaluation**
```bash
python main.py --config configs/base_config.yaml --stage eval --experiment_name hindi_babylm_baseline
```

5. **Run Complete Pipeline**
```bash
python main.py --config configs/base_config.yaml --stage all --experiment_name hindi_babylm_baseline
```

## Experiments

Run tokenization comparison:
```bash
python experiments/run_tokenization_experiments.py
```

Run architecture comparison:  
```bash
python experiments/run_architecture_experiments.py
```

Run curriculum learning experiments:
```bash
python experiments/run_curriculum_experiments.py
```

## Analysis

Generate comprehensive analysis:
```bash
python src/utils/results_analyzer.py --results_dir results/ --output_dir analysis/
```

## Key Components

### Data Processing
- **IndicCorp Hindi**: News articles dataset
- **Hindi Wikipedia**: Encyclopedia content
- **Children's Stories**: Developmentally appropriate text
- **Quality Filtering**: Language detection, deduplication, readability

### Tokenization  
- **SentencePiece**: Subword tokenization with Hindi optimization
- **WordPiece**: BERT-style tokenization
- **BPE**: Byte-Pair Encoding
- **Morphological Analysis**: Hindi-specific evaluation

### Models
- **GPT-2 Style**: Autoregressive language modeling
- **BERT Style**: Masked language modeling  
- **Hybrid**: Combined causal and masked objectives

### Evaluation
- **IndicGLUE**: Hindi NLP benchmark suite
- **MultiBLiMP**: Grammatical competence evaluation
- **Morphological Probes**: Case marking, agreement, etc.

## Configuration

Edit `configs/base_config.yaml` to modify:
- Data sources and mixing ratios
- Tokenization strategy and vocabulary size
- Model architecture and hyperparameters
- Training settings and curriculum learning
- Evaluation metrics and benchmarks

## Results

Results are automatically saved to `results/` directory with:
- Training logs and metrics
- Model checkpoints  
- Evaluation scores
- Analysis visualizations

## Citation

If you use this code for research, please cite:

```bibtex
@article{kumar_ayush_2025_hindi_babylm,
  title={Hindi BabyLM: Data-Efficient Language Modeling for Morphologically Rich Languages},
  author={Ayush Kumar},
  journal={Technical University of Munich},
  year={2025}
}
```
