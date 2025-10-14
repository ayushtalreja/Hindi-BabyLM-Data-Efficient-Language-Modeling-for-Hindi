#!/usr/bin/env python3
"""
Hindi BabyLM: Main Training Pipeline

This is the main entry point for the Hindi BabyLM project. It orchestrates
the complete pipeline from data collection to evaluation.

Usage:
    # Run complete pipeline
    python main.py --config configs/base_config.yaml --experiment_name my_experiment

    # Run specific stage
    python main.py --config configs/base_config.yaml --stage data

    # Resume training from checkpoint
    python main.py --config configs/base_config.yaml --stage train --resume checkpoints/checkpoint_epoch_5.pt

    # Force reprocess data
    python main.py --config configs/base_config.yaml --stage all --force-reprocess
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processing.corpus_builder import CorpusBuilder
from src.tokenization.tokenizer_factory import TokenizerFactory
from src.models.model_factory import ModelFactory
from src.training.trainer import HindiLanguageModelTrainer
from src.evaluation.evaluation_manager import EvaluationManager
from src.utils.experiment_config import ExperimentConfig
from src.utils.seed_manager import set_global_seed


# Configure logging
def setup_logging(experiment_dir: Path):
    """Setup logging to both file and console"""
    log_file = experiment_dir / 'experiment.log'

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def print_banner(text: str, char: str = "="):
    """Print a formatted banner"""
    print(f"\n{char * 80}")
    print(f"{text.center(80)}")
    print(f"{char * 80}\n")


def save_experiment_metadata(
    experiment_dir: Path,
    config: ExperimentConfig,
    experiment_name: str
):
    """Save experiment metadata for reproducibility"""
    metadata = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    # Add CUDA info if available
    if torch.cuda.is_available():
        metadata['cuda_version'] = torch.version.cuda
        metadata['gpu_count'] = torch.cuda.device_count()
        metadata['gpu_names'] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]

    # Save metadata
    metadata_path = experiment_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save config
    config_path = experiment_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)

    logging.info(f"Metadata saved to {metadata_path}")
    logging.info(f"Config saved to {config_path}")


def stage_data_processing(
    config: ExperimentConfig,
    force_reprocess: bool = False
) -> Dict:
    """
    Stage 1: Data Collection and Processing

    Returns:
        Dictionary with train/val/test splits
    """
    print_banner("STAGE 1: DATA PROCESSING")
    logging.info("Starting data processing stage...")

    corpus_builder = CorpusBuilder(config)

    # Check if processed data already exists
    data_dir = Path(config.__dict__.get('data_dir', 'data'))
    splits_dir = data_dir / 'splits'
    splits_exist = (
        (splits_dir / 'train.pkl').exists() and
        (splits_dir / 'val.pkl').exists() and
        (splits_dir / 'test.pkl').exists()
    )

    if splits_exist and not force_reprocess:
        logging.info("Found existing processed data. Loading splits...")
        logging.info("(Use --force-reprocess to reprocess from scratch)")
        splits = corpus_builder.load_splits()
    else:
        if force_reprocess:
            logging.info("Force reprocessing data from scratch...")
        else:
            logging.info("No existing data found. Processing from scratch...")

        # Step 1: Collect raw data
        logging.info("\n📥 Collecting data from all sources...")
        raw_data = corpus_builder.collect_all_data()
        total_raw = sum(len(texts) for texts in raw_data.values())
        logging.info(f"   Collected {total_raw:,} raw documents")

        # Step 2: Process and filter
        logging.info("\n🔧 Processing and filtering data...")
        processed_data = corpus_builder.process_and_filter(raw_data)
        logging.info(f"   After processing: {len(processed_data):,} documents")

        # Step 3: Create splits
        logging.info("\n✂️  Creating train/val/test splits...")
        splits = corpus_builder.create_splits(processed_data)
        logging.info(f"   Train: {len(splits['train']):,} samples")
        logging.info(f"   Val:   {len(splits['val']):,} samples")
        logging.info(f"   Test:  {len(splits['test']):,} samples")

        # Step 4: Save splits
        logging.info("\n💾 Saving splits to disk...")
        corpus_builder.save_splits(splits)
        logging.info(f"   Saved to: {splits_dir}")

    logging.info("\n✅ Data processing completed successfully!")
    return splits


def stage_training(
    config: ExperimentConfig,
    splits: Dict,
    experiment_dir: Path,
    resume_from: Optional[str] = None
):
    """
    Stage 2: Model Training

    Args:
        config: Experiment configuration
        splits: Data splits dictionary
        experiment_dir: Directory to save results
        resume_from: Optional checkpoint path to resume from
    """
    print_banner("STAGE 2: MODEL TRAINING")
    logging.info("Starting model training stage...")

    # Step 1: Create tokenizer
    logging.info("\n🔤 Creating tokenizer...")
    tokenizer_factory = TokenizerFactory(config)
    tokenizer = tokenizer_factory.create_tokenizer(splits['train'])

    # Save tokenizer
    tokenizer_save_dir = experiment_dir / 'tokenizer'
    tokenizer_save_dir.mkdir(exist_ok=True)
    tokenizer_factory.save_tokenizer(tokenizer, str(tokenizer_save_dir))
    logging.info(f"   Tokenizer saved to: {tokenizer_save_dir}")
    logging.info(f"   Vocabulary size: {tokenizer.vocab_size:,}")

    # Step 2: Create model
    logging.info("\n🏗️  Creating model...")
    model_factory = ModelFactory(config)
    model = model_factory.create_model(tokenizer.vocab_size)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"   Total parameters: {total_params:,}")
    logging.info(f"   Trainable parameters: {trainable_params:,}")

    # Step 3: Create trainer
    logging.info("\n👨‍🏫 Initializing trainer...")
    trainer = HindiLanguageModelTrainer(model, tokenizer, config.__dict__)

    # Resume from checkpoint if specified
    if resume_from:
        logging.info(f"   Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Step 4: Create data loaders
    logging.info("\n📊 Creating data loaders...")
    corpus_builder = CorpusBuilder(config)
    train_dataloader = corpus_builder.create_dataloader(
        splits['train'], tokenizer, 'train'
    )
    val_dataloader = corpus_builder.create_dataloader(
        splits['val'], tokenizer, 'val'
    )
    logging.info(f"   Train batches: {len(train_dataloader):,}")
    logging.info(f"   Val batches:   {len(val_dataloader):,}")

    # Step 5: Train model
    logging.info("\n🚀 Starting training...")
    logging.info(f"   Epochs: {config.__dict__.get('num_epochs', 10)}")
    logging.info(f"   Batch size: {config.__dict__.get('batch_size', 32)}")
    logging.info(f"   Learning rate: {config.__dict__.get('learning_rate', 5e-4)}")

    num_epochs = config.__dict__.get('num_epochs', 10)
    trainer.train(train_dataloader, val_dataloader, num_epochs)

    # Step 6: Save training summary
    logging.info("\n💾 Saving training summary...")
    summary = trainer.get_training_summary()
    summary_path = experiment_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"   Summary saved to: {summary_path}")

    logging.info("\n✅ Model training completed successfully!")
    return model, tokenizer


def stage_evaluation(
    config: ExperimentConfig,
    model,
    tokenizer,
    splits: Dict,
    experiment_dir: Path
):
    """
    Stage 3: Comprehensive Evaluation

    Args:
        config: Experiment configuration
        model: Trained model
        tokenizer: Trained tokenizer
        splits: Data splits dictionary
        experiment_dir: Directory to save results
    """
    print_banner("STAGE 3: EVALUATION")
    logging.info("Starting evaluation stage...")

    # Create evaluator
    logging.info("\n📋 Initializing evaluation manager...")
    evaluator = EvaluationManager(model, tokenizer, config.__dict__)

    # Run comprehensive evaluation
    logging.info("\n🔍 Running comprehensive evaluation...")
    logging.info("   This may take several minutes...")

    # Get enabled benchmarks from config
    benchmarks = config.__dict__.get('evaluation', {}).get('benchmarks', [])
    if not benchmarks:
        benchmarks = ['indicglue', 'multiblimp', 'morphological_probes']

    logging.info(f"   Enabled benchmarks: {', '.join(benchmarks)}")

    results = evaluator.run_comprehensive_evaluation()

    # Save results
    logging.info("\n💾 Saving evaluation results...")
    results_path = experiment_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"   Results saved to: {results_path}")

    # Print summary
    logging.info("\n📊 Evaluation Summary:")
    for benchmark, scores in results.items():
        if isinstance(scores, dict):
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            logging.info(f"   {benchmark}: {avg_score:.4f} average")

    logging.info("\n✅ Evaluation completed successfully!")
    return results


def main():
    """Main pipeline execution"""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Hindi BabyLM: Data-Efficient Language Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --config configs/base_config.yaml --experiment_name baseline

  # Run specific stage
  python main.py --config configs/base_config.yaml --stage data

  # Resume training from checkpoint
  python main.py --config configs/base_config.yaml --stage train --resume checkpoints/checkpoint_epoch_5.pt

  # Force reprocess data
  python main.py --config configs/base_config.yaml --stage all --force-reprocess

  # Run with custom seed
  python main.py --config configs/base_config.yaml --seed 123
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['data', 'train', 'eval', 'all'],
        default='all',
        help='Which stage to run (default: all)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='Name for this experiment (auto-generated if not provided)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint path'
    )
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing of data even if it exists'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for training (default: auto)'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = ExperimentConfig.load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    # Set experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        # Auto-generate experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config.__dict__.get('model', {}).get('type', 'model')
        experiment_name = f"{model_type}_{timestamp}"

    config.experiment_name = experiment_name

    # Create experiment directory
    experiment_dir = Path('results') / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(experiment_dir)

    # Print header
    print_banner("HINDI BABYLM: DATA-EFFICIENT LANGUAGE MODELING", "=")
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Stage: {args.stage}")
    logging.info(f"Results directory: {experiment_dir}")
    logging.info(f"Configuration: {args.config}")

    # Setup reproducibility
    seed = args.seed if args.seed else config.__dict__.get('reproducibility', {}).get('seed', 42)
    deterministic = config.__dict__.get('reproducibility', {}).get('set_deterministic', True)

    logging.info(f"\n🎲 Setting up reproducibility...")
    logging.info(f"   Seed: {seed}")
    logging.info(f"   Deterministic: {deterministic}")

    set_global_seed(seed=seed, deterministic=deterministic)

    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logging.info(f"\n💻 Device configuration:")
    logging.info(f"   Device: {device}")
    if device == 'cuda':
        logging.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"   CUDA Version: {torch.version.cuda}")

    # Save experiment metadata
    save_experiment_metadata(experiment_dir, config, experiment_name)

    # Initialize variables for stages
    splits = None
    model = None
    tokenizer = None

    try:
        # Stage 1: Data Processing
        if args.stage in ['data', 'all']:
            splits = stage_data_processing(config, args.force_reprocess)

        # Stage 2: Training
        if args.stage in ['train', 'all']:
            # Load splits if not already loaded
            if splits is None:
                corpus_builder = CorpusBuilder(config)
                splits = corpus_builder.load_splits()

            model, tokenizer = stage_training(
                config,
                splits,
                experiment_dir,
                resume_from=args.resume
            )

        # Stage 3: Evaluation
        if args.stage in ['eval', 'all']:
            # Load model and tokenizer if not already loaded
            if model is None or tokenizer is None:
                logging.info("\nLoading trained model and tokenizer...")
                model_factory = ModelFactory(config)
                model = model_factory.load_trained_model(experiment_name)
                tokenizer = TokenizerFactory.load_tokenizer(experiment_name)

            # Load splits if needed
            if splits is None:
                corpus_builder = CorpusBuilder(config)
                splits = corpus_builder.load_splits()

            stage_evaluation(
                config,
                model,
                tokenizer,
                splits,
                experiment_dir
            )

        # Mark experiment as completed
        completion_marker = experiment_dir / 'COMPLETED'
        completion_marker.touch()

        # Final summary
        print_banner("PIPELINE COMPLETED SUCCESSFULLY", "=")
        logging.info(f"✅ All stages completed successfully!")
        logging.info(f"📁 Results directory: {experiment_dir}")
        logging.info(f"📊 View results:")
        logging.info(f"   - Training summary: {experiment_dir}/training_summary.json")
        logging.info(f"   - Evaluation results: {experiment_dir}/evaluation_results.json")
        logging.info(f"   - Logs: {experiment_dir}/experiment.log")

        return 0

    except KeyboardInterrupt:
        logging.warning("\n⚠️  Pipeline interrupted by user")
        failure_marker = experiment_dir / 'INTERRUPTED'
        failure_marker.touch()
        return 130

    except Exception as e:
        logging.error(f"\n❌ Pipeline failed with error: {e}", exc_info=True)

        # Mark experiment as failed
        failure_marker = experiment_dir / 'FAILED'
        with open(failure_marker, 'w') as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")

        return 1


if __name__ == "__main__":
    sys.exit(main())
