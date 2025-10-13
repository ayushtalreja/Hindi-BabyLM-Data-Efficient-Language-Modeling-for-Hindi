#!/usr/bin/env python3
"""
Experiment Orchestration Script for Hindi BabyLM

This script provides a unified interface for running experiments with different
configurations, automatic result tracking, and reproducibility guarantees.

Usage:
    # Run single experiment
    python experiments/run_experiment.py --config configs/base_config.yaml --name baseline

    # Run multiple experiments
    python experiments/run_experiment.py --experiment_suite tokenization_comparison

    # Resume from checkpoint
    python experiments/run_experiment.py --config configs/base_config.yaml --resume checkpoints/checkpoint_best.pt
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.corpus_builder import CorpusBuilder
from src.tokenization.tokenizer_factory import TokenizerFactory
from src.models.model_factory import ModelFactory
from src.training.trainer import HindiLanguageModelTrainer
from src.evaluation.evaluation_manager import EvaluationManager
from src.utils.experiment_config import ExperimentConfig
from src.utils.seed_manager import set_global_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiments/experiment.log')
    ]
)
logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """
    Orchestrates experiments with automatic tracking and reproducibility

    Features:
    - Automatic experiment naming and versioning
    - Git commit tracking
    - Environment snapshot
    - Result aggregation
    - Checkpoint management
    """

    def __init__(self, config_path: str, experiment_name: Optional[str] = None):
        """
        Initialize experiment orchestrator

        Args:
            config_path: Path to configuration file
            experiment_name: Optional experiment name override
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Set experiment name
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            self.experiment_name = self._generate_experiment_name()

        self.config['experiment_name'] = self.experiment_name

        # Create experiment directory
        self.experiment_dir = Path('results') / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Results directory: {self.experiment_dir}")

        # Setup reproducibility
        self._setup_reproducibility()

        # Track experiment metadata
        self.metadata = self._collect_metadata()
        self._save_metadata()

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        logger.info(f"Loading configuration from {self.config_path}")

        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested structure for easier access
        flat_config = self._flatten_config(config_dict)

        return flat_config

    def _flatten_config(self, config_dict: Dict, parent_key: str = '') -> Dict:
        """Flatten nested configuration dictionary"""
        items = []
        for k, v in config_dict.items():
            new_key = f"{parent_key}.{k}" if parent_key else k

            if isinstance(v, dict) and not any(key in k for key in ['project', 'directories']):
                # Recursively flatten
                items.extend(self._flatten_config(v, new_key).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def _generate_experiment_name(self) -> str:
        """Generate unique experiment name with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.config.get('model.type', 'unknown')
        tokenizer_type = self.config.get('tokenization.type', 'unknown')

        return f"{model_type}_{tokenizer_type}_{timestamp}"

    def _setup_reproducibility(self):
        """Setup reproducibility (seeds, determinism)"""
        seed = self.config.get('reproducibility.seed', 42)
        deterministic = self.config.get('reproducibility.set_deterministic', True)

        logger.info(f"Setting up reproducibility with seed={seed}")
        self.seed_manager = set_global_seed(seed=seed, deterministic=deterministic)

    def _collect_metadata(self) -> Dict:
        """Collect experiment metadata"""
        metadata = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config_path': str(self.config_path),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }

        # Add CUDA info if available
        if torch.cuda.is_available():
            metadata['cuda_version'] = torch.version.cuda
            metadata['cudnn_version'] = torch.backends.cudnn.version()
            metadata['gpu_count'] = torch.cuda.device_count()
            metadata['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

        # Add git info if requested
        if self.config.get('reproducibility.track_git_commit', False):
            metadata['git_info'] = self._get_git_info()

        # Add environment info
        if self.config.get('reproducibility.save_environment', True):
            metadata['pip_packages'] = self._get_pip_freeze()

        return metadata

    def _get_git_info(self) -> Dict:
        """Get git repository information"""
        try:
            import subprocess

            git_info = {}

            # Get commit hash
            git_info['commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=project_root
            ).decode('utf-8').strip()

            # Get branch name
            git_info['branch'] = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=project_root
            ).decode('utf-8').strip()

            # Check for uncommitted changes
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=project_root
            ).decode('utf-8').strip()
            git_info['has_uncommitted_changes'] = bool(status)

            if git_info['has_uncommitted_changes']:
                logger.warning("Repository has uncommitted changes!")

            return git_info

        except Exception as e:
            logger.warning(f"Could not get git info: {e}")
            return {}

    def _get_pip_freeze(self) -> List[str]:
        """Get list of installed Python packages"""
        try:
            import subprocess
            packages = subprocess.check_output(
                [sys.executable, '-m', 'pip', 'freeze']
            ).decode('utf-8').strip().split('\n')
            return packages
        except Exception as e:
            logger.warning(f"Could not get pip freeze: {e}")
            return []

    def _save_metadata(self):
        """Save experiment metadata"""
        metadata_path = self.experiment_dir / 'metadata.json'

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

        # Also save config
        config_save_path = self.experiment_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"Config saved to {config_save_path}")

    def run_data_processing(self) -> Dict:
        """
        Run data collection and processing

        Returns:
            Dictionary with data splits
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: DATA PROCESSING")
        logger.info("="*60)

        # Create corpus builder
        corpus_builder = CorpusBuilder(self.config)

        # Check if processed data exists
        data_dir = Path(self.config.get('directories.data_dir', 'data'))
        splits_exist = (data_dir / 'train.txt').exists()

        if splits_exist and not self.config.get('force_reprocess', False):
            logger.info("Loading existing processed data...")
            splits = corpus_builder.load_splits()
        else:
            logger.info("Processing data from scratch...")

            # Collect data
            raw_data = corpus_builder.collect_all_data()
            logger.info(f"Collected {len(raw_data)} raw documents")

            # Process and filter
            processed_data = corpus_builder.process_and_filter(raw_data)
            logger.info(f"After processing: {len(processed_data)} documents")

            # Create splits
            splits = corpus_builder.create_splits(processed_data)
            logger.info(f"Created splits: train={len(splits['train'])}, "
                       f"val={len(splits['val'])}, test={len(splits['test'])}")

            # Save splits
            corpus_builder.save_splits(splits)

        logger.info("✓ Data processing completed")
        return splits

    def run_training(self, splits: Dict, resume_from: Optional[str] = None):
        """
        Run model training

        Args:
            splits: Data splits dictionary
            resume_from: Optional checkpoint path to resume from

        Returns:
            Trained model and tokenizer
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: MODEL TRAINING")
        logger.info("="*60)

        # Create tokenizer
        logger.info("Creating tokenizer...")
        tokenizer_factory = TokenizerFactory(self.config)
        tokenizer = tokenizer_factory.create_tokenizer(splits['train'])

        # Save tokenizer
        tokenizer_factory.save_tokenizer(tokenizer, self.experiment_name)

        # Create model
        logger.info("Creating model...")
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model(tokenizer.vocab_size)

        # Create trainer
        logger.info("Initializing trainer...")
        trainer = HindiLanguageModelTrainer(model, tokenizer, self.config)

        # Resume from checkpoint if specified
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            trainer.load_checkpoint(resume_from)

        # Create data loaders
        train_loader = self._create_dataloader(splits['train'], tokenizer, 'train')
        val_loader = self._create_dataloader(splits['val'], tokenizer, 'val')

        # Train
        logger.info("Starting training...")
        trainer.train(train_loader, val_loader)

        # Get training summary
        summary = trainer.get_training_summary()
        self._save_training_summary(summary)

        logger.info("✓ Training completed")
        return model, tokenizer

    def run_evaluation(self, model, tokenizer, splits: Dict) -> Dict:
        """
        Run comprehensive evaluation

        Args:
            model: Trained model
            tokenizer: Tokenizer
            splits: Data splits

        Returns:
            Evaluation results dictionary
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: EVALUATION")
        logger.info("="*60)

        # Create evaluator
        evaluator = EvaluationManager(model, tokenizer, self.config)

        # Run evaluations
        results = evaluator.run_comprehensive_evaluation()

        # Save results
        results_path = self.experiment_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Evaluation completed, results saved to {results_path}")
        return results

    def _create_dataloader(self, texts: List[str], tokenizer, split: str):
        """Create data loader from texts"""
        from torch.utils.data import DataLoader, Dataset

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                # Simple encoding (adjust based on your tokenizer interface)
                tokens = self.tokenizer.encode(text)[:self.max_length]

                # Pad to max_length
                input_ids = tokens + [0] * (self.max_length - len(tokens))
                attention_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))

                return {
                    'input_ids': torch.tensor(input_ids),
                    'attention_mask': torch.tensor(attention_mask)
                }

        dataset = TextDataset(texts, tokenizer, self.config.get('model.architecture.max_position_embeddings', 512))

        batch_size = self.config.get('training.batch_size', 32)
        num_workers = self.config.get('resources.num_workers', 4)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=self.config.get('resources.pin_memory', True),
            worker_init_fn=self.seed_manager.worker_init_fn if split == 'train' else None
        )

        return loader

    def _save_training_summary(self, summary: Dict):
        """Save training summary"""
        summary_path = self.experiment_dir / 'training_summary.json'

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {summary_path}")

    def run_full_pipeline(self, resume_from: Optional[str] = None):
        """
        Run complete experiment pipeline

        Args:
            resume_from: Optional checkpoint to resume training from
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"# STARTING EXPERIMENT: {self.experiment_name}")
        logger.info(f"{'#'*60}\n")

        try:
            # Stage 1: Data Processing
            splits = self.run_data_processing()

            # Stage 2: Training
            model, tokenizer = self.run_training(splits, resume_from=resume_from)

            # Stage 3: Evaluation
            results = self.run_evaluation(model, tokenizer, splits)

            # Mark experiment as completed
            completion_marker = self.experiment_dir / 'COMPLETED'
            completion_marker.touch()

            logger.info(f"\n{'#'*60}")
            logger.info(f"# EXPERIMENT COMPLETED SUCCESSFULLY!")
            logger.info(f"# Results: {self.experiment_dir}")
            logger.info(f"{'#'*60}\n")

            return {
                'status': 'success',
                'experiment_name': self.experiment_name,
                'results_dir': str(self.experiment_dir),
                'evaluation_results': results
            }

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)

            # Mark experiment as failed
            failure_marker = self.experiment_dir / 'FAILED'
            with open(failure_marker, 'w') as f:
                f.write(str(e))

            return {
                'status': 'failed',
                'experiment_name': self.experiment_name,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(description="Hindi BabyLM Experiment Orchestrator")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--name', type=str,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    parser.add_argument('--stage', type=str, choices=['data', 'train', 'eval', 'all'],
                       default='all', help='Which stage to run')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of data even if it exists')

    args = parser.parse_args()

    # Add force_reprocess to config if needed
    if args.force_reprocess:
        # Load config temporarily to add this flag
        with open(args.config, 'r') as f:
            temp_config = yaml.safe_load(f)
        temp_config['force_reprocess'] = True

        # Save temporarily
        temp_config_path = '/tmp/temp_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f)
        config_path = temp_config_path
    else:
        config_path = args.config

    # Create orchestrator
    orchestrator = ExperimentOrchestrator(config_path, experiment_name=args.name)

    # Run requested stage
    if args.stage == 'all':
        result = orchestrator.run_full_pipeline(resume_from=args.resume)
    elif args.stage == 'data':
        orchestrator.run_data_processing()
    elif args.stage == 'train':
        splits = orchestrator.run_data_processing()
        orchestrator.run_training(splits, resume_from=args.resume)
    elif args.stage == 'eval':
        # Load model and run evaluation
        logger.error("Eval-only stage not yet implemented")
        sys.exit(1)

    logger.info("Experiment orchestration completed")


if __name__ == "__main__":
    main()
