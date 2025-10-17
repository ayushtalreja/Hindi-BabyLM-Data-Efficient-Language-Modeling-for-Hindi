import os
import torch
from typing import Optional, Dict, Any

from .gpt_model import HindiGPTModel
from .bert_model import HindiBERTModel


class ModelFactory:
    """Factory class for creating and managing different model architectures"""

    def __init__(self, config):
        self.config = config
        self.model_type = config.model_type
        self.model_dir = config.__dict__.get('model_dir', 'models')
        self.experiment_name = config.__dict__.get('experiment_name', 'default_experiment')

        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'checkpoints'), exist_ok=True)

    def create_model(self, vocab_size: int):
        """Create a model based on config"""
        print(f"\nCreating {self.model_type} model...")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Number of layers: {self.config.num_layers}")
        print(f"  Number of heads: {self.config.num_heads}")

        # Prepare config dict for model
        model_config = {
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'max_length': self.config.max_length,
            'dropout': self.config.__dict__.get('dropout', 0.1),
            'intermediate_size': self.config.__dict__.get('intermediate_size', 3072)
        }

        if self.model_type == "gpt":
            model = HindiGPTModel(vocab_size=vocab_size, config=model_config)
        elif self.model_type == "bert":
            model = HindiBERTModel(vocab_size=vocab_size, config=model_config)
        elif self.model_type == "hybrid":
            # For hybrid model, we need to create separate configs for BERT and GPT
            from transformers import BertConfig, GPT2Config

            bert_config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=model_config['hidden_size'],
                num_hidden_layers=model_config['num_layers'],
                num_attention_heads=model_config['num_heads'],
                intermediate_size=model_config['intermediate_size'],
                max_position_embeddings=model_config['max_length'],
                hidden_dropout_prob=model_config['dropout'],
                attention_probs_dropout_prob=model_config['dropout']
            )

            gpt_config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=model_config['max_length'],
                n_embd=model_config['hidden_size'],
                n_layer=model_config['num_layers'],
                n_head=model_config['num_heads'],
                resid_pdrop=model_config['dropout'],
                embd_pdrop=model_config['dropout'],
                attn_pdrop=model_config['dropout']
            )

            hybrid_config = {
                **model_config,
                'bert_config': bert_config,
                'gpt_config': gpt_config
            }

            model = HybridGPTBERTModel(vocab_size=vocab_size, config=hybrid_config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Total parameters: {num_params:,}")
        print(f"  Trainable parameters: {num_trainable_params:,}")

        return model

    def save_model(self, model, tokenizer, checkpoint_name: Optional[str] = None, metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint with metadata"""
        if checkpoint_name is None:
            checkpoint_name = f"{self.experiment_name}_final"

        checkpoint_path = os.path.join(self.model_dir, 'checkpoints', f"{checkpoint_name}.pt")

        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_type': self.model_type,
            'vocab_size': tokenizer.vocab_size,
            'config': self.config.__dict__,
            'experiment_name': self.experiment_name
        }

        if metrics:
            checkpoint['metrics'] = metrics

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

        # Also save just the model state dict for easier loading
        model_path = os.path.join(self.model_dir, f"{self.experiment_name}_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model state dict saved to {model_path}")

        return checkpoint_path

    def load_model(self, checkpoint_path: str, vocab_size: Optional[int] = None):
        """Load model from checkpoint"""
        print(f"Loading model from {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get vocab size from checkpoint if not provided
        if vocab_size is None:
            vocab_size = checkpoint.get('vocab_size')
            if vocab_size is None:
                raise ValueError("vocab_size not found in checkpoint and not provided")

        # Create model with same architecture
        model_type = checkpoint.get('model_type', self.model_type)
        saved_config = checkpoint.get('config', {})

        # Update config with saved values
        for key, value in saved_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Create model
        original_model_type = self.model_type
        self.model_type = model_type
        model = self.create_model(vocab_size)
        self.model_type = original_model_type

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded successfully")

        if 'metrics' in checkpoint:
            print(f"Checkpoint metrics: {checkpoint['metrics']}")

        return model

    def load_trained_model(self, experiment_name: str):
        """Load a trained model by experiment name"""
        # Try to find checkpoint
        checkpoint_path = os.path.join(self.model_dir, 'checkpoints', f"{experiment_name}_final.pt")

        if not os.path.exists(checkpoint_path):
            # Try alternative paths
            checkpoint_path = os.path.join(self.model_dir, 'checkpoints', f"{experiment_name}_best.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found for experiment: {experiment_name}")

        return self.load_model(checkpoint_path)

    def save_checkpoint(self, model, optimizer, epoch: int, step: int, metrics: Dict[str, float]):
        """Save training checkpoint with optimizer state"""
        checkpoint_name = f"{self.experiment_name}_epoch{epoch}_step{step}"
        checkpoint_path = os.path.join(self.model_dir, 'checkpoints', f"{checkpoint_name}.pt")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'model_type': self.model_type,
            'config': self.config.__dict__,
            'experiment_name': self.experiment_name
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Training checkpoint saved to {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None):
        """Load training checkpoint and restore optimizer state"""
        print(f"Loading training checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Return training metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {})
        }

        print(f"Checkpoint loaded - Epoch: {metadata['epoch']}, Step: {metadata['step']}")

        return model, optimizer, metadata

    def get_model_info(self, model) -> Dict[str, Any]:
        """Get information about the model"""
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            'model_type': self.model_type,
            'total_parameters': num_params,
            'trainable_parameters': num_trainable_params,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'max_length': self.config.max_length
        }

        return info
