import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import os

class HindiLanguageModelTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'learning_rate': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # For language modeling, labels = input_ids
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            wandb.log({
                'batch_loss': loss.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.size(0)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            'val_loss': avg_loss,
            'perplexity': perplexity.item()
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int):
        """Main training loop"""
        
        # Initialize wandb
        wandb.init(
            project=self.config['project_name'],
            config=self.config
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_dataloader)
            
            # Validation
            val_metrics = self.evaluate(val_dataloader)
            
            # Logging
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_metrics['val_loss'])
            self.metrics['perplexity'].append(val_metrics['perplexity'])
            
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['val_loss'],
                'perplexity': val_metrics['perplexity']
            })
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Perplexity: {val_metrics['perplexity']:.4f}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, val_metrics['val_loss'])
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
