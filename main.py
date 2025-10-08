import argparse
import yaml
from src.data_processing.corpus_builder import CorpusBuilder
from src.tokenization.tokenizer_factory import TokenizerFactory
from src.models.model_factory import ModelFactory
from src.training.trainer import HindiLanguageModelTrainer
from src.evaluation.evaluation_manager import EvaluationManager
from src.utils.experiment_config import ExperimentConfig
import wandb
import os

def main():
    parser = argparse.ArgumentParser(description="Hindi BabyLM Training Pipeline")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--stage', type=str, choices=['data', 'train', 'eval', 'all'], 
                       default='all', help='Which stage to run')
    parser.add_argument('--experiment_name', type=str, help='Name for this experiment')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.load_config(args.config)
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    print(f"Running Hindi BabyLM Pipeline - Stage: {args.stage}")
    print(f"Experiment: {config.experiment_name}")
    
    # Stage 1: Data Processing
    if args.stage in ['data', 'all']:
        print("\n=== STAGE 1: DATA PROCESSING ===")
        corpus_builder = CorpusBuilder(config)
        
        # Download and process data
        raw_data = corpus_builder.collect_all_data()
        processed_data = corpus_builder.process_and_filter(raw_data)
        
        # Create train/val/test splits
        splits = corpus_builder.create_splits(processed_data)
        
        # Save processed data
        corpus_builder.save_splits(splits)
        print("Data processing completed.")
    
    # Stage 2: Training
    if args.stage in ['train', 'all']:
        print("\n=== STAGE 2: MODEL TRAINING ===")
        
        # Load processed data
        corpus_builder = CorpusBuilder(config)
        splits = corpus_builder.load_splits()
        
        # Create tokenizer
        tokenizer_factory = TokenizerFactory(config)
        tokenizer = tokenizer_factory.create_tokenizer(splits['train'])
        
        # Create model
        model_factory = ModelFactory(config)
        model = model_factory.create_model(tokenizer.vocab_size)
        
        # Create trainer
        trainer = HindiLanguageModelTrainer(model, tokenizer, config.__dict__)
        
        # Create data loaders
        train_dataloader = corpus_builder.create_dataloader(splits['train'], tokenizer, 'train')
        val_dataloader = corpus_builder.create_dataloader(splits['val'], tokenizer, 'val')
        
        # Train model
        trainer.train(train_dataloader, val_dataloader, config.num_epochs)
        print("Model training completed.")
    
    # Stage 3: Evaluation  
    if args.stage in ['eval', 'all']:
        print("\n=== STAGE 3: EVALUATION ===")
        
        # Load trained model and tokenizer
        model_factory = ModelFactory(config)
        model = model_factory.load_trained_model(config.experiment_name)
        tokenizer = TokenizerFactory.load_tokenizer(config.experiment_name)
        
        # Run comprehensive evaluation
        evaluator = EvaluationManager(model, tokenizer, config.__dict__)
        results = evaluator.run_comprehensive_evaluation()
        
        print("Evaluation completed.")
        print(f"Results saved to: {evaluator.save_results()}")
    
    print(f"\nHindi BabyLM Pipeline completed successfully!")
    print(f"Experiment: {config.experiment_name}")

if __name__ == "__main__":
    main()