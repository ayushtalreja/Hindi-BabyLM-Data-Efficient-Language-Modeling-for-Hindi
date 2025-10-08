import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import json
import os

class ResultsAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.results_data = self.load_all_results()
    
    def load_all_results(self) -> List[Dict]:
        """Load results from all experiments"""
        all_results = []
        
        for experiment_dir in os.listdir(self.results_dir):
            experiment_path = os.path.join(self.results_dir, experiment_dir)
            if os.path.isdir(experiment_path):
                results_file = os.path.join(experiment_path, 'evaluation_results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                        results['experiment_name'] = experiment_dir
                        all_results.append(results)
        
        return all_results
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Create DataFrame for easy comparison across experiments"""
        comparison_data = []
        
        for result in self.results_data:
            row = {
                'experiment': result['experiment_name'],
                'indicglue_avg': result['summary']['overall_scores'].get('indicglue_avg', 0),
                'multiblimp_accuracy': result['summary']['overall_scores'].get('multiblimp_accuracy', 0),
                'morphological_avg': result['summary']['overall_scores'].get('morphological_avg', 0),
            }
            
            # Add individual IndicGLUE scores
            for task, scores in result.get('indicglue', {}).items():
                if isinstance(scores, dict) and 'accuracy' in scores:
                    row[f'indicglue_{task}'] = scores['accuracy']
            
            # Add individual MultiBLiMP scores
            for phenomenon, scores in result.get('multiblimp', {}).items():
                if isinstance(scores, dict) and 'accuracy' in scores and phenomenon != 'overall':
                    row[f'multiblimp_{phenomenon}'] = scores['accuracy']
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_tokenization_comparison(self, df: pd.DataFrame, save_path: str):
        """Plot comparison of tokenization strategies"""
        # Filter tokenization experiments
        tokenization_df = df[df['experiment'].str.contains('tokenization')]
        
        if len(tokenization_df) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['indicglue_avg', 'multiblimp_accuracy', 'morphological_avg']
        titles = ['IndicGLUE Average', 'MultiBLiMP Accuracy', 'Morphological Probes Average']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            # Extract tokenizer names
            tokenization_df['tokenizer'] = tokenization_df['experiment'].str.replace('tokenization_', '')
            
            sns.barplot(data=tokenization_df, x='tokenizer', y=metric, ax=ax)
            ax.set_title(title)
            ax.set_ylabel('Accuracy')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'tokenization_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_architecture_comparison(self, df: pd.DataFrame, save_path: str):
        """Plot comparison of model architectures"""
        architecture_df = df[df['experiment'].str.contains('architecture')]
        
        if len(architecture_df) == 0:
            return
        
        # Extract architecture names
        architecture_df['architecture'] = architecture_df['experiment'].str.replace('architecture_', '')
        
        # Create grouped bar plot
        metrics = ['indicglue_avg', 'multiblimp_accuracy', 'morphological_avg']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(architecture_df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, architecture_df[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Architecture Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(architecture_df['architecture'])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'architecture_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_curriculum_comparison(self, df: pd.DataFrame, save_path: str):
        """Plot comparison of curriculum learning strategies"""
        curriculum_df = df[df['experiment'].str.contains('curriculum')]
        
        if len(curriculum_df) == 0:
            return
        
        curriculum_df['strategy'] = curriculum_df['experiment'].str.replace('curriculum_', '')
        
        # Create radar chart for curriculum comparison
        metrics = ['indicglue_avg', 'multiblimp_accuracy', 'morphological_avg']
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for _, row in curriculum_df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['strategy'])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Curriculum Learning Strategy Comparison', y=1.08)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'curriculum_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_report(self, save_path: str):
        """Generate comprehensive analysis report"""
        df = self.create_comparison_dataframe()
        
        # Create visualizations
        os.makedirs(save_path, exist_ok=True)
        
        self.plot_tokenization_comparison(df, save_path)
        self.plot_architecture_comparison(df, save_path)
        self.plot_curriculum_comparison(df, save_path)
        
        # Generate summary statistics
        summary_stats = df.describe()
        summary_stats.to_csv(os.path.join(save_path, 'summary_statistics.csv'))
        
        # Find best performing configurations
        best_configs = {
            'indicglue': df.loc[df['indicglue_avg'].idxmax()]['experiment'],
            'multiblimp': df.loc[df['multiblimp_accuracy'].idxmax()]['experiment'],
            'morphological': df.loc[df['morphological_avg'].idxmax()]['experiment']
        }
        
        # Save best configurations
        with open(os.path.join(save_path, 'best_configurations.json'), 'w') as f:
            json.dump(best_configs, f, indent=2)
        
        print(f"Analysis report generated in: {save_path}")
        print("Best configurations:")
        for metric, config in best_configs.items():
            print(f"  {metric}: {config}")
