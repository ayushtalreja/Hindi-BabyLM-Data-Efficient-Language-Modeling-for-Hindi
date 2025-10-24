"""
Results Analysis Tools for Hindi BabyLM

This module provides comprehensive tools for analyzing experimental results:
- Training curve visualization
- Statistical significance testing
- Model comparison
- Performance metrics aggregation
- LaTeX table generation for thesis

Features:
- Multiple experiments comparison
- Statistical tests (t-test, Wilcoxon, bootstrap CI)
- Beautiful publication-ready plots
- Automatic LaTeX table generation
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class ResultsAnalyzer:
    """
    Comprehensive results analysis and visualization

    Features:
    - Load results from multiple experiments
    - Compare models statistically
    - Generate publication-ready plots
    - Create LaTeX tables
    - Compute aggregated metrics
    """

    def __init__(self, results_dir: Union[str, Path] = "results"):
        """
        Initialize results analyzer

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.loaded_experiments = set()

        if not self.results_dir.exists():
            logger.warning(f"Results directory not found: {results_dir}")

    def load_experiment(self, experiment_name: str) -> Dict:
        """
        Load results from a single experiment

        Args:
            experiment_name: Name of experiment directory

        Returns:
            Dictionary with experiment results
        """
        exp_dir = self.results_dir / experiment_name

        if not exp_dir.exists():
            raise ValueError(f"Experiment not found: {experiment_name}")

        results = {}

        # Load metadata
        metadata_file = exp_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                results['metadata'] = json.load(f)

        # Load training summary
        training_file = exp_dir / "training_summary.json"
        if training_file.exists():
            with open(training_file, 'r') as f:
                results['training'] = json.load(f)

        # Load evaluation results
        eval_file = exp_dir / "evaluation_results.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                results['evaluation'] = json.load(f)

        # Load config
        config_file = exp_dir / "config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                results['config'] = yaml.safe_load(f)

        self.experiments[experiment_name] = results
        self.loaded_experiments.add(experiment_name)

        logger.info(f"Loaded experiment: {experiment_name}")
        return results

    def load_all_experiments(self) -> int:
        """
        Load all experiments from results directory

        Returns:
            Number of experiments loaded
        """
        if not self.results_dir.exists():
            return 0

        count = 0
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name not in self.loaded_experiments:
                try:
                    self.load_experiment(exp_dir.name)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load {exp_dir.name}: {e}")

        logger.info(f"Loaded {count} experiments")
        return count

    def get_training_history(self, experiment_name: str) -> pd.DataFrame:
        """
        Get training history as DataFrame

        Args:
            experiment_name: Name of experiment

        Returns:
            DataFrame with training history
        """
        if experiment_name not in self.experiments:
            self.load_experiment(experiment_name)

        training_data = self.experiments[experiment_name].get('training', {})
        history = training_data.get('history', [])

        if not history:
            return pd.DataFrame()

        return pd.DataFrame(history)

    def plot_training_curves(self, experiment_names: Optional[List[str]] = None,
                            metrics: List[str] = ['loss', 'perplexity'],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training curves for one or more experiments

        Args:
            experiment_names: List of experiments to plot (None for all)
            metrics: Metrics to plot
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if experiment_names is None:
            experiment_names = list(self.experiments.keys())

        if not experiment_names:
            raise ValueError("No experiments to plot")

        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

        if num_metrics == 1:
            axes = [axes]

        for metric, ax in zip(metrics, axes):
            for exp_name in experiment_names:
                history = self.get_training_history(exp_name)

                if metric in history.columns:
                    ax.plot(history['epoch'], history[metric],
                           label=exp_name, marker='o', markersize=4)

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'Training {metric.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_evaluation_comparison(self, experiment_names: Optional[List[str]] = None,
                                   eval_type: str = 'indicglue',
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot evaluation comparison across experiments

        Args:
            experiment_names: List of experiments to compare
            eval_type: Type of evaluation ('indicglue', 'multiblimp', 'probes')
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if experiment_names is None:
            experiment_names = list(self.experiments.keys())

        # Collect evaluation results
        results_data = defaultdict(dict)

        for exp_name in experiment_names:
            if exp_name not in self.experiments:
                continue

            eval_results = self.experiments[exp_name].get('evaluation', {})

            if eval_type == 'indicglue':
                for task, task_results in eval_results.get('indicglue', {}).items():
                    if isinstance(task_results, dict) and 'accuracy' in task_results:
                        results_data[task][exp_name] = task_results['accuracy']

            elif eval_type == 'multiblimp':
                for phenomenon, phenom_results in eval_results.get('multiblimp', {}).items():
                    if isinstance(phenom_results, dict) and 'accuracy' in phenom_results:
                        results_data[phenomenon][exp_name] = phenom_results['accuracy']

        if not results_data:
            logger.warning(f"No {eval_type} results found")
            return plt.figure()

        # Create DataFrame
        df = pd.DataFrame(results_data).T

        # Plot
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))

        df.plot(kind='barh', ax=ax, width=0.8)

        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Task/Phenomenon')
        ax.set_title(f'{eval_type.upper()} Evaluation Comparison')
        ax.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def compare_models_statistically(self, exp1: str, exp2: str,
                                    metric: str = 'accuracy',
                                    eval_type: str = 'indicglue') -> Dict:
        """
        Statistical comparison between two models

        Args:
            exp1: First experiment name
            exp2: Second experiment name
            metric: Metric to compare
            eval_type: Type of evaluation

        Returns:
            Dictionary with statistical test results
        """
        # Collect results
        results1 = []
        results2 = []

        eval_results1 = self.experiments[exp1].get('evaluation', {}).get(eval_type, {})
        eval_results2 = self.experiments[exp2].get('evaluation', {}).get(eval_type, {})

        for task in eval_results1.keys():
            if task in eval_results2:
                if isinstance(eval_results1[task], dict) and metric in eval_results1[task]:
                    results1.append(eval_results1[task][metric])
                    results2.append(eval_results2[task][metric])

        if len(results1) < 2:
            return {'error': 'Insufficient data for statistical testing'}

        results1 = np.array(results1)
        results2 = np.array(results2)

        # Perform tests
        comparison = {}

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(results1, results2)
        comparison['t_test'] = {
            'statistic': float(t_stat),
            'p_value': float(t_pval),
            'significant': t_pval < 0.05
        }

        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pval = stats.wilcoxon(results1, results2)
        comparison['wilcoxon'] = {
            'statistic': float(w_stat),
            'p_value': float(w_pval),
            'significant': w_pval < 0.05
        }

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(results1) + np.var(results2)) / 2)
        cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std
        comparison['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': self._interpret_cohens_d(cohens_d)
        }

        # Bootstrap confidence intervals
        comparison['bootstrap_ci'] = self._bootstrap_difference(results1, results2)

        # Summary statistics
        comparison['summary'] = {
            'exp1_mean': float(np.mean(results1)),
            'exp1_std': float(np.std(results1)),
            'exp2_mean': float(np.mean(results2)),
            'exp2_std': float(np.std(results2)),
            'difference': float(np.mean(results1) - np.mean(results2)),
            'num_tasks': len(results1)
        }

        return comparison

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"

    def _bootstrap_difference(self, sample1: np.ndarray, sample2: np.ndarray,
                             n_bootstrap: int = 10000, ci: float = 0.95) -> Dict:
        """
        Bootstrap confidence interval for difference between means

        Args:
            sample1: First sample
            sample2: Second sample
            n_bootstrap: Number of bootstrap samples
            ci: Confidence interval level

        Returns:
            Dictionary with CI bounds
        """
        differences = []

        for _ in range(n_bootstrap):
            idx1 = np.random.choice(len(sample1), len(sample1), replace=True)
            idx2 = np.random.choice(len(sample2), len(sample2), replace=True)

            diff = np.mean(sample1[idx1]) - np.mean(sample2[idx2])
            differences.append(diff)

        differences = np.array(differences)
        alpha = (1 - ci) / 2

        return {
            'mean': float(np.mean(differences)),
            'lower': float(np.percentile(differences, alpha * 100)),
            'upper': float(np.percentile(differences, (1 - alpha) * 100)),
            'ci_level': ci
        }

    def generate_latex_table(self, experiment_names: Optional[List[str]] = None,
                            eval_type: str = 'indicglue',
                            metric: str = 'accuracy',
                            caption: str = "Model Comparison",
                            label: str = "tab:comparison",
                            save_path: Optional[str] = None) -> str:
        """
        Generate LaTeX table for thesis

        Args:
            experiment_names: Experiments to include
            eval_type: Type of evaluation
            metric: Metric to display
            caption: Table caption
            label: LaTeX label
            save_path: Path to save .tex file

        Returns:
            LaTeX table string
        """
        if experiment_names is None:
            experiment_names = list(self.experiments.keys())

        # Collect results
        results_data = defaultdict(dict)

        for exp_name in experiment_names:
            eval_results = self.experiments[exp_name].get('evaluation', {}).get(eval_type, {})

            for task, task_results in eval_results.items():
                if isinstance(task_results, dict) and metric in task_results:
                    results_data[task][exp_name] = task_results[metric]

        # Create DataFrame
        df = pd.DataFrame(results_data).T

        # Generate LaTeX
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\begin{tabular}{l" + "c" * len(experiment_names) + "}\n"
        latex += "\\toprule\n"

        # Header
        latex += "Task & " + " & ".join(experiment_names) + " \\\\\n"
        latex += "\\midrule\n"

        # Data rows
        for task in df.index:
            row_values = [task]
            for exp in experiment_names:
                value = df.loc[task, exp]
                if pd.notna(value):
                    # Bold best value
                    if value == df.loc[task].max():
                        row_values.append(f"\\textbf{{{value:.3f}}}")
                    else:
                        row_values.append(f"{value:.3f}")
                else:
                    row_values.append("--")

            latex += " & ".join(row_values) + " \\\\\n"

        # Average row
        latex += "\\midrule\n"
        avg_values = ["Average"]
        for exp in experiment_names:
            avg = df[exp].mean()
            avg_values.append(f"{avg:.3f}")
        latex += " & ".join(avg_values) + " \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(latex)
            logger.info(f"Saved LaTeX table to {save_path}")

        return latex

    def generate_summary_report(self, experiment_name: str,
                               save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive summary report for an experiment

        Args:
            experiment_name: Experiment to summarize
            save_path: Path to save report

        Returns:
            Markdown report string
        """
        if experiment_name not in self.experiments:
            self.load_experiment(experiment_name)

        exp = self.experiments[experiment_name]

        report = f"# Experiment Report: {experiment_name}\n\n"

        # Metadata
        if 'metadata' in exp:
            report += "## Metadata\n\n"
            metadata = exp['metadata']
            report += f"- **Timestamp**: {metadata.get('timestamp', 'N/A')}\n"
            report += f"- **Git Commit**: {metadata.get('git_commit', 'N/A')}\n"
            report += f"- **Device**: {metadata.get('device', 'N/A')}\n\n"

        # Training summary
        if 'training' in exp:
            report += "## Training Summary\n\n"
            training = exp['training']
            report += f"- **Epochs**: {training.get('num_epochs', 'N/A')}\n"
            report += f"- **Final Loss**: {training.get('final_loss', 'N/A'):.4f}\n"
            report += f"- **Best Val Loss**: {training.get('best_val_loss', 'N/A'):.4f}\n"
            report += f"- **Training Time**: {training.get('total_time', 'N/A'):.2f}s\n\n"

        # Evaluation results
        if 'evaluation' in exp:
            report += "## Evaluation Results\n\n"
            evaluation = exp['evaluation']

            for eval_type, eval_results in evaluation.items():
                report += f"### {eval_type.upper()}\n\n"

                if isinstance(eval_results, dict):
                    for task, task_results in eval_results.items():
                        if isinstance(task_results, dict) and 'accuracy' in task_results:
                            report += f"- **{task}**: {task_results['accuracy']:.4f}\n"

                report += "\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved report to {save_path}")

        return report


# Convenience functions
def analyze_experiments(results_dir: str = "results") -> ResultsAnalyzer:
    """
    Create analyzer and load all experiments

    Args:
        results_dir: Results directory

    Returns:
        ResultsAnalyzer instance
    """
    analyzer = ResultsAnalyzer(results_dir)
    analyzer.load_all_experiments()
    return analyzer


def quick_comparison(exp1: str, exp2: str, results_dir: str = "results") -> Dict:
    """
    Quick statistical comparison between two experiments

    Args:
        exp1: First experiment
        exp2: Second experiment
        results_dir: Results directory

    Returns:
        Comparison results
    """
    analyzer = ResultsAnalyzer(results_dir)
    analyzer.load_experiment(exp1)
    analyzer.load_experiment(exp2)
    return analyzer.compare_models_statistically(exp1, exp2)
