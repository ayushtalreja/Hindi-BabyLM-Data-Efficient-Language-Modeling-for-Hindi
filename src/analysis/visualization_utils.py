"""
Visualization Utilities for Hindi BabyLM

Specialized plotting functions for thesis-quality figures:
- Learning rate schedules
- Gradient norm tracking
- Attention weights visualization
- Token distribution analysis
- Model size comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Publication-quality settings
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


class ThesisPlotter:
    """
    Create publication-quality plots for thesis

    Features:
    - Consistent styling
    - Proper legends and labels
    - High-resolution output
    - LaTeX-compatible formatting
    """

    def __init__(self, style: str = 'thesis'):
        """
        Initialize plotter

        Args:
            style: Style preset ('thesis', 'presentation', 'paper')
        """
        self.style = style
        self._apply_style()

    def _apply_style(self):
        """Apply style settings"""
        if self.style == 'thesis':
            plt.rcParams['figure.figsize'] = (8, 5)
            plt.rcParams['font.size'] = 11
        elif self.style == 'presentation':
            plt.rcParams['figure.figsize'] = (12, 7)
            plt.rcParams['font.size'] = 14
        elif self.style == 'paper':
            plt.rcParams['figure.figsize'] = (6, 4)
            plt.rcParams['font.size'] = 9

    def plot_learning_rate_schedule(self, steps: List[int], lrs: List[float],
                                   title: str = "Learning Rate Schedule",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot learning rate schedule

        Args:
            steps: Training steps
            lrs: Learning rates
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(steps, lrs, linewidth=2, color='#2E86AB')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_gradient_norms(self, epochs: List[int], grad_norms: List[float],
                           title: str = "Gradient Norms During Training",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot gradient norms over training

        Args:
            epochs: Epoch numbers
            grad_norms: Gradient norm values
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(epochs, grad_norms, linewidth=2, color='#A23B72', marker='o', markersize=4)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Clipping Threshold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_multi_run_comparison(self, data: Dict[str, Dict[str, List[float]]],
                                 metric_name: str = "Accuracy",
                                 title: str = "Multi-Run Comparison",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple runs with error bars

        Args:
            data: Dict of {experiment: {run_name: values}}
            metric_name: Name of metric
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        experiments = list(data.keys())
        x_pos = np.arange(len(experiments))
        width = 0.8 / max(len(list(data.values())[0].keys()), 1)

        colors = plt.cm.Set2(np.linspace(0, 1, len(list(data.values())[0].keys())))

        for i, (run_name, color) in enumerate(zip(list(data.values())[0].keys(), colors)):
            means = []
            stds = []

            for exp in experiments:
                values = data[exp].get(run_name, [])
                means.append(np.mean(values))
                stds.append(np.std(values))

            ax.bar(x_pos + i * width, means, width, yerr=stds,
                  label=run_name, color=color, alpha=0.8, capsize=5)

        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + width * (len(list(data.values())[0].keys()) - 1) / 2)
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_performance_vs_model_size(self, model_sizes: List[int],
                                      accuracies: List[float],
                                      model_names: List[str],
                                      title: str = "Performance vs Model Size",
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot performance vs model size

        Args:
            model_sizes: Model sizes in millions of parameters
            accuracies: Corresponding accuracies
            model_names: Model names for labels
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(model_sizes)))

        for i, (size, acc, name, color) in enumerate(zip(model_sizes, accuracies, model_names, colors)):
            ax.scatter(size, acc, s=200, color=color, alpha=0.7, edgecolors='black', linewidth=1.5)
            ax.annotate(name, (size, acc), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')

        # Fit and plot trend line
        z = np.polyfit(model_sizes, accuracies, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(model_sizes), max(model_sizes), 100)
        ax.plot(x_smooth, p(x_smooth), '--', color='gray', alpha=0.5, linewidth=2, label='Trend')

        ax.set_xlabel('Model Size (M parameters)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_token_distribution(self, token_counts: Dict[str, int],
                                top_n: int = 20,
                                title: str = "Token Distribution",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot token distribution

        Args:
            token_counts: Dict of token -> count
            top_n: Number of top tokens to show
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        # Sort by count
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        tokens, counts = zip(*sorted_tokens)

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(tokens)))
        bars = ax.barh(range(len(tokens)), counts, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=10)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')

        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + max(counts) * 0.01, i, f'{count:,}',
                   va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             title: str = "Confusion Matrix",
                             normalize: bool = True,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix
            class_names: Class names
            title: Plot title
            normalize: Whether to normalize
            save_path: Save path

        Returns:
            Figure object
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
              yticks=np.arange(cm.shape[0]),
              xticklabels=class_names,
              yticklabels=class_names,
              title=title,
              ylabel='True Label',
              xlabel='Predicted Label')

        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_layer_wise_probe_results(self, layer_results: Dict[int, float],
                                     probe_name: str = "Probe",
                                     title: str = "Layer-wise Probe Accuracy",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot layer-wise probing results

        Args:
            layer_results: Dict of layer -> accuracy
            probe_name: Name of probe
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        layers = sorted(layer_results.keys())
        accuracies = [layer_results[layer] for layer in layers]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(layers, accuracies, marker='o', markersize=8, linewidth=2,
               color='#E63946', label=probe_name)

        # Highlight best layer
        best_layer = max(layer_results, key=layer_results.get)
        best_acc = layer_results[best_layer]
        ax.scatter([best_layer], [best_acc], s=300, color='gold',
                  edgecolors='black', linewidth=2, zorder=5, label=f'Best: Layer {best_layer}')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(layers)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def create_figure_grid(self, plots: List[Tuple[str, callable, Dict]],
                          ncols: int = 2,
                          figsize: Tuple[int, int] = (14, 10),
                          title: str = "Analysis Summary",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create grid of multiple plots

        Args:
            plots: List of (title, plot_function, kwargs) tuples
            ncols: Number of columns
            figsize: Figure size
            title: Overall title
            save_path: Save path

        Returns:
            Figure object
        """
        nrows = int(np.ceil(len(plots) / ncols))

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows, ncols, figure=fig, hspace=0.3, wspace=0.3)

        for i, (plot_title, plot_func, kwargs) in enumerate(plots):
            row = i // ncols
            col = i % ncols
            ax = fig.add_subplot(gs[row, col])

            # Call plot function with axis
            plot_func(ax=ax, **kwargs)
            ax.set_title(plot_title, fontsize=11, fontweight='bold')

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure grid to {save_path}")

        return fig


# Convenience functions
def quick_training_plot(train_losses: List[float], val_losses: List[float],
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Quick training/validation loss plot

    Args:
        train_losses: Training losses
        val_losses: Validation losses
        save_path: Save path

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = list(range(1, len(train_losses) + 1))

    ax.plot(epochs, train_losses, marker='o', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, marker='s', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
