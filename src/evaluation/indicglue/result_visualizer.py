"""
Result visualization utilities for IndicGLUE evaluation.

Generates confusion matrices and per-class metric plots for evaluation results.
Extracted from indicglue_evaluator.py to improve testability and maintainability.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """
    Generates and saves evaluation visualizations.

    Supports:
    - Confusion matrices (normalized and raw)
    - Per-class metrics (precision/recall/F1)
    - Multiple output formats (PNG, HTML, etc.)
    """

    def __init__(
        self,
        save_visualizations: bool = True,
        visualization_formats: Optional[List[str]] = None
    ):
        """
        Initialize ResultVisualizer.

        Args:
            save_visualizations: Whether to save visualizations to disk
            visualization_formats: List of formats to save ('png', 'html', 'svg', etc.)
                                  Defaults to ['png', 'html']
        """
        self.save_visualizations = save_visualizations
        self.visualization_formats = visualization_formats or ['png', 'html']

        # Configure matplotlib for better output
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.figsize'] = (10, 8)

    def plot_confusion_matrix(
        self,
        predictions: List[int],
        labels: List[int],
        class_names: Optional[List[str]] = None,
        task_name: str = "",
        output_path: Optional[Path] = None,
        normalize: bool = True
    ) -> Optional[matplotlib.figure.Figure]:
        """
        Plot confusion matrix.

        Args:
            predictions: Predicted labels (list of integers)
            labels: True labels (list of integers)
            class_names: Optional class names for axis labels
            task_name: Task name for title
            output_path: Path to save figure (without extension)
            normalize: Whether to normalize confusion matrix rows

        Returns:
            matplotlib Figure object, or None if plotting fails
        """
        try:
            from sklearn.metrics import confusion_matrix

            # Compute confusion matrix
            cm = confusion_matrix(labels, predictions)

            # Normalize if requested
            if normalize:
                cm_plot = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
                fmt = '.2f'
                title_suffix = ' (Normalized)'
            else:
                cm_plot = cm
                fmt = 'd'
                title_suffix = ''

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot heatmap
            sns.heatmap(
                cm_plot,
                annot=True,
                fmt=fmt,
                cmap='Blues',
                xticklabels=class_names or range(len(cm)),
                yticklabels=class_names or range(len(cm)),
                ax=ax,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'}
            )

            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'Confusion Matrix: {task_name}{title_suffix}', fontsize=14, fontweight='bold')

            # Rotate labels if there are many classes
            if len(cm) > 10:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
                plt.setp(ax.get_yticklabels(), rotation=0)

            plt.tight_layout()

            # Save if requested
            if output_path and self.save_visualizations:
                self._save_figure(fig, output_path, 'confusion_matrix')

            return fig

        except Exception as e:
            logger.error(f"Failed to plot confusion matrix for {task_name}: {e}")
            return None
        finally:
            plt.close('all')  # Clean up to avoid memory leaks

    def plot_per_class_metrics(
        self,
        metrics: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        task_name: str = "",
        output_path: Optional[Path] = None
    ) -> Optional[matplotlib.figure.Figure]:
        """
        Plot per-class precision, recall, and F1 scores.

        Args:
            metrics: Dict with 'precision', 'recall', 'f1' arrays (per-class scores)
            class_names: Optional class names for x-axis labels
            task_name: Task name for title
            output_path: Path to save figure (without extension)

        Returns:
            matplotlib Figure object, or None if plotting fails
        """
        try:
            # Extract per-class metrics
            precision = metrics.get('precision', [])
            recall = metrics.get('recall', [])
            f1 = metrics.get('f1', [])

            if not precision or not recall or not f1:
                logger.warning(f"Missing per-class metrics for {task_name}")
                return None

            # Generate class names if not provided
            if not class_names:
                class_names = [f"Class {i}" for i in range(len(precision))]

            # Create bar chart
            x = np.arange(len(class_names))
            width = 0.25

            fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.8), 6))

            # Plot bars
            ax.bar(x - width, precision, width, label='Precision', color='#1f77b4', alpha=0.8)
            ax.bar(x, recall, width, label='Recall', color='#ff7f0e', alpha=0.8)
            ax.bar(x + width, f1, width, label='F1 Score', color='#2ca02c', alpha=0.8)

            # Customize plot
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'Per-Class Metrics: {task_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.legend(loc='best', fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1.05)

            # Add value labels on bars (if not too many classes)
            if len(class_names) <= 15:
                for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
                    ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
                    ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
                    ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()

            # Save if requested
            if output_path and self.save_visualizations:
                self._save_figure(fig, output_path, 'per_class_metrics')

            return fig

        except Exception as e:
            logger.error(f"Failed to plot per-class metrics for {task_name}: {e}")
            return None
        finally:
            plt.close('all')  # Clean up to avoid memory leaks

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        task_name: str = "",
        output_path: Optional[Path] = None
    ) -> Optional[matplotlib.figure.Figure]:
        """
        Plot training and validation loss/accuracy curves.

        Args:
            history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
            task_name: Task name for title
            output_path: Path to save figure (without extension)

        Returns:
            matplotlib Figure object, or None if plotting fails
        """
        try:
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            train_acc = history.get('train_acc', [])
            val_acc = history.get('val_acc', [])

            if not train_loss and not train_acc:
                logger.warning(f"No training history to plot for {task_name}")
                return None

            epochs = range(1, len(train_loss) + 1) if train_loss else range(1, len(train_acc) + 1)

            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Plot loss
            if train_loss:
                ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
                if val_loss:
                    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
                ax1.set_xlabel('Epoch', fontsize=12)
                ax1.set_ylabel('Loss', fontsize=12)
                ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
                ax1.legend(loc='best')
                ax1.grid(alpha=0.3)

            # Plot accuracy
            if train_acc:
                ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
                if val_acc:
                    ax2.plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('Accuracy', fontsize=12)
                ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
                ax2.legend(loc='best')
                ax2.grid(alpha=0.3)

            fig.suptitle(f'Training History: {task_name}', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            # Save if requested
            if output_path and self.save_visualizations:
                self._save_figure(fig, output_path, 'training_history')

            return fig

        except Exception as e:
            logger.error(f"Failed to plot training history for {task_name}: {e}")
            return None
        finally:
            plt.close('all')  # Clean up to avoid memory leaks

    def _save_figure(self, fig: matplotlib.figure.Figure, output_path: Path, suffix: str):
        """
        Save figure in multiple formats.

        Args:
            fig: matplotlib Figure object
            output_path: Base path for saving (without extension)
            suffix: Suffix to add to filename (e.g., 'confusion_matrix')
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build filename with suffix
            base_name = output_path.stem
            if suffix and suffix not in base_name:
                filename = f"{base_name}_{suffix}"
            else:
                filename = base_name

            # Save in each requested format
            for fmt in self.visualization_formats:
                save_path = output_path.parent / f"{filename}.{fmt}"

                if fmt == 'html':
                    # For HTML, convert to interactive plotly or save as static HTML
                    # For now, we'll skip HTML or save as PNG embedded in HTML
                    logger.debug(f"Skipping HTML format (not implemented)")
                else:
                    fig.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
                    logger.info(f"Saved visualization to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save figure: {e}")

    def create_summary_report(
        self,
        results: Dict[str, Any],
        task_name: str = "",
        output_path: Optional[Path] = None
    ) -> str:
        """
        Create a text summary report of evaluation results.

        Args:
            results: Dict with evaluation results (accuracy, f1, etc.)
            task_name: Task name
            output_path: Optional path to save report as text file

        Returns:
            Summary report as string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"EVALUATION REPORT: {task_name}")
        lines.append("=" * 60)
        lines.append("")

        # Overall metrics
        if 'accuracy' in results:
            lines.append(f"Accuracy:        {results['accuracy']:.4f}")
        if 'f1_macro' in results:
            lines.append(f"F1 (Macro):      {results['f1_macro']:.4f}")
        if 'f1_weighted' in results:
            lines.append(f"F1 (Weighted):   {results['f1_weighted']:.4f}")

        lines.append("")
        lines.append("-" * 60)

        # Per-class metrics (if available)
        if 'per_class_metrics' in results:
            lines.append("Per-Class Metrics:")
            lines.append("-" * 60)
            per_class = results['per_class_metrics']

            precision = per_class.get('precision', [])
            recall = per_class.get('recall', [])
            f1 = per_class.get('f1', [])
            support = per_class.get('support', [])

            if precision and recall and f1:
                lines.append(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
                lines.append("-" * 60)
                for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
                    class_name = results.get('class_names', [f"Class {i}"])[i] if i < len(results.get('class_names', [])) else f"Class {i}"
                    lines.append(f"{class_name:<20} {p:<12.4f} {r:<12.4f} {f:<12.4f} {int(s):<10}")

        lines.append("=" * 60)

        report = "\n".join(lines)

        # Save to file if requested
        if output_path and self.save_visualizations:
            try:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                report_path = output_path.parent / f"{output_path.stem}_report.txt"
                report_path.write_text(report)
                logger.info(f"Saved report to {report_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")

        return report
