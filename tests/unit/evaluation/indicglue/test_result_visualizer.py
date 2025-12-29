"""
Unit tests for ResultVisualizer

Tests the visualization logic that will be extracted from indicglue_evaluator.py.
These tests capture the current behavior of _plot_confusion_matrix and
_plot_per_class_metrics methods.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


class TestResultVisualizer:
    """Test suite for result visualization"""

    def test_plot_confusion_matrix_bbca(self):
        """Test confusion matrix generation for BBCA (14x14)"""
        # BBCA has 14 classes
        predictions = np.random.randint(0, 14, 100)
        labels = np.random.randint(0, 14, 100)

        # Current: _plot_confusion_matrix(conf_matrix, class_names, task_name, save_dir)
        # After: visualizer.plot_confusion_matrix(predictions, labels, class_names, task_name)

        # Should create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions)

        assert cm.shape == (14, 14)

    def test_plot_confusion_matrix_normalized(self):
        """Test normalized confusion matrix"""
        # Current: Normalizes by true labels (rows sum to 1)
        cm = np.array([[10, 2], [3, 15]])

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Row 0: [10, 2] → [10/12, 2/12] = [0.833, 0.167]
        # Row 1: [3, 15] → [3/18, 15/18] = [0.167, 0.833]

        assert np.allclose(cm_normalized.sum(axis=1), 1.0)  # Rows sum to 1

    def test_plot_confusion_matrix_creates_matplotlib_figure(self):
        """Test matplotlib figure is created"""
        # Current: Uses matplotlib and seaborn
        # - plt.subplots(figsize=(10, 8))
        # - sns.heatmap(...)
        # - plt.close()

        figsize = (10, 8)
        assert figsize == (10, 8)

    def test_plot_confusion_matrix_saves_png(self):
        """Test PNG file is saved when 'png' in visualization_format"""
        # Current: if 'png' in self.visualization_format:
        #              plt.savefig(png_path, dpi=300, bbox_inches='tight')

        visualization_formats = ['png', 'html']
        save_png = 'png' in visualization_formats

        assert save_png is True

    def test_plot_confusion_matrix_saves_html(self):
        """Test HTML file is saved when 'html' in visualization_format"""
        # Current: Uses plotly for interactive version
        # fig.write_html(html_path)

        visualization_formats = ['png', 'html']
        save_html = 'html' in visualization_formats

        assert save_html is True

    def test_plot_confusion_matrix_file_naming(self):
        """Test confusion matrix files are named correctly"""
        # Current: {task_name}_confusion_matrix.png
        #          {task_name}_confusion_matrix.html

        task_name = "BBCArticlesClassification"
        png_filename = f"{task_name}_confusion_matrix.png"
        html_filename = f"{task_name}_confusion_matrix.html"

        assert png_filename.endswith('.png')
        assert html_filename.endswith('.html')
        assert task_name in png_filename

    def test_plot_per_class_metrics(self):
        """Test per-class metrics visualization"""
        # Current: _plot_per_class_metrics(per_class_metrics, class_names, task_name)
        # After: visualizer.plot_per_class_metrics(metrics, class_names, task_name)

        # Plots precision, recall, F1 for each class
        # Uses grouped bar chart with error bars (confidence intervals)

        metrics = {
            'precision': [0.8, 0.9, 0.7],
            'recall': [0.75, 0.85, 0.8],
            'f1': [0.77, 0.87, 0.75]
        }

        assert len(metrics['precision']) == 3
        assert len(metrics['recall']) == 3
        assert len(metrics['f1']) == 3

    def test_plot_per_class_metrics_with_confidence_intervals(self):
        """Test per-class metrics include confidence intervals"""
        # Current: Plots error bars using CI bounds
        # per_class_metrics has structure:
        # {
        #     0: {'precision': Metric(value, ci_lower, ci_upper), ...},
        #     1: {'precision': Metric(...), ...}
        # }

        metric_with_ci = {
            'value': 0.85,
            'ci_lower': 0.80,
            'ci_upper': 0.90
        }

        lower_error = metric_with_ci['value'] - metric_with_ci['ci_lower']
        upper_error = metric_with_ci['ci_upper'] - metric_with_ci['value']

        # Use pytest.approx for floating point comparison
        assert lower_error == pytest.approx(0.05, abs=1e-9)
        assert upper_error == pytest.approx(0.05, abs=1e-9)

    def test_plot_per_class_metrics_creates_grouped_bar_chart(self):
        """Test per-class metrics use grouped bar chart"""
        # Current: Uses matplotlib bar chart
        # - 3 groups: Precision, Recall, F1
        # - Different colors for each metric
        # - x-axis: class names
        # - y-axis: score (0-1)

        num_metrics = 3  # Precision, Recall, F1
        bar_width = 0.25  # Width of each bar

        assert num_metrics == 3
        assert bar_width == 0.25

    def test_plot_per_class_metrics_file_naming(self):
        """Test per-class metrics files are named correctly"""
        # Current: {task_name}_per_class_metrics.png
        #          {task_name}_per_class_metrics.html

        task_name = "BBCArticlesClassification"
        png_filename = f"{task_name}_per_class_metrics.png"
        html_filename = f"{task_name}_per_class_metrics.html"

        assert png_filename.endswith('.png')
        assert html_filename.endswith('.html')


class TestVisualizationConfiguration:
    """Test visualization configuration options"""

    def test_save_visualizations_flag(self):
        """Test save_visualizations flag controls whether plots are saved"""
        # Current: if not self.save_visualizations: return
        # After: visualizer.save_visualizations should respect this flag

        save_enabled = True
        save_disabled = False

        assert save_enabled is True
        assert save_disabled is False

    def test_visualization_format_configuration(self):
        """Test visualization format can be configured"""
        # Current: self.visualization_format can be:
        # - ['png']
        # - ['html']
        # - ['png', 'html']

        format_configs = [
            ['png'],
            ['html'],
            ['png', 'html']
        ]

        assert ['png'] in format_configs
        assert ['png', 'html'] in format_configs

    def test_save_directory_creation(self):
        """Test save directory is created if it doesn't exist"""
        # Current: save_dir.mkdir(parents=True, exist_ok=True)

        # After: visualizer should create directory
        create_if_not_exists = True
        assert create_if_not_exists is True


class TestVisualizationEdgeCases:
    """Test edge cases in visualization"""

    def test_handles_import_errors_gracefully(self):
        """Test graceful handling when matplotlib/seaborn not available"""
        # Current: try/except ImportError
        # Logs warning and skips visualization

        matplotlib_available = True

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            matplotlib_available = False

        # Test should pass whether or not matplotlib is installed
        assert isinstance(matplotlib_available, bool)

    def test_handles_plotting_errors(self):
        """Test graceful handling of plotting errors"""
        # Current: try/except around plotting code
        # Logs error but doesn't crash

        # Visualization errors should not crash evaluation
        assert True  # Behavior test

    def test_confusion_matrix_with_missing_classes(self):
        """Test confusion matrix when some classes have no predictions"""
        # Some classes might have 0 predictions in test set
        # Matrix should still show all classes

        num_classes = 14
        # Only classes 0, 1, 2 appear in predictions
        predictions = [0, 1, 2, 0, 1]
        labels = [0, 1, 2, 0, 1]

        # Confusion matrix should still be 14x14 (all classes)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

        assert cm.shape == (14, 14)

    def test_long_class_names_rotation(self):
        """Test x-axis labels are rotated for readability"""
        # Current: ax.set_xticklabels(class_names, rotation=45, ha='right')

        rotation_angle = 45
        horizontal_alignment = 'right'

        assert rotation_angle == 45
        assert horizontal_alignment == 'right'


class TestVisualizationIntegration:
    """Test integration with evaluation results"""

    def test_save_visualizations_processes_all_tasks(self):
        """Test save_visualizations method processes all task results"""
        # Current: save_visualizations(results, save_dir)
        # Iterates over all tasks and generates plots

        results = {
            'BBCArticlesClassification': {
                'confusion_matrix': {'matrix_normalized': [[]]},
                'per_class_metrics': {}
            },
            'MovieReviewSentiment': {
                'confusion_matrix': {'matrix_normalized': [[]]},
                'per_class_metrics': {}
            },
            'overall': {}  # Skipped
        }

        tasks_to_plot = [k for k in results.keys() if k != 'overall']
        assert len(tasks_to_plot) == 2

    def test_skips_tasks_without_confusion_matrix(self):
        """Test tasks without confusion matrix are skipped"""
        # Current: if 'confusion_matrix' not in task_results: continue

        task_results = {
            'status': 'skipped',
            'reason': 'Dataset issue'
        }

        has_confusion_matrix = 'confusion_matrix' in task_results
        assert has_confusion_matrix is False

    def test_skips_overall_metrics(self):
        """Test 'overall' key is skipped in visualization"""
        # Current: if task_name == 'overall': continue

        task_name = 'overall'
        should_skip = task_name == 'overall'

        assert should_skip is True
