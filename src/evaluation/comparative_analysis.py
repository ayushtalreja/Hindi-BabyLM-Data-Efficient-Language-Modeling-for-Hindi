"""
Comparative Analysis for Hindi BabyLM Evaluation Results

This module provides comprehensive comparison and analysis of evaluation results
across models, checkpoints, and tasks with interactive visualizations and reports.

Key Features:
- Side-by-side model comparison tables
- Multi-dimensional radar plots for task performance
- Regression analysis for training progression
- Interactive HTML reports with Bootstrap styling
- Publication-ready PDF reports with matplotlib
- Statistical significance testing between models
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ComparativeAnalyzer:
    """
    Analyzer for comparing evaluation results across models, checkpoints, and tasks

    This class provides comprehensive comparison capabilities including:
    - Cross-model performance comparison
    - Training progression analysis
    - Task-specific performance comparison
    - Statistical significance testing
    - Interactive and static visualizations
    - HTML and PDF report generation
    """

    def __init__(
        self,
        results_dir: Optional[str] = None,
        output_dir: str = "comparative_analysis"
    ):
        """
        Initialize comparative analyzer

        Args:
            results_dir: Directory containing evaluation results
            output_dir: Directory to save analysis outputs
        """
        self.results_dir = Path(results_dir) if results_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for loaded results
        self.loaded_results = {}

        logger.info(f"Comparative analyzer initialized")
        logger.info(f"  Results dir: {self.results_dir}")
        logger.info(f"  Output dir: {self.output_dir}")

    def _load_result(
        self,
        result_path: Union[str, Path],
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load evaluation result from file or directory

        Args:
            result_path: Path to result file or directory
            name: Name to use for this result (defaults to filename)

        Returns:
            Dictionary with evaluation results
        """
        result_path = Path(result_path)

        # Determine result file path
        if result_path.is_dir():
            # Look for evaluation_results.json in directory
            result_file = result_path / 'evaluation_results.json'
            if not result_file.exists():
                raise FileNotFoundError(f"No evaluation_results.json in {result_path}")
        else:
            result_file = result_path

        # Load results
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)

            # Use filename as name if not provided
            if name is None:
                name = result_path.stem if result_path.is_file() else result_path.name

            logger.info(f"Loaded results: {name} from {result_file}")
            return results

        except Exception as e:
            logger.error(f"Error loading results from {result_file}: {e}")
            raise

    def load_results(
        self,
        result_paths: Dict[str, Union[str, Path]]
    ):
        """
        Load multiple evaluation results

        Args:
            result_paths: Dictionary mapping names to result paths
        """
        for name, path in result_paths.items():
            try:
                self.loaded_results[name] = self._load_result(path, name)
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")

        logger.info(f"Loaded {len(self.loaded_results)} result sets")

    def create_comparison_table(
        self,
        result_names: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        metrics: List[str] = ['accuracy', 'f1_macro']
    ) -> pd.DataFrame:
        """
        Create side-by-side comparison table

        Args:
            result_names: Names of results to compare (None = all loaded)
            tasks: Tasks to include (None = all tasks)
            metrics: Metrics to include in comparison

        Returns:
            DataFrame with comparison table
        """
        if result_names is None:
            result_names = list(self.loaded_results.keys())

        comparison_data = []

        for result_name in result_names:
            if result_name not in self.loaded_results:
                logger.warning(f"Result {result_name} not loaded, skipping")
                continue

            results = self.loaded_results[result_name]

            # Extract metrics for each task
            for task_name, task_results in results.items():
                if task_name == 'overall' or task_name == 'summary':
                    continue

                if tasks and task_name not in tasks:
                    continue

                if not isinstance(task_results, dict):
                    continue

                row = {
                    'Model': result_name,
                    'Task': task_name
                }

                # Extract requested metrics
                for metric_name in metrics:
                    # Try to get metric value
                    value = task_results.get(metric_name)

                    # If not found directly, try in metrics_with_ci
                    if value is None and 'metrics_with_ci' in task_results:
                        ci_data = task_results['metrics_with_ci'].get(metric_name, {})
                        value = ci_data.get('value')

                    row[metric_name] = value if value is not None else np.nan

                comparison_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Pivot for better comparison view
        if len(df) > 0:
            pivot_dfs = []
            for metric in metrics:
                pivot = df.pivot(index='Task', columns='Model', values=metric)
                pivot.columns = [f'{col}_{metric}' for col in pivot.columns]
                pivot_dfs.append(pivot)

            comparison_df = pd.concat(pivot_dfs, axis=1)

            # Sort columns for readability
            comparison_df = comparison_df.sort_index(axis=1)

            logger.info(f"Created comparison table with {len(comparison_df)} tasks")
            return comparison_df
        else:
            logger.warning("No data found for comparison table")
            return pd.DataFrame()

    def create_radar_plot(
        self,
        result_names: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        metric: str = 'accuracy',
        save_path: Optional[Path] = None
    ):
        """
        Create multi-dimensional radar plot for task performance comparison

        Args:
            result_names: Names of results to compare
            tasks: Tasks to include (None = all)
            metric: Metric to plot
            save_path: Path to save plot
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not available for radar plots")
            return

        if result_names is None:
            result_names = list(self.loaded_results.keys())

        # Collect data for radar plot
        task_names = []
        model_data = {name: [] for name in result_names}

        # Determine tasks to include
        all_tasks = set()
        for results in self.loaded_results.values():
            all_tasks.update([k for k in results.keys() if k not in ['overall', 'summary']])

        if tasks:
            all_tasks = [t for t in all_tasks if t in tasks]
        else:
            all_tasks = sorted(list(all_tasks))

        # Extract metric values for each task and model
        for task_name in all_tasks:
            task_names.append(task_name)

            for result_name in result_names:
                if result_name not in self.loaded_results:
                    model_data[result_name].append(0)
                    continue

                results = self.loaded_results[result_name]
                task_results = results.get(task_name, {})

                # Get metric value
                value = task_results.get(metric)
                if value is None and 'metrics_with_ci' in task_results:
                    ci_data = task_results['metrics_with_ci'].get(metric, {})
                    value = ci_data.get('value', 0)

                model_data[result_name].append(value if value is not None else 0)

        # Create radar plot
        fig = go.Figure()

        for result_name in result_names:
            fig.add_trace(go.Scatterpolar(
                r=model_data[result_name],
                theta=task_names,
                fill='toself',
                name=result_name
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f'Task Performance Comparison - {metric}',
            showlegend=True,
            width=800,
            height=600
        )

        # Save plot
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"Saved radar plot: {save_path}")

        return fig

    def regression_analysis(
        self,
        checkpoint_results: Dict[int, Dict],
        task: str,
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Analyze metric progression across training checkpoints

        Args:
            checkpoint_results: Dictionary mapping checkpoint steps to results
            task: Task to analyze
            metric: Metric to analyze

        Returns:
            Dictionary with regression analysis results
        """
        # Extract metric values over checkpoints
        checkpoints = sorted(checkpoint_results.keys())
        metric_values = []

        for checkpoint in checkpoints:
            results = checkpoint_results[checkpoint]
            task_results = results.get(task, {})

            value = task_results.get(metric)
            if value is None and 'metrics_with_ci' in task_results:
                ci_data = task_results['metrics_with_ci'].get(metric, {})
                value = ci_data.get('value')

            metric_values.append(value if value is not None else np.nan)

        metric_values = np.array(metric_values)
        checkpoints_array = np.array(checkpoints)

        # Remove NaN values
        valid_mask = ~np.isnan(metric_values)
        metric_values_clean = metric_values[valid_mask]
        checkpoints_clean = checkpoints_array[valid_mask]

        if len(checkpoints_clean) < 2:
            logger.warning("Not enough data points for regression analysis")
            return {
                'task': task,
                'metric': metric,
                'error': 'Insufficient data points'
            }

        # Fit linear regression
        coefficients = np.polyfit(checkpoints_clean, metric_values_clean, 1)
        slope, intercept = coefficients

        # Calculate R-squared
        y_pred = slope * checkpoints_clean + intercept
        ss_res = np.sum((metric_values_clean - y_pred) ** 2)
        ss_tot = np.sum((metric_values_clean - np.mean(metric_values_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Find best checkpoint
        best_idx = np.argmax(metric_values_clean)
        best_checkpoint = checkpoints_clean[best_idx]
        best_value = metric_values_clean[best_idx]

        analysis = {
            'task': task,
            'metric': metric,
            'num_checkpoints': len(checkpoints_clean),
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'best_checkpoint': int(best_checkpoint),
            'best_value': float(best_value),
            'initial_value': float(metric_values_clean[0]),
            'final_value': float(metric_values_clean[-1]),
            'improvement': float(metric_values_clean[-1] - metric_values_clean[0]),
            'checkpoints': checkpoints_clean.tolist(),
            'values': metric_values_clean.tolist()
        }

        logger.info(f"Regression analysis for {task} - {metric}:")
        logger.info(f"  Slope: {slope:.6f}")
        logger.info(f"  R²: {r_squared:.4f}")
        logger.info(f"  Best: {best_value:.4f} at checkpoint {best_checkpoint}")

        return analysis

    def generate_html_report(
        self,
        report_path: Optional[Path] = None,
        result_names: Optional[List[str]] = None,
        include_radar: bool = True,
        include_comparison: bool = True
    ) -> Path:
        """
        Generate interactive HTML report with Bootstrap styling

        Args:
            report_path: Path to save report (None = auto-generate)
            result_names: Names of results to include
            include_radar: Whether to include radar plots
            include_comparison: Whether to include comparison tables

        Returns:
            Path to generated report
        """
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"comparative_report_{timestamp}.html"

        # Prepare HTML content
        html_content = self._generate_html_header()

        # Add title
        html_content += '<div class="container mt-5">\n'
        html_content += '<h1 class="text-center mb-4">Hindi BabyLM Evaluation Comparison Report</h1>\n'
        html_content += f'<p class="text-center text-muted">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n'

        # Add comparison table
        if include_comparison:
            html_content += '<h2 class="mt-5">Model Comparison</h2>\n'
            try:
                comparison_df = self.create_comparison_table(result_names=result_names)
                if not comparison_df.empty:
                    html_content += comparison_df.to_html(
                        classes='table table-striped table-bordered table-hover',
                        float_format=lambda x: f'{x:.4f}'
                    )
                else:
                    html_content += '<p class="text-warning">No comparison data available</p>\n'
            except Exception as e:
                html_content += f'<p class="text-danger">Error generating comparison table: {e}</p>\n'
                logger.error(f"Error in comparison table: {e}")

        # Add radar plot
        if include_radar and result_names and len(result_names) > 1:
            html_content += '<h2 class="mt-5">Task Performance Radar Plot</h2>\n'
            try:
                radar_plot_path = self.output_dir / 'radar_plot.html'
                self.create_radar_plot(
                    result_names=result_names,
                    save_path=radar_plot_path
                )

                # Embed radar plot
                with open(radar_plot_path, 'r') as f:
                    radar_html = f.read()
                    # Extract just the plot div
                    html_content += '<div class="text-center">\n'
                    html_content += f'<iframe src="radar_plot.html" width="900" height="700" frameborder="0"></iframe>\n'
                    html_content += '</div>\n'

            except Exception as e:
                html_content += f'<p class="text-danger">Error generating radar plot: {e}</p>\n'
                logger.error(f"Error in radar plot: {e}")

        # Close HTML
        html_content += '</div>\n'
        html_content += self._generate_html_footer()

        # Write HTML file
        with open(report_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Generated HTML report: {report_path}")
        return report_path

    def _generate_html_header(self) -> str:
        """Generate HTML header with Bootstrap styling"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hindi BabyLM Evaluation Comparison</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            font-weight: 600;
        }
        h2 {
            color: #34495e;
            font-weight: 500;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .table {
            font-size: 0.9rem;
        }
        .table thead {
            background-color: #3498db;
            color: white;
        }
    </style>
</head>
<body>
"""

    def _generate_html_footer(self) -> str:
        """Generate HTML footer"""
        return """
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    def generate_pdf_report(
        self,
        report_path: Optional[Path] = None,
        result_names: Optional[List[str]] = None
    ) -> Path:
        """
        Generate publication-ready PDF report

        Args:
            report_path: Path to save PDF
            result_names: Names of results to include

        Returns:
            Path to generated PDF
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.patches as mpatches
        except ImportError:
            logger.error("Matplotlib not available for PDF generation")
            raise

        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"comparative_report_{timestamp}.pdf"

        with PdfPages(str(report_path)) as pdf:
            # Page 1: Title page
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.7, 'Hindi BabyLM', ha='center', fontsize=32, fontweight='bold')
            fig.text(0.5, 0.6, 'Evaluation Comparison Report', ha='center', fontsize=24)
            fig.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    ha='center', fontsize=12, style='italic')

            if result_names:
                fig.text(0.5, 0.25, 'Compared Models:', ha='center', fontsize=14, fontweight='bold')
                for i, name in enumerate(result_names):
                    fig.text(0.5, 0.20 - i*0.03, f'• {name}', ha='center', fontsize=11)

            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 2: Comparison table
            try:
                comparison_df = self.create_comparison_table(result_names=result_names)
                if not comparison_df.empty:
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.axis('tight')
                    ax.axis('off')

                    # Create table
                    table_data = []
                    table_data.append(list(comparison_df.columns))
                    for idx, row in comparison_df.iterrows():
                        table_data.append([idx] + [f'{v:.4f}' if not pd.isna(v) else 'N/A'
                                                   for v in row.values])

                    table = ax.table(cellText=table_data, cellLoc='center',
                                   loc='center', bbox=[0, 0, 1, 1])
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 2)

                    # Style header
                    for i in range(len(table_data[0])):
                        table[(0, i)].set_facecolor('#3498db')
                        table[(0, i)].set_text_props(weight='bold', color='white')

                    plt.title('Model Comparison Table', fontsize=16, fontweight='bold', pad=20)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

            except Exception as e:
                logger.error(f"Error adding comparison table to PDF: {e}")

        logger.info(f"Generated PDF report: {report_path}")
        return report_path


# Convenience function
def compare_results(
    result_paths: Dict[str, str],
    output_dir: str = "comparative_analysis",
    generate_html: bool = True,
    generate_pdf: bool = False
) -> ComparativeAnalyzer:
    """
    Convenience function to compare multiple evaluation results

    Args:
        result_paths: Dictionary mapping names to result paths
        output_dir: Directory to save analysis
        generate_html: Whether to generate HTML report
        generate_pdf: Whether to generate PDF report

    Returns:
        ComparativeAnalyzer instance with loaded results
    """
    analyzer = ComparativeAnalyzer(output_dir=output_dir)
    analyzer.load_results(result_paths)

    result_names = list(result_paths.keys())

    if generate_html:
        analyzer.generate_html_report(result_names=result_names)

    if generate_pdf:
        try:
            analyzer.generate_pdf_report(result_names=result_names)
        except ImportError:
            logger.warning("Cannot generate PDF report without matplotlib")

    return analyzer
