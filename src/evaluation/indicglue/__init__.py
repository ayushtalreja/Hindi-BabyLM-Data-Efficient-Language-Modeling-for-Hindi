"""
IndicGLUE evaluation package (refactored)

Modular components for IndicGLUE benchmark evaluation.
"""

from .task_registry import TaskRegistry, TaskConfig
from .data_extractor import TaskDataExtractor
from .dataloader_factory import DataLoaderFactory
from .evaluation_strategies import (
    EvaluationStrategy,
    ClassificationStrategy,
    MultipleChoiceStrategy,
    PerplexityStrategy
)
from .fine_tuning_manager import FineTuningManager
from .result_visualizer import ResultVisualizer
from .memory_utils import cleanup_cuda_memory, move_model_to_cpu, cleanup_dataloader

__all__ = [
    'TaskRegistry', 'TaskConfig', 'TaskDataExtractor', 'DataLoaderFactory',
    'EvaluationStrategy', 'ClassificationStrategy', 'MultipleChoiceStrategy',
    'PerplexityStrategy',
    'FineTuningManager', 'ResultVisualizer',
    'cleanup_cuda_memory', 'move_model_to_cpu', 'cleanup_dataloader'
]
