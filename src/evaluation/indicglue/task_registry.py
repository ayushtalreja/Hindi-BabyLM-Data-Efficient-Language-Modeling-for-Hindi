"""
Task configuration registry for IndicGLUE benchmark.

Centralizes all task metadata, label mappings, and split configurations.
Extracted from indicglue_evaluator.py as part of test-driven refactoring.
"""

from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a single IndicGLUE task"""
    name: str
    task_type: str  # 'classification', 'nli', 'multiple_choice'
    num_labels: Optional[int] = None
    num_choices: Optional[int] = None
    use_multiple_choice_wrapper: bool = False
    metric: str = 'accuracy'
    class_names: Optional[List[str]] = None
    hf_config: str = ''  # HuggingFace config name

    # Label conversion
    label_map: Optional[Dict] = None
    label_field: str = 'label'  # Dataset field name for labels

    # Text extraction
    text_fields: List[str] = field(default_factory=list)

    # Split remapping (for corrupted datasets)
    split_remapping: Optional[Dict[str, str]] = None


class TaskRegistry:
    """
    Centralized registry for IndicGLUE task configurations.

    Provides:
    - Task metadata lookup
    - Label conversion functions
    - Split remapping configuration

    This module centralizes task configuration that was previously scattered
    throughout indicglue_evaluator.py, making it easier to add new tasks
    and maintain consistency.
    """

    # Label mappings (moved from module-level constants)
    # BBCA (BBC Articles Classification) label mapping
    # The IndicGLUE BBCA dataset uses string labels, but models return integer predictions
    # This mapping converts string labels to integer indices for metric computation
    BBCA_LABEL_MAP = {
        'business': 0,
        'china': 1,
        'entertainment': 2,
        'india': 3,
        'institutional': 4,
        'international': 5,
        'learningenglish': 6,
        'multimedia': 7,
        'news': 8,
        'pakistan': 9,
        'science': 10,
        'social': 11,
        'southasia': 12,
        'sport': 13
    }

    # DiscourseMode label mapping
    # The IndicGLUE DiscourseMode dataset uses string labels, but models return integer predictions
    # This mapping converts string labels to integer indices for metric computation
    DISCOURSE_MODE_LABEL_MAP = {
        'Narrative': 0,
        'Descriptive': 1,
        'Dialogue': 2,
        'Informative': 3,
        'Argumentative': 4,
        'Other': 5
    }

    # Task-specific split remappings for corrupted datasets
    # Format: {task_name: {requested_split: actual_split_to_load}}
    SPLIT_REMAPPING = {
        'Choice of Plausible Alternatives': {
            'test': 'validation',      # Test corrupted → use validation (88 examples)
            'validation': 'test',       # Use corrupted test as validation (449 examples)
            'train': 'train'           # Train is fine, keep unchanged (362 examples)
        }
    }

    def __init__(self):
        """Initialize task registry with all IndicGLUE task configurations"""
        self._tasks: Dict[str, TaskConfig] = {}
        self._initialize_tasks()

    def _initialize_tasks(self):
        """Initialize all IndicGLUE task configurations"""

        # BBC Articles Classification
        self._tasks['BBCArticlesClassification'] = TaskConfig(
            name='BBCArticlesClassification',
            task_type='classification',
            num_labels=14,
            metric='accuracy',
            class_names=['business', 'china', 'entertainment', 'india', 'institutional',
                        'international', 'learningenglish', 'multimedia', 'news', 'pakistan',
                        'science', 'social', 'southasia', 'sport'],
            hf_config='bbca.hi',
            label_map=self.BBCA_LABEL_MAP,
            label_field='label',
            text_fields=['text']
        )

        # Wikipedia Section Title Prediction
        self._tasks['Wikipedia Section Title Prediction'] = TaskConfig(
            name='Wikipedia Section Title Prediction',
            task_type='multiple_choice',
            num_choices=4,
            use_multiple_choice_wrapper=True,
            metric='accuracy',
            hf_config='wstp.hi',
            label_field='correctTitle',
            text_fields=['sectionText', 'titleA', 'titleB', 'titleC', 'titleD']
        )

        # Cloze-style multiple-choice QA
        self._tasks['Cloze-style multiple-choice QA'] = TaskConfig(
            name='Cloze-style multiple-choice QA',
            task_type='multiple_choice',
            num_choices=4,
            use_multiple_choice_wrapper=True,
            metric='accuracy',
            hf_config='csqa.hi',
            label_field='answer',
            text_fields=['title', 'category', 'question', 'options']
        )

        # WinogradNLI
        self._tasks['WinogradNLI'] = TaskConfig(
            name='WinogradNLI',
            task_type='nli',
            num_labels=3,
            metric='accuracy',
            class_names=['Not Entailment', 'Entailment', 'None'],
            hf_config='wnli.hi',
            label_field='label',
            text_fields=['premise', 'hypothesis']
        )

        # Choice of Plausible Alternatives
        self._tasks['Choice of Plausible Alternatives'] = TaskConfig(
            name='Choice of Plausible Alternatives',
            task_type='multiple_choice',
            num_choices=2,
            use_multiple_choice_wrapper=True,
            metric='accuracy',
            hf_config='copa.hi',
            label_field='label',
            text_fields=['premise', 'question', 'choice1', 'choice2'],
            split_remapping=self.SPLIT_REMAPPING['Choice of Plausible Alternatives']
        )

        # MovieReviewSentiment
        self._tasks['MovieReviewSentiment'] = TaskConfig(
            name='MovieReviewSentiment',
            task_type='classification',
            num_labels=3,
            metric='accuracy',
            class_names=['Negative', 'Neutral', 'Positive'],
            hf_config='iitp-mr.hi',
            label_field='label',
            text_fields=['text']
        )

        # ProductReviewSentiment
        self._tasks['ProductReviewSentiment'] = TaskConfig(
            name='ProductReviewSentiment',
            task_type='classification',
            num_labels=3,
            metric='accuracy',
            class_names=['Negative', 'Neutral', 'Positive'],
            hf_config='iitp-pr.hi',
            label_field='label',
            text_fields=['text']
        )

        # DiscourseMode
        self._tasks['DiscourseMode'] = TaskConfig(
            name='DiscourseMode',
            task_type='classification',
            num_labels=6,
            metric='accuracy',
            class_names=['Narrative', 'Descriptive', 'Dialogue',
                        'Informative', 'Argumentative', 'Other'],
            hf_config='md.hi',
            label_map=self.DISCOURSE_MODE_LABEL_MAP,
            label_field='discourse_mode',
            text_fields=['sentence']
        )

    def get_task_config(self, task_name: str) -> TaskConfig:
        """
        Get configuration for a specific task

        Args:
            task_name: Name of the task

        Returns:
            TaskConfig object with task metadata

        Raises:
            ValueError: If task_name is not registered
        """
        if task_name not in self._tasks:
            available = ', '.join(self._tasks.keys())
            raise ValueError(
                f"Unknown task: '{task_name}'. "
                f"Available tasks: {available}"
            )
        return self._tasks[task_name]

    def get_all_task_names(self) -> List[str]:
        """
        Get all registered task names

        Returns:
            List of task names
        """
        return list(self._tasks.keys())

    def get_label_converter(self, task_name: str) -> Optional[Callable]:
        """
        Get label conversion function for task (if applicable)

        For tasks with string labels (BBCA, DiscourseMode), returns a function
        that converts string labels to integer indices. For tasks with integer
        labels, returns None.

        Args:
            task_name: Name of the task

        Returns:
            Label conversion function or None
        """
        task_config = self.get_task_config(task_name)

        if task_config.label_map is None:
            return None

        def convert_label(label):
            """Convert string label to integer"""
            if isinstance(label, int):
                return label
            if label not in task_config.label_map:
                logger.warning(
                    f"Unknown label '{label}' for {task_name}, defaulting to 0. "
                    f"Valid labels: {list(task_config.label_map.keys())}"
                )
                return 0
            return task_config.label_map[label]

        return convert_label

    def get_split_remapping(self, task_name: str) -> Optional[Dict[str, str]]:
        """
        Get split remapping configuration for task

        Some tasks (e.g., COPA) have corrupted splits in the dataset that need
        to be remapped. This method returns the remapping dictionary if one
        exists for the task.

        Args:
            task_name: Name of the task

        Returns:
            Split remapping dictionary or None

        Example:
            >>> registry.get_split_remapping('Choice of Plausible Alternatives')
            {'test': 'validation', 'validation': 'test', 'train': 'train'}
        """
        task_config = self.get_task_config(task_name)
        return task_config.split_remapping

    def has_label_map(self, task_name: str) -> bool:
        """
        Check if task has a label mapping

        Args:
            task_name: Name of the task

        Returns:
            True if task has label_map, False otherwise
        """
        task_config = self.get_task_config(task_name)
        return task_config.label_map is not None

    def uses_multiple_choice_wrapper(self, task_name: str) -> bool:
        """
        Check if task uses multiple-choice wrapper

        Args:
            task_name: Name of the task

        Returns:
            True if task uses MC wrapper, False otherwise
        """
        task_config = self.get_task_config(task_name)
        return task_config.use_multiple_choice_wrapper
