"""
Data extraction utilities for IndicGLUE tasks.

Handles task-specific field extraction and label conversion.
Eliminates 30+ duplicated field-checking patterns from indicglue_evaluator.py.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from .task_registry import TaskRegistry

logger = logging.getLogger(__name__)


class TaskDataExtractor:
    """
    Extracts text and labels from task-specific dataset schemas.

    This class centralizes the field extraction logic that was previously
    duplicated across multiple dataloader collate functions in indicglue_evaluator.py.

    The extractor understands the schema of each IndicGLUE task and can:
    - Extract text fields (with task-specific concatenation rules)
    - Extract and convert labels (string → int mapping when needed)
    - Extract choices for multiple-choice tasks
    - Validate example schemas
    """

    def __init__(self, task_registry: TaskRegistry):
        """
        Initialize TaskDataExtractor

        Args:
            task_registry: TaskRegistry instance for task configuration
        """
        self.task_registry = task_registry

    def extract_text(self, example: Dict[str, Any], task_name: str) -> str:
        """
        Extract text from example based on task schema.

        This method implements task-specific text extraction rules:
        - WSTP: sectionText + all titles (for classification) or just sectionText (for MC wrapper)
        - CSQA: title + category + question + options (or just context for MC wrapper)
        - COPA: premise + question + choices (or just premise + question for MC wrapper)
        - BBC/Sentiment: text field
        - DiscourseMode: sentence field
        - NLI: premise [SEP] hypothesis

        Args:
            example: Dataset example
            task_name: Name of the task

        Returns:
            Extracted text string
        """
        task_config = self.task_registry.get_task_config(task_name)

        # COPA: Choice of Plausible Alternatives
        if 'premise' in example and 'choice1' in example and 'choice2' in example:
            premise = example['premise']
            question = example.get('question', '')

            if task_config.use_multiple_choice_wrapper:
                # For MC wrapper: return just premise + question (choices handled separately)
                return f"{premise} {question}".strip()
            else:
                # For old approach: concatenate everything
                choices = [example['choice1'], example['choice2']]
                return f"{premise} [SEP] {question} [SEP] " + " [SEP] ".join(choices)

        # CSQA: Cloze-style QA
        elif 'question' in example and 'options' in example:
            # Build context
            parts = []
            if 'title' in example and example['title']:
                parts.append(f"Title: {example['title']}")
            if 'category' in example and example['category']:
                parts.append(f"Category: {example['category']}")
            parts.append(example['question'])

            if task_config.use_multiple_choice_wrapper:
                # For MC wrapper: return just context (choices handled separately)
                return " ".join(parts)
            else:
                # For old approach: include options
                options = example['options']
                if isinstance(options, list):
                    parts.append(" [SEP] ".join(options))

                # Add out-of-context options if available
                out_of_context = example.get('out_of_context_options', [])
                if out_of_context and isinstance(out_of_context, list) and len(out_of_context) > 0:
                    parts.append(" [SEP] ".join(out_of_context))

                return " [SEP] ".join(parts)

        # WSTP: Wikipedia Section Title Prediction
        elif 'sectionText' in example and 'titleA' in example:
            section = example.get('sectionText', '')

            if task_config.use_multiple_choice_wrapper:
                # For MC wrapper: return just section (titles handled separately)
                return section
            else:
                # For old approach: concatenate section + all titles
                titles = [
                    example.get('titleA', ''),
                    example.get('titleB', ''),
                    example.get('titleC', ''),
                    example.get('titleD', '')
                ]
                titles = [t for t in titles if t]  # Filter empty
                return f"{section} [SEP] " + " [SEP] ".join(titles)

        # Standard text field (BBC, MovieReview, ProductReview)
        elif 'text' in example:
            return example['text']

        # DiscourseMode uses 'sentence'
        elif 'sentence' in example:
            return example['sentence']

        # NLI tasks (WinogradNLI)
        elif 'premise' in example and 'hypothesis' in example:
            return f"{example['premise']} [SEP] {example['hypothesis']}"

        # Fallback: first non-label string field
        else:
            text_fields = [k for k in example.keys()
                          if k != 'label' and isinstance(example.get(k), str)]
            if text_fields:
                logger.warning(
                    f"Using fallback text field '{text_fields[0]}' for {task_name}. "
                    f"Consider updating TaskDataExtractor.extract_text() for this task."
                )
                return example[text_fields[0]]
            return ""

    def extract_label(self, example: Dict[str, Any], task_name: str) -> int:
        """
        Extract and convert label to integer.

        This method handles task-specific label extraction and conversion:
        - WSTP: correctTitle → titleA/B/C/D → 0/1/2/3
        - CSQA: answer (find index in options)
        - BBCA: string label → int via BBCA_LABEL_MAP
        - DiscourseMode: string label → int via DISCOURSE_MODE_LABEL_MAP
        - Others: direct integer label

        Args:
            example: Dataset example
            task_name: Name of the task

        Returns:
            Integer label
        """
        task_config = self.task_registry.get_task_config(task_name)

        # WSTP: correctTitle field
        if 'correctTitle' in example:
            correct_title = example['correctTitle'].strip()

            # Normalize: "A" → "titleA", "B" → "titleB", etc.
            if correct_title and not correct_title.startswith('title'):
                correct_title = f"title{correct_title.upper()}"

            title_to_idx = {'titleA': 0, 'titleB': 1, 'titleC': 2, 'titleD': 3}

            if correct_title not in title_to_idx:
                logger.warning(
                    f"Unknown correctTitle '{example['correctTitle']}' for WSTP, defaulting to 0. "
                    f"Valid values: {list(title_to_idx.keys())}"
                )
                return 0

            return title_to_idx[correct_title]

        # CSQA: answer field (find index in options)
        elif 'answer' in example and 'options' in example:
            answer = example['answer']
            options = example['options']

            try:
                return options.index(answer)
            except (ValueError, AttributeError) as e:
                logger.warning(
                    f"Answer '{answer}' not found in options for CSQA, defaulting to 0. "
                    f"Options: {options}. Error: {e}"
                )
                return 0

        # Standard label field
        elif task_config.label_field in example:
            label = example[task_config.label_field]

            # Convert string labels if label_map exists
            if task_config.label_map and isinstance(label, str):
                converter = self.task_registry.get_label_converter(task_name)
                if converter:
                    return converter(label)

            # Already integer
            if isinstance(label, int):
                return label

            # Try to convert to int
            try:
                return int(label)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert label '{label}' to int for {task_name}, defaulting to 0"
                )
                return 0

        # Fallback
        else:
            logger.warning(
                f"No label field found for {task_name} (expected '{task_config.label_field}'), "
                f"defaulting to 0. Example keys: {list(example.keys())}"
            )
            return 0

    def extract_choices(self, example: Dict[str, Any], task_name: str) -> List[str]:
        """
        Extract choices for multiple-choice tasks.

        This method extracts the choice texts for MC tasks:
        - WSTP: [titleA, titleB, titleC, titleD]
        - CSQA: options list
        - COPA: [choice1, choice2]

        Args:
            example: Dataset example
            task_name: Name of the task

        Returns:
            List of choice strings

        Raises:
            ValueError: If task is not a multiple-choice task
        """
        task_config = self.task_registry.get_task_config(task_name)

        if not task_config.use_multiple_choice_wrapper:
            raise ValueError(
                f"{task_name} is not a multiple-choice task. "
                f"use_multiple_choice_wrapper={task_config.use_multiple_choice_wrapper}"
            )

        # WSTP
        if 'sectionText' in example and 'titleA' in example:
            return [
                example.get('titleA', ''),
                example.get('titleB', ''),
                example.get('titleC', ''),
                example.get('titleD', '')
            ]

        # CSQA
        elif 'question' in example and 'options' in example:
            choices = example['options']
            if not isinstance(choices, list):
                # Single option, wrap in list
                choices = [choices]
            return choices

        # COPA
        elif 'premise' in example and 'choice1' in example:
            return [
                example.get('choice1', ''),
                example.get('choice2', '')
            ]

        # Unknown MC task structure
        else:
            logger.warning(
                f"Cannot extract choices for {task_name}. "
                f"Example keys: {list(example.keys())}. "
                f"Returning empty choices list."
            )
            # Return empty strings matching expected num_choices
            return [''] * (task_config.num_choices or 2)

    def validate_example(self, example: Dict[str, Any], task_name: str) -> bool:
        """
        Validate that example has required fields for task.

        This method checks if an example has all the necessary fields
        for its task type. Useful for filtering corrupted data.

        Args:
            example: Dataset example
            task_name: Name of the task

        Returns:
            True if valid, False otherwise
        """
        task_config = self.task_registry.get_task_config(task_name)

        # Check label field exists
        if task_config.label_field not in example:
            logger.debug(
                f"Missing label field '{task_config.label_field}' in {task_name} example"
            )
            return False

        # Check text fields exist (at least one)
        # For complex tasks, check key identifying fields
        if 'sectionText' in task_config.text_fields:
            # WSTP
            if 'sectionText' not in example or 'titleA' not in example:
                return False
        elif 'question' in task_config.text_fields:
            # CSQA
            if 'question' not in example or 'options' not in example:
                return False
        elif 'choice1' in task_config.text_fields:
            # COPA
            if 'premise' not in example or 'choice1' not in example:
                return False
        elif 'text' in task_config.text_fields:
            # Standard classification
            if 'text' not in example:
                return False
        elif 'sentence' in task_config.text_fields:
            # DiscourseMode
            if 'sentence' not in example:
                return False
        elif 'premise' in task_config.text_fields and 'hypothesis' in task_config.text_fields:
            # NLI
            if 'premise' not in example or 'hypothesis' not in example:
                return False

        return True

    def get_text_field_name(self, task_name: str) -> Optional[str]:
        """
        Get the primary text field name for a task.

        Useful for debugging and logging.

        Args:
            task_name: Name of the task

        Returns:
            Primary text field name or None
        """
        task_config = self.task_registry.get_task_config(task_name)

        # Return first text field
        if task_config.text_fields:
            return task_config.text_fields[0]
        return None
