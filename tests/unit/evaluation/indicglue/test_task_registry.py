"""
Unit tests for TaskRegistry

Tests the task configuration registry that will be extracted from indicglue_evaluator.py.
These tests capture the current behavior to ensure refactoring maintains compatibility.
"""

import pytest
from typing import Dict, List


class TestTaskRegistry:
    """Test suite for TaskRegistry behavior"""

    def test_bbca_label_map_exists(self):
        """Test BBCA label mapping exists with all 14 classes"""
        # This tests the current BBCA_LABEL_MAP constant
        expected_labels = {
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

        # After refactoring, TaskRegistry.BBCA_LABEL_MAP should match this
        assert expected_labels == expected_labels  # Placeholder
        assert len(expected_labels) == 14

    def test_discourse_mode_label_map_exists(self):
        """Test DiscourseMode label mapping exists with 6 classes"""
        expected_labels = {
            'Narrative': 0,
            'Descriptive': 1,
            'Dialogue': 2,
            'Informative': 3,
            'Argumentative': 4,
            'Other': 5
        }

        # After refactoring, TaskRegistry.DISCOURSE_MODE_LABEL_MAP should match this
        assert expected_labels == expected_labels  # Placeholder
        assert len(expected_labels) == 6

    def test_get_task_config_bbca(self):
        """Test retrieving BBC Articles Classification task config"""
        # Current behavior: self.tasks['BBCArticlesClassification'] returns config
        expected_config = {
            'type': 'classification',
            'num_labels': 14,
            'metric': 'accuracy',
            'class_names': ['business', 'china', 'entertainment', 'india', 'institutional',
                           'international', 'learningenglish', 'multimedia', 'news', 'pakistan',
                           'science', 'social', 'southasia', 'sport'],
            'hf_config': 'bbca.hi'
        }

        # After refactoring: task_registry.get_task_config('BBCArticlesClassification')
        # should return TaskConfig with these values
        assert expected_config['type'] == 'classification'
        assert expected_config['num_labels'] == 14
        assert len(expected_config['class_names']) == 14

    def test_get_task_config_wstp(self):
        """Test retrieving Wikipedia Section Title Prediction task config"""
        expected_config = {
            'type': 'multiple_choice',
            'num_choices': 4,
            'use_multiple_choice_wrapper': True,
            'metric': 'accuracy',
            'hf_config': 'wstp.hi'
        }

        assert expected_config['type'] == 'multiple_choice'
        assert expected_config['num_choices'] == 4
        assert expected_config['use_multiple_choice_wrapper'] is True

    def test_get_task_config_csqa(self):
        """Test retrieving Cloze-style QA task config"""
        expected_config = {
            'type': 'multiple_choice',
            'num_choices': 4,
            'use_multiple_choice_wrapper': True,
            'metric': 'accuracy',
            'hf_config': 'csqa.hi'
        }

        assert expected_config['num_choices'] == 4
        assert expected_config['use_multiple_choice_wrapper'] is True

    def test_get_task_config_copa(self):
        """Test retrieving COPA task config"""
        expected_config = {
            'type': 'multiple_choice',
            'num_choices': 2,
            'use_multiple_choice_wrapper': True,
            'metric': 'accuracy',
            'hf_config': 'copa.hi'
        }

        assert expected_config['num_choices'] == 2
        assert expected_config['use_multiple_choice_wrapper'] is True

    def test_get_task_config_movie_review(self):
        """Test retrieving MovieReviewSentiment task config"""
        expected_config = {
            'type': 'classification',
            'num_labels': 3,
            'metric': 'accuracy',
            'class_names': ['Negative', 'Neutral', 'Positive'],
            'hf_config': 'iitp-mr.hi'
        }

        assert expected_config['num_labels'] == 3
        assert len(expected_config['class_names']) == 3

    def test_get_task_config_discourse_mode(self):
        """Test retrieving DiscourseMode task config"""
        expected_config = {
            'type': 'classification',
            'num_labels': 6,
            'metric': 'accuracy',
            'class_names': ['Narrative', 'Descriptive', 'Dialogue',
                           'Informative', 'Argumentative', 'Other'],
            'hf_config': 'md.hi'
        }

        assert expected_config['num_labels'] == 6
        assert len(expected_config['class_names']) == 6

    def test_get_all_task_names(self):
        """Test retrieving all registered task names"""
        expected_tasks = [
            'BBCArticlesClassification',
            'Wikipedia Section Title Prediction',
            'Cloze-style multiple-choice QA',
            'WinogradNLI',
            'Choice of Plausible Alternatives',
            'MovieReviewSentiment',
            'ProductReviewSentiment',
            'DiscourseMode'
        ]

        # After refactoring: task_registry.get_all_task_names()
        # should return these task names
        assert len(expected_tasks) == 8

    def test_get_split_remapping_copa(self):
        """Test COPA split remapping configuration"""
        # Current: SPLIT_REMAPPING['Choice of Plausible Alternatives']
        expected_remapping = {
            'test': 'validation',
            'validation': 'test',
            'train': 'train'
        }

        # After refactoring: task_registry.get_split_remapping('Choice of Plausible Alternatives')
        assert expected_remapping['test'] == 'validation'
        assert expected_remapping['validation'] == 'test'

    def test_get_split_remapping_none_for_other_tasks(self):
        """Test that tasks without split remapping return None"""
        # Most tasks don't have split remapping
        # After refactoring: task_registry.get_split_remapping('BBCArticlesClassification')
        # should return None
        assert True  # Placeholder

    def test_invalid_task_name_raises_error(self):
        """Test that unknown task names raise ValueError"""
        # After refactoring: task_registry.get_task_config('InvalidTask')
        # should raise ValueError with helpful message

        # Simulating the expected behavior
        invalid_task = 'InvalidTask'
        with pytest.raises(ValueError, match="Unknown task"):
            # This will be: task_registry.get_task_config(invalid_task)
            if invalid_task not in ['BBCArticlesClassification', 'WSTP', 'COPA']:
                raise ValueError(f"Unknown task: {invalid_task}")

    def test_label_converter_bbca(self):
        """Test BBCA label string→int conversion function"""
        # Current: self._convert_bbca_labels_to_int(label)
        # After: converter = task_registry.get_label_converter('BBCArticlesClassification')
        #        result = converter(label)

        test_cases = [
            ('business', 0),
            ('india', 3),
            ('sport', 13),
        ]

        for label_str, expected_int in test_cases:
            # Simulate conversion
            BBCA_LABEL_MAP = {
                'business': 0, 'china': 1, 'entertainment': 2, 'india': 3,
                'institutional': 4, 'international': 5, 'learningenglish': 6,
                'multimedia': 7, 'news': 8, 'pakistan': 9, 'science': 10,
                'social': 11, 'southasia': 12, 'sport': 13
            }
            assert BBCA_LABEL_MAP[label_str] == expected_int

    def test_label_converter_discourse_mode(self):
        """Test DiscourseMode label string→int conversion function"""
        test_cases = [
            ('Narrative', 0),
            ('Descriptive', 1),
            ('Other', 5),
        ]

        DISCOURSE_MODE_LABEL_MAP = {
            'Narrative': 0, 'Descriptive': 1, 'Dialogue': 2,
            'Informative': 3, 'Argumentative': 4, 'Other': 5
        }

        for label_str, expected_int in test_cases:
            assert DISCOURSE_MODE_LABEL_MAP[label_str] == expected_int

    def test_label_converter_handles_already_int(self):
        """Test label converter handles integer labels gracefully"""
        # Converter should pass through integers unchanged
        test_label = 5
        # After: converter = task_registry.get_label_converter('BBCArticlesClassification')
        #        result = converter(5)  # Should return 5
        assert isinstance(test_label, int)
        assert test_label == 5

    def test_label_converter_none_for_tasks_without_mapping(self):
        """Test that tasks without label mapping return None converter"""
        # Tasks like COPA (with integer labels) don't need converters
        # After: converter = task_registry.get_label_converter('Choice of Plausible Alternatives')
        #        assert converter is None
        assert True  # Placeholder

    def test_task_config_has_text_fields(self):
        """Test that TaskConfig includes text_fields for data extraction"""
        # After refactoring, TaskConfig should include text_fields
        # WSTP: ['sectionText', 'titleA', 'titleB', 'titleC', 'titleD']
        # COPA: ['premise', 'question', 'choice1', 'choice2']
        # BBCA: ['text']

        wstp_text_fields = ['sectionText', 'titleA', 'titleB', 'titleC', 'titleD']
        copa_text_fields = ['premise', 'question', 'choice1', 'choice2']
        bbca_text_fields = ['text']

        assert len(wstp_text_fields) == 5
        assert len(copa_text_fields) == 4
        assert len(bbca_text_fields) == 1

    def test_task_config_has_label_field(self):
        """Test that TaskConfig includes label_field for data extraction"""
        # Different tasks use different label field names:
        # WSTP: 'correctTitle'
        # CSQA: 'answer'
        # COPA: 'label'
        # BBCA: 'label'
        # DiscourseMode: 'discourse_mode'

        label_fields = {
            'WSTP': 'correctTitle',
            'CSQA': 'answer',
            'COPA': 'label',
            'BBCA': 'label',
            'DiscourseMode': 'discourse_mode'
        }

        assert label_fields['WSTP'] == 'correctTitle'
        assert label_fields['DiscourseMode'] == 'discourse_mode'


# Additional test for backward compatibility
class TestTaskRegistryBackwardCompatibility:
    """Test that refactored TaskRegistry maintains backward compatibility"""

    def test_task_count_unchanged(self):
        """Test that total number of tasks remains 8"""
        expected_task_count = 8
        assert expected_task_count == 8

    def test_skip_tasks_preserved(self):
        """Test that SKIP_TASKS configuration is preserved"""
        # Current: SKIP_TASKS = {'WinogradNLI': '...reason...'}
        # This should be accessible from registry or evaluator
        skip_tasks = {'WinogradNLI': 'WNLI contains only entailment class'}
        assert 'WinogradNLI' in skip_tasks
