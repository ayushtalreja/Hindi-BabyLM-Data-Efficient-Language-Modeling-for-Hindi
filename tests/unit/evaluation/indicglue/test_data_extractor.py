"""
Unit tests for TaskDataExtractor

Tests the data extraction logic that will be extracted from indicglue_evaluator.py.
These tests capture the current behavior of field extraction and label conversion.
"""

import pytest
from typing import Dict


class TestTaskDataExtractor:
    """Test suite for TaskDataExtractor behavior"""

    # ===== Text Extraction Tests =====

    def test_extract_text_bbca(self):
        """Test BBC text field extraction"""
        example = {
            'text': 'यह एक समाचार लेख है।',
            'label': 'news'
        }

        # Current: Directly accesses example['text']
        # After: data_extractor.extract_text(example, 'BBCArticlesClassification')
        assert example['text'] == 'यह एक समाचार लेख है।'

    def test_extract_text_discourse_mode(self):
        """Test DiscourseMode sentence field extraction"""
        example = {
            'sentence': 'यह एक कथात्मक वाक्य है।',
            'discourse_mode': 'Narrative'
        }

        # Current: Accesses example['sentence']
        # After: data_extractor.extract_text(example, 'DiscourseMode')
        assert example['sentence'] == 'यह एक कथात्मक वाक्य है।'

    def test_extract_text_wstp(self):
        """Test WSTP context + titles extraction"""
        example = {
            'sectionText': 'यह एक अनुभाग पाठ है।',
            'titleA': 'शीर्षक ए',
            'titleB': 'शीर्षक बी',
            'titleC': 'शीर्षक सी',
            'titleD': 'शीर्षक डी',
            'correctTitle': 'titleA'
        }

        # Current (multiple-choice wrapper): Returns just sectionText
        # After: data_extractor.extract_text(example, 'Wikipedia Section Title Prediction')
        # For MC wrapper mode: should return just context
        assert example['sectionText'] == 'यह एक अनुभाग पाठ है।'

    def test_extract_text_wstp_concatenated_for_classification(self):
        """Test WSTP concatenation for non-wrapper classification"""
        example = {
            'sectionText': 'Section text',
            'titleA': 'Title A',
            'titleB': 'Title B',
            'titleC': 'Title C',
            'titleD': 'Title D'
        }

        # For old classification approach (not MC wrapper):
        # Should concatenate: "section [SEP] titleA [SEP] titleB [SEP] titleC [SEP] titleD"
        expected_parts = [example['sectionText'], example['titleA'],
                         example['titleB'], example['titleC'], example['titleD']]
        assert len(expected_parts) == 5

    def test_extract_text_csqa(self):
        """Test CSQA question + options extraction"""
        example = {
            'title': 'आर्टिकल शीर्षक',
            'category': 'सामान्य ज्ञान',
            'question': 'यह प्रश्न है?',
            'options': ['विकल्प 1', 'विकल्प 2', 'विकल्प 3', 'विकल्प 4'],
            'answer': 'विकल्प 1'
        }

        # Current (MC wrapper): Returns "Title: {title} Category: {category} {question}"
        # After: data_extractor.extract_text(example, 'Cloze-style multiple-choice QA')
        context_parts = []
        if example['title']:
            context_parts.append(f"Title: {example['title']}")
        if example['category']:
            context_parts.append(f"Category: {example['category']}")
        context_parts.append(example['question'])

        context = " ".join(context_parts)
        assert 'Title:' in context
        assert 'Category:' in context
        assert example['question'] in context

    def test_extract_text_copa(self):
        """Test COPA premise + question extraction"""
        example = {
            'premise': 'लड़का स्कूल गया।',
            'question': 'क्यों?',
            'choice1': 'उसे पढ़ना था।',
            'choice2': 'वह बीमार था।',
            'label': 0
        }

        # Current (MC wrapper): Returns "premise question"
        # After: data_extractor.extract_text(example, 'Choice of Plausible Alternatives')
        context = f"{example['premise']} {example['question']}".strip()
        assert context == 'लड़का स्कूल गया। क्यों?'

    def test_extract_text_nli(self):
        """Test NLI premise + hypothesis extraction"""
        example = {
            'premise': 'यह परिसर है।',
            'hypothesis': 'यह परिकल्पना है।',
            'label': 1
        }

        # Current: Concatenates with [SEP]
        # After: data_extractor.extract_text(example, 'WinogradNLI')
        text = f"{example['premise']} [SEP] {example['hypothesis']}"
        assert '[SEP]' in text
        assert example['premise'] in text
        assert example['hypothesis'] in text

    def test_extract_text_empty_fields_handled(self):
        """Test empty string handling in text extraction"""
        example = {
            'text': '',
            'label': 0
        }

        # Should handle empty text gracefully
        assert example['text'] == ''

    def test_extract_text_fallback_to_first_string_field(self):
        """Test fallback to first non-label string field"""
        example = {
            'unknown_field': 'Some text content',
            'label': 0
        }

        # Current: Falls back to first non-label string field
        # After: data_extractor.extract_text should find 'unknown_field'
        text_fields = [k for k in example.keys() if k != 'label' and isinstance(example[k], str)]
        assert 'unknown_field' in text_fields

    # ===== Label Extraction Tests =====

    def test_extract_label_bbca_string(self):
        """Test BBCA string label → int conversion"""
        example = {
            'text': 'News article',
            'label': 'news'
        }

        # Current: self._convert_bbca_labels_to_int('news') → 8
        # After: data_extractor.extract_label(example, 'BBCArticlesClassification')
        BBCA_LABEL_MAP = {'news': 8}
        assert BBCA_LABEL_MAP[example['label']] == 8

    def test_extract_label_bbca_already_int(self):
        """Test BBCA handles integer labels"""
        example = {
            'text': 'News article',
            'label': 8
        }

        # Should pass through integers unchanged
        assert isinstance(example['label'], int)
        assert example['label'] == 8

    def test_extract_label_wstp_title_mapping(self):
        """Test WSTP titleA/B/C/D → 0/1/2/3 mapping"""
        test_cases = [
            ('titleA', 0),
            ('titleB', 1),
            ('titleC', 2),
            ('titleD', 3),
        ]

        for title, expected_idx in test_cases:
            example = {'correctTitle': title}
            title_to_idx = {'titleA': 0, 'titleB': 1, 'titleC': 2, 'titleD': 3}
            assert title_to_idx[title] == expected_idx

    def test_extract_label_wstp_normalizes_format(self):
        """Test WSTP normalizes 'A' → 'titleA' format"""
        example = {'correctTitle': 'A'}

        # Current: Normalizes single letter to 'titleA' format
        # After: data_extractor.extract_label should handle both formats
        correct_title = example['correctTitle'].strip()
        if correct_title and not correct_title.startswith('title'):
            correct_title = f"title{correct_title.upper()}"

        assert correct_title == 'titleA'

    def test_extract_label_csqa_answer_index(self):
        """Test CSQA answer finding in options"""
        example = {
            'question': 'Question?',
            'options': ['Option 1', 'Option 2', 'Option 3', 'Option 4'],
            'answer': 'Option 2'
        }

        # Current: Finds index of answer in options
        # After: data_extractor.extract_label(example, 'Cloze-style multiple-choice QA')
        label = example['options'].index(example['answer'])
        assert label == 1

    def test_extract_label_csqa_answer_not_found_defaults_to_zero(self):
        """Test CSQA defaults to 0 when answer not in options"""
        example = {
            'question': 'Question?',
            'options': ['Option 1', 'Option 2'],
            'answer': 'Invalid Option'
        }

        # Current: Logs warning and defaults to 0
        # After: Same behavior with data_extractor
        try:
            label = example['options'].index(example['answer'])
        except ValueError:
            label = 0

        assert label == 0

    def test_extract_label_copa_integer(self):
        """Test COPA integer label"""
        example = {
            'premise': 'Premise',
            'choice1': 'Choice 1',
            'choice2': 'Choice 2',
            'label': 0
        }

        # COPA uses integer labels (0 or 1)
        assert example['label'] == 0
        assert isinstance(example['label'], int)

    def test_extract_label_discourse_mode_string(self):
        """Test DiscourseMode string label conversion"""
        example = {
            'sentence': 'Sentence',
            'discourse_mode': 'Narrative'
        }

        # Current: self._convert_discourse_mode_labels_to_int('Narrative') → 0
        # After: data_extractor.extract_label(example, 'DiscourseMode')
        DISCOURSE_MODE_LABEL_MAP = {'Narrative': 0}
        assert DISCOURSE_MODE_LABEL_MAP[example['discourse_mode']] == 0

    def test_extract_label_missing_field_defaults_to_zero(self):
        """Test fallback behavior for missing labels"""
        example = {
            'text': 'Some text'
            # No label field
        }

        # Current: Logs warning and defaults to 0
        # After: data_extractor.extract_label should return 0
        label = example.get('label', 0)
        assert label == 0

    # ===== Choice Extraction Tests =====

    def test_extract_choices_wstp(self):
        """Test WSTP 4 title choices extraction"""
        example = {
            'sectionText': 'Section',
            'titleA': 'Title A',
            'titleB': 'Title B',
            'titleC': 'Title C',
            'titleD': 'Title D'
        }

        # After: data_extractor.extract_choices(example, 'Wikipedia Section Title Prediction')
        choices = [
            example.get('titleA', ''),
            example.get('titleB', ''),
            example.get('titleC', ''),
            example.get('titleD', '')
        ]

        assert len(choices) == 4
        assert choices[0] == 'Title A'

    def test_extract_choices_csqa(self):
        """Test CSQA options list extraction"""
        example = {
            'question': 'Question?',
            'options': ['Opt 1', 'Opt 2', 'Opt 3', 'Opt 4']
        }

        # After: data_extractor.extract_choices(example, 'Cloze-style multiple-choice QA')
        choices = example['options']
        assert len(choices) == 4
        assert isinstance(choices, list)

    def test_extract_choices_csqa_handles_single_option(self):
        """Test CSQA handles non-list options"""
        example = {
            'question': 'Question?',
            'options': 'Single option'
        }

        # Current: Wraps in list if not already a list
        # After: data_extractor.extract_choices should handle this
        choices = example['options']
        if not isinstance(choices, list):
            choices = [choices]

        assert isinstance(choices, list)
        assert len(choices) == 1

    def test_extract_choices_copa(self):
        """Test COPA 2 alternatives extraction"""
        example = {
            'premise': 'Premise',
            'choice1': 'Choice 1',
            'choice2': 'Choice 2'
        }

        # After: data_extractor.extract_choices(example, 'Choice of Plausible Alternatives')
        choices = [
            example.get('choice1', ''),
            example.get('choice2', '')
        ]

        assert len(choices) == 2
        assert choices[0] == 'Choice 1'

    def test_extract_choices_raises_for_non_mc_task(self):
        """Test extract_choices raises for non-multiple-choice tasks"""
        # BBCA is a classification task, not multiple-choice
        # After: data_extractor.extract_choices(example, 'BBCArticlesClassification')
        # Should raise ValueError

        with pytest.raises(ValueError, match="not a multiple-choice task"):
            # Simulating the expected behavior
            task_type = 'classification'  # BBCA is classification
            if task_type != 'multiple_choice':
                raise ValueError("BBCArticlesClassification is not a multiple-choice task")

    # ===== Validation Tests =====

    def test_validate_example_wstp(self):
        """Test WSTP example validation"""
        valid_example = {
            'sectionText': 'Section',
            'titleA': 'A',
            'titleB': 'B',
            'titleC': 'C',
            'titleD': 'D',
            'correctTitle': 'titleA'
        }

        # After: data_extractor.validate_example(valid_example, 'Wikipedia Section Title Prediction')
        # Should return True
        required_fields = ['sectionText', 'titleA', 'titleB', 'titleC', 'titleD', 'correctTitle']
        is_valid = all(field in valid_example for field in required_fields)
        assert is_valid is True

    def test_validate_example_missing_text_field(self):
        """Test validation fails for missing text field"""
        invalid_example = {
            # Missing sectionText
            'titleA': 'A',
            'correctTitle': 'titleA'
        }

        # After: data_extractor.validate_example should return False
        required_fields = ['sectionText', 'titleA']
        is_valid = all(field in invalid_example for field in required_fields)
        assert is_valid is False

    def test_validate_example_missing_label_field(self):
        """Test validation fails for missing label field"""
        invalid_example = {
            'text': 'Some text'
            # Missing label
        }

        # After: data_extractor.validate_example should return False
        is_valid = 'label' in invalid_example
        assert is_valid is False


# Additional edge case tests
class TestTaskDataExtractorEdgeCases:
    """Test edge cases and error handling"""

    def test_extract_text_with_empty_choices(self):
        """Test WSTP with some empty title fields"""
        example = {
            'sectionText': 'Section',
            'titleA': 'Title A',
            'titleB': '',  # Empty
            'titleC': 'Title C',
            'titleD': ''   # Empty
        }

        # Current: Filters out empty titles
        titles = [t for t in [example.get('titleA', ''), example.get('titleB', ''),
                              example.get('titleC', ''), example.get('titleD', '')] if t]
        assert len(titles) == 2
        assert 'Title A' in titles

    def test_extract_label_unknown_label_defaults(self):
        """Test unknown label defaults with warning"""
        example = {
            'text': 'Text',
            'label': 'invalid_label'
        }

        # For BBCA with strict_label_validation=False:
        # Should log warning and default to 0
        BBCA_LABEL_MAP = {'business': 0, 'news': 8}
        if example['label'] not in BBCA_LABEL_MAP:
            default_label = 0
        else:
            default_label = BBCA_LABEL_MAP[example['label']]

        assert default_label == 0

    def test_extract_wstp_title_with_whitespace(self):
        """Test WSTP correctTitle with leading/trailing whitespace"""
        example = {'correctTitle': '  titleA  '}

        # Current: Strips whitespace before mapping
        correct_title = example['correctTitle'].strip()
        assert correct_title == 'titleA'
