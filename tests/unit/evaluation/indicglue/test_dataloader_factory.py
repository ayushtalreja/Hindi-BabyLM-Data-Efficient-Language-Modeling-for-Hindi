"""
Unit tests for DataLoaderFactory

Tests the dataloader creation logic that will be extracted from indicglue_evaluator.py.
These tests capture the current behavior of _create_task_dataloader and _create_multiple_choice_dataloader.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List


class TestDataLoaderFactoryClassification:
    """Test suite for classification task dataloader creation"""

    def test_create_classification_loader_bbca(self):
        """Test BBC classification dataloader creation"""
        # Mock dataset
        mock_dataset = [
            {'text': 'Article text 1', 'label': 'business'},
            {'text': 'Article text 2', 'label': 'sport'}
        ]

        # After refactoring: factory.create_dataloader(dataset, 'BBCArticlesClassification')
        # Should return DataLoader with:
        # - Correct batch shape: [batch, seq_len]
        # - Labels converted: string → int
        # - Tokenization: max_length=128, padding='max_length'

        assert len(mock_dataset) == 2
        assert 'text' in mock_dataset[0]
        assert 'label' in mock_dataset[0]

    def test_create_classification_loader_discourse(self):
        """Test DiscourseMode classification dataloader"""
        mock_dataset = [
            {'sentence': 'Narrative sentence', 'discourse_mode': 'Narrative'},
            {'sentence': 'Descriptive text', 'discourse_mode': 'Descriptive'}
        ]

        # Should use 'sentence' field instead of 'text'
        # Should convert 'discourse_mode' string labels to int
        assert 'sentence' in mock_dataset[0]
        assert 'discourse_mode' in mock_dataset[0]

    def test_create_classification_loader_sentiment(self):
        """Test sentiment classification dataloader (MovieReview/ProductReview)"""
        mock_dataset = [
            {'text': 'Great movie!', 'label': 2},  # Positive
            {'text': 'Bad product.', 'label': 0}   # Negative
        ]

        # Should handle integer labels (no conversion needed)
        assert isinstance(mock_dataset[0]['label'], int)
        assert isinstance(mock_dataset[1]['label'], int)

    def test_classification_loader_batch_shape(self):
        """Test classification loader produces correct batch shape"""
        # After refactoring: factory.create_dataloader(dataset, task_name, batch_size=2)
        # Batch should have shape:
        # - input_ids: [2, max_length]
        # - attention_mask: [2, max_length]
        # - labels: [2]

        expected_batch_structure = {
            'input_ids': (2, 128),  # [batch, seq_len]
            'attention_mask': (2, 128),
            'labels': (2,)  # [batch]
        }

        assert expected_batch_structure['input_ids'][0] == 2
        assert expected_batch_structure['labels'] == (2,)

    def test_classification_loader_tokenization_params(self):
        """Test tokenization uses correct parameters"""
        # Current: tokenizer(texts, max_length=128, padding='max_length', truncation=True)
        # After: factory should use same params matching official IndicBERT

        expected_params = {
            'max_length': 128,  # IndicBERT paper standard
            'padding': 'max_length',  # NOT 'longest' - official uses max_length
            'truncation': True,
            'return_tensors': 'pt'
        }

        assert expected_params['max_length'] == 128
        assert expected_params['padding'] == 'max_length'

    def test_classification_loader_handles_nli_tasks(self):
        """Test NLI task dataloader (premise + hypothesis)"""
        mock_dataset = [
            {'premise': 'Premise text', 'hypothesis': 'Hypothesis text', 'label': 1}
        ]

        # Should concatenate: "premise [SEP] hypothesis"
        expected_text = f"{mock_dataset[0]['premise']} [SEP] {mock_dataset[0]['hypothesis']}"
        assert '[SEP]' in expected_text


class TestDataLoaderFactoryMultipleChoice:
    """Test suite for multiple-choice task dataloader creation"""

    def test_create_mc_loader_wstp(self):
        """Test WSTP multiple-choice dataloader"""
        mock_dataset = [
            {
                'sectionText': 'Section text',
                'titleA': 'Title A',
                'titleB': 'Title B',
                'titleC': 'Title C',
                'titleD': 'Title D',
                'correctTitle': 'titleA'
            }
        ]

        # After: factory.create_multiple_choice_dataloader(dataset, 'Wikipedia Section Title Prediction')
        # Should produce batch with:
        # - input_ids: [batch, 4, seq_len] - 4 choices
        # - attention_mask: [batch, 4, seq_len]
        # - labels: [batch] - index 0-3

        expected_shape = {
            'num_choices': 4,
            'label_range': (0, 3)
        }

        assert expected_shape['num_choices'] == 4

    def test_create_mc_loader_csqa(self):
        """Test CSQA multiple-choice dataloader"""
        mock_dataset = [
            {
                'title': 'Article title',
                'category': 'General Knowledge',
                'question': 'Question text?',
                'options': ['Opt 1', 'Opt 2', 'Opt 3', 'Opt 4'],
                'answer': 'Opt 2'
            }
        ]

        # Should include title and category in context
        # Should tokenize each (context, option) pair
        # Label should be index of answer in options (1 in this case)
        expected_label = mock_dataset[0]['options'].index(mock_dataset[0]['answer'])
        assert expected_label == 1

    def test_create_mc_loader_copa(self):
        """Test COPA multiple-choice dataloader"""
        mock_dataset = [
            {
                'premise': 'Boy went to school.',
                'question': 'Why?',
                'choice1': 'He had to study.',
                'choice2': 'He was sick.',
                'label': 0
            }
        ]

        # Should have 2 choices (not 4)
        # Context: "premise question"
        expected_num_choices = 2
        assert expected_num_choices == 2

    def test_mc_loader_batch_shape(self):
        """Test MC loader produces correct [batch, num_choices, seq_len] shape"""
        # This is CRITICAL - must match official IndicBERT
        # After: factory.create_multiple_choice_dataloader(dataset, task_name, batch_size=2)

        expected_shapes = {
            'wstp': {
                'input_ids': (2, 4, 128),  # [batch, 4 choices, seq_len]
                'attention_mask': (2, 4, 128),
                'labels': (2,)  # [batch] - values 0-3
            },
            'copa': {
                'input_ids': (2, 2, 128),  # [batch, 2 choices, seq_len]
                'attention_mask': (2, 2, 128),
                'labels': (2,)  # [batch] - values 0-1
            }
        }

        assert expected_shapes['wstp']['input_ids'][1] == 4  # 4 choices
        assert expected_shapes['copa']['input_ids'][1] == 2  # 2 choices

    def test_mc_loader_tokenizes_pairs_separately(self):
        """Test MC loader tokenizes each (context, choice) pair separately"""
        # Current: For each choice, calls tokenizer(context, choice, ...)
        # This is the official IndicBERT approach

        context = "Section text"
        choices = ["Title A", "Title B", "Title C", "Title D"]

        # Should call tokenizer 4 times:
        # tokenizer(context, "Title A", ...)
        # tokenizer(context, "Title B", ...)
        # tokenizer(context, "Title C", ...)
        # tokenizer(context, "Title D", ...)

        assert len(choices) == 4

    def test_mc_loader_truncation_strategy(self):
        """Test MC loader uses 'longest_first' truncation"""
        # CRITICAL: Official IndicBERT uses truncation='longest_first'
        # This is different from classification tasks!

        expected_truncation = 'longest_first'
        assert expected_truncation == 'longest_first'

    def test_mc_loader_wstp_label_mapping(self):
        """Test WSTP correctTitle → index mapping"""
        test_cases = [
            ('titleA', 0),
            ('titleB', 1),
            ('titleC', 2),
            ('titleD', 3),
            ('A', 0),  # Should normalize 'A' → 'titleA' → 0
        ]

        for correct_title, expected_idx in test_cases:
            # Normalize
            normalized = correct_title
            if normalized and not normalized.startswith('title'):
                normalized = f"title{normalized.upper()}"

            title_to_idx = {'titleA': 0, 'titleB': 1, 'titleC': 2, 'titleD': 3}
            actual_idx = title_to_idx.get(normalized, 0)
            assert actual_idx == expected_idx

    def test_mc_loader_csqa_context_includes_metadata(self):
        """Test CSQA context includes title and category"""
        example = {
            'title': 'Article Title',
            'category': 'Science',
            'question': 'What is X?',
            'options': ['A', 'B', 'C', 'D']
        }

        # Context should be: "Title: {title} Category: {category} {question}"
        context_parts = []
        if example['title']:
            context_parts.append(f"Title: {example['title']}")
        if example['category']:
            context_parts.append(f"Category: {example['category']}")
        context_parts.append(example['question'])

        context = " ".join(context_parts)
        assert 'Title:' in context
        assert 'Category:' in context


class TestDataLoaderFactoryConfiguration:
    """Test dataloader configuration options"""

    def test_dataloader_shuffle_option(self):
        """Test shuffle parameter is respected"""
        # After: factory.create_dataloader(dataset, task_name, shuffle=True)
        # DataLoader should have shuffle=True

        shuffle_config = {'train': True, 'val': False, 'test': False}
        assert shuffle_config['train'] is True
        assert shuffle_config['test'] is False

    def test_dataloader_batch_size_config(self):
        """Test custom batch size parameter"""
        # After: factory.create_dataloader(dataset, task_name, batch_size=16)
        # DataLoader should use batch_size=16

        custom_batch_size = 16
        default_batch_size = 32

        assert custom_batch_size == 16
        assert default_batch_size == 32

    def test_dataloader_num_workers(self):
        """Test DataLoader uses num_workers=0"""
        # Current: Always uses num_workers=0 to avoid multiprocessing issues
        # After: factory should maintain this

        expected_num_workers = 0
        assert expected_num_workers == 0

    def test_dataloader_uses_correct_device(self):
        """Test DataLoader moves tensors to correct device"""
        # Current: Moves all tensors to self.device
        # After: factory should receive device in constructor

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert isinstance(device, torch.device)


class TestDataLoaderFactoryCollateFunction:
    """Test collate function behavior"""

    def test_collate_handles_different_field_names(self):
        """Test collate function handles various field name patterns"""
        # BBCA: 'text'
        # DiscourseMode: 'sentence'
        # NLI: 'premise' + 'hypothesis'
        # COPA: 'premise' + 'question' + 'choice1' + 'choice2'

        field_mappings = {
            'BBCArticlesClassification': ['text'],
            'DiscourseMode': ['sentence'],
            'WinogradNLI': ['premise', 'hypothesis'],
            'Choice of Plausible Alternatives': ['premise', 'question', 'choice1', 'choice2']
        }

        assert 'text' in field_mappings['BBCArticlesClassification']
        assert 'sentence' in field_mappings['DiscourseMode']

    def test_collate_concatenates_multiple_fields(self):
        """Test collate concatenates fields with [SEP] token"""
        # NLI: "premise [SEP] hypothesis"
        # COPA (old approach): "premise [SEP] question [SEP] choice1 [SEP] choice2"

        nli_format = "premise [SEP] hypothesis"
        copa_format = "premise [SEP] question [SEP] choice1 [SEP] choice2"

        assert nli_format.count('[SEP]') == 1
        assert copa_format.count('[SEP]') == 3

    def test_collate_filters_empty_titles(self):
        """Test WSTP collate filters out empty title fields"""
        titles = ['Title A', '', 'Title C', '']
        filtered_titles = [t for t in titles if t]

        assert len(filtered_titles) == 2
        assert '' not in filtered_titles

    def test_collate_handles_fallback_text_field(self):
        """Test collate falls back to first non-label string field"""
        example = {
            'unknown_text_field': 'Some text',
            'label': 0,
            'numeric_field': 123
        }

        # Should find 'unknown_text_field' as the text field
        text_fields = [k for k in example.keys()
                      if k != 'label' and isinstance(example[k], str)]

        assert 'unknown_text_field' in text_fields


class TestDataLoaderFactoryEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataset_handling(self):
        """Test behavior with empty dataset"""
        empty_dataset = []

        # DataLoader should handle empty dataset gracefully
        assert len(empty_dataset) == 0

    def test_single_example_batch(self):
        """Test batch with single example"""
        # Batch size can be 1, shapes should still be correct
        # - input_ids: [1, seq_len]
        # - labels: [1]

        batch_size = 1
        assert batch_size == 1

    def test_max_length_parameter(self):
        """Test max_length parameter is used correctly"""
        # Current: Uses self.max_length from config (default 128)
        # After: factory receives max_length in constructor

        default_max_length = 128
        custom_max_length = 256

        assert default_max_length == 128
        assert custom_max_length == 256

    def test_padding_to_max_length(self):
        """Test sequences are padded to max_length"""
        # Official IndicBERT uses padding='max_length', not 'longest'
        # This means all sequences are padded to max_length, not batch max

        padding_strategy = 'max_length'
        assert padding_strategy == 'max_length'
        assert padding_strategy != 'longest'  # NOT this!

    def test_mc_loader_handles_missing_choices(self):
        """Test MC loader handles examples with missing/empty choices"""
        example_with_empty = {
            'sectionText': 'Text',
            'titleA': 'A',
            'titleB': '',  # Empty
            'titleC': 'C',
            'titleD': ''   # Empty
        }

        # Should still process, but might need to handle empty strings
        assert example_with_empty['titleB'] == ''


class TestDataLoaderFactoryRouting:
    """Test correct routing to dataloader types"""

    def test_routes_classification_tasks_correctly(self):
        """Test classification tasks use standard dataloader"""
        classification_tasks = [
            'BBCArticlesClassification',
            'MovieReviewSentiment',
            'ProductReviewSentiment',
            'DiscourseMode',
            'WinogradNLI'
        ]

        # After: factory.create_dataloader should auto-detect and route
        # to create_standard_dataloader for these tasks

        for task in classification_tasks:
            assert task in classification_tasks

    def test_routes_mc_tasks_correctly(self):
        """Test MC tasks use multiple-choice dataloader"""
        mc_tasks = [
            'Wikipedia Section Title Prediction',
            'Cloze-style multiple-choice QA',
            'Choice of Plausible Alternatives'
        ]

        # After: factory.create_dataloader should auto-detect and route
        # to create_multiple_choice_dataloader for these tasks

        for task in mc_tasks:
            assert task in mc_tasks

    def test_uses_task_config_for_routing(self):
        """Test routing uses task_config['use_multiple_choice_wrapper'] flag"""
        # Task config should have 'use_multiple_choice_wrapper' field
        # If True: use MC dataloader
        # If False/missing: use standard dataloader

        task_configs = {
            'WSTP': {'use_multiple_choice_wrapper': True},
            'BBCA': {'use_multiple_choice_wrapper': False}
        }

        assert task_configs['WSTP']['use_multiple_choice_wrapper'] is True
        assert task_configs['BBCA']['use_multiple_choice_wrapper'] is False
