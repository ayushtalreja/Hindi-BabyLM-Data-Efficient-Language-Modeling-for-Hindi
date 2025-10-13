"""
Pytest configuration and shared fixtures for Hindi BabyLM tests
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List

# Sample Hindi texts for testing
SAMPLE_HINDI_TEXTS = [
    "यह एक परीक्षण वाक्य है।",
    "हिंदी भाषा बहुत सुंदर है।",
    "मैं स्कूल जाता हूं।",
    "आज मौसम बहुत अच्छा है।",
    "किताब पढ़ना मुझे पसंद है।"
]


@pytest.fixture(scope="session")
def sample_hindi_texts():
    """Sample Hindi texts for testing"""
    return SAMPLE_HINDI_TEXTS


@pytest.fixture(scope="session")
def test_config():
    """Basic test configuration"""
    return {
        'seed': 42,
        'deterministic': True,
        'device': 'cpu',  # Use CPU for tests
        'batch_size': 4,
        'vocab_size': 1000,
        'hidden_size': 64,
        'num_layers': 2,
        'num_heads': 4,
        'max_length': 128,
        'tokenizer_type': 'sentencepiece',
        'model_type': 'gpt',
        'num_epochs': 1,
        'learning_rate': 1e-4,
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_corpus_file(temp_dir, sample_hindi_texts):
    """Create a sample corpus file for testing"""
    corpus_file = temp_dir / "sample_corpus.txt"

    with open(corpus_file, 'w', encoding='utf-8') as f:
        for text in sample_hindi_texts:
            f.write(text + '\n')

    return corpus_file


@pytest.fixture(scope="session")
def device():
    """Get appropriate device for testing"""
    return torch.device('cpu')  # Always use CPU for tests


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset PyTorch seed before each test"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_config_dict():
    """Mock configuration dictionary"""
    return {
        'project': {
            'name': 'test-hindi-babylm',
            'version': '1.0.0'
        },
        'data': {
            'max_tokens': 10000,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
        },
        'tokenization': {
            'type': 'sentencepiece',
            'vocab_size': 1000
        },
        'model': {
            'type': 'gpt',
            'architecture': {
                'hidden_size': 64,
                'num_hidden_layers': 2,
                'num_attention_heads': 4,
                'max_position_embeddings': 128
            }
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 1,
            'optimizer': {
                'type': 'adamw',
                'learning_rate': 1e-4
            }
        }
    }


def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
