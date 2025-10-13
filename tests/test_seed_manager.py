"""
Tests for seed management and reproducibility
"""

import pytest
import random
import numpy as np
import torch
from src.utils.seed_manager import SeedManager, set_global_seed


@pytest.mark.unit
class TestSeedManager:
    """Test SeedManager class"""

    def test_seed_manager_initialization(self):
        """Test SeedManager can be initialized"""
        seed_manager = SeedManager(seed=42, deterministic=True)
        assert seed_manager.seed == 42
        assert seed_manager.deterministic is True

    def test_set_all_seeds(self):
        """Test that set_all_seeds sets seeds for all RNGs"""
        seed_manager = SeedManager(seed=123)
        seed_manager.set_all_seeds()

        # Verify Python random
        r1 = random.random()
        random.seed(123)
        r2 = random.random()
        assert r1 == r2

    def test_reproducibility_validation(self):
        """Test reproducibility validation"""
        seed_manager = SeedManager(seed=42, deterministic=True)
        seed_manager.set_all_seeds()

        # Should return True for reproducible setup
        is_reproducible = seed_manager.validate_reproducibility()
        assert is_reproducible is True

    def test_worker_init_fn(self):
        """Test worker initialization function"""
        seed_manager = SeedManager(seed=42)

        # Should not raise error
        seed_manager.worker_init_fn(0)
        seed_manager.worker_init_fn(1)

    def test_get_generator(self):
        """Test getting a seeded generator"""
        seed_manager = SeedManager(seed=42)
        generator = seed_manager.get_generator()

        assert isinstance(generator, torch.Generator)

        # Generate some numbers
        nums = torch.randint(0, 100, (5,), generator=generator)
        assert len(nums) == 5

    def test_context_manager(self):
        """Test SeedManager as context manager"""
        with SeedManager(seed=42) as sm:
            assert sm.seed == 42
            val = random.random()
            assert val is not None

    def test_get_context_info(self):
        """Test getting context information"""
        seed_manager = SeedManager(seed=42)
        info = seed_manager.get_context_info()

        assert 'seed' in info
        assert 'deterministic' in info
        assert 'torch_version' in info
        assert info['seed'] == 42


@pytest.mark.unit
def test_set_global_seed():
    """Test set_global_seed convenience function"""
    seed_manager = set_global_seed(seed=999, deterministic=False)

    assert isinstance(seed_manager, SeedManager)
    assert seed_manager.seed == 999


@pytest.mark.unit
def test_reproducibility_across_runs():
    """Test that same seed produces same results"""
    # First run
    seed_manager1 = SeedManager(seed=42)
    seed_manager1.set_all_seeds()
    tensor1 = torch.rand(10)

    # Second run
    seed_manager2 = SeedManager(seed=42)
    seed_manager2.set_all_seeds()
    tensor2 = torch.rand(10)

    # Should be identical
    assert torch.allclose(tensor1, tensor2)


@pytest.mark.unit
def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results"""
    # Seed 1
    seed_manager1 = SeedManager(seed=42)
    seed_manager1.set_all_seeds()
    tensor1 = torch.rand(10)

    # Seed 2
    seed_manager2 = SeedManager(seed=123)
    seed_manager2.set_all_seeds()
    tensor2 = torch.rand(10)

    # Should be different
    assert not torch.allclose(tensor1, tensor2)
