"""
Seed Management Utility for Reproducible Experiments

This module provides comprehensive seed management for ensuring reproducibility
across all random number generators used in the Hindi BabyLM project.
"""

import os
import random
import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SeedManager:
    """
    Centralized seed management for reproducible experiments

    Manages seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CuDNN

    Example:
        >>> seed_manager = SeedManager(seed=42, deterministic=True)
        >>> seed_manager.set_all_seeds()
    """

    def __init__(self, seed: int = 42, deterministic: bool = True):
        """
        Initialize seed manager

        Args:
            seed: Random seed value
            deterministic: If True, sets deterministic algorithms for PyTorch
        """
        self.seed = seed
        self.deterministic = deterministic
        self._original_env = {}

    def set_all_seeds(self) -> None:
        """Set seeds for all random number generators"""
        logger.info(f"Setting all seeds to {self.seed}")

        # Python random
        random.seed(self.seed)
        logger.debug("✓ Python random seed set")

        # NumPy
        np.random.seed(self.seed)
        logger.debug("✓ NumPy seed set")

        # PyTorch CPU
        torch.manual_seed(self.seed)
        logger.debug("✓ PyTorch CPU seed set")

        # PyTorch CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # For multi-GPU
            logger.debug("✓ PyTorch CUDA seeds set")

        # Set deterministic behavior
        if self.deterministic:
            self.set_deterministic_mode()

    def set_deterministic_mode(self) -> None:
        """
        Enable deterministic mode for PyTorch

        Warning: This may reduce performance but ensures reproducibility
        """
        logger.info("Enabling deterministic mode for PyTorch")

        # PyTorch deterministic operations
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.debug("✓ PyTorch deterministic algorithms enabled")

        # CuDNN settings
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug("✓ CuDNN deterministic mode enabled")
            logger.warning("CuDNN benchmark disabled - may reduce training speed")

        # Set environment variables for further determinism
        self._set_deterministic_env_vars()

    def _set_deterministic_env_vars(self) -> None:
        """Set environment variables for deterministic behavior"""
        deterministic_env_vars = {
            'PYTHONHASHSEED': str(self.seed),
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',  # For CUDA operations
        }

        for key, value in deterministic_env_vars.items():
            if key not in os.environ:
                self._original_env[key] = os.environ.get(key)
                os.environ[key] = value
                logger.debug(f"✓ Set {key}={value}")

    def restore_env(self) -> None:
        """Restore original environment variables"""
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        logger.debug("Environment variables restored")

    def worker_init_fn(self, worker_id: int) -> None:
        """
        Worker initialization function for DataLoader

        Ensures each worker has a unique but reproducible seed

        Args:
            worker_id: Worker process ID

        Usage:
            >>> DataLoader(dataset, worker_init_fn=seed_manager.worker_init_fn)
        """
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        logger.debug(f"Worker {worker_id} initialized with seed {worker_seed}")

    def get_generator(self, device: Optional[torch.device] = None) -> torch.Generator:
        """
        Get a seeded PyTorch Generator

        Args:
            device: Device for the generator (cpu, cuda, etc.)

        Returns:
            Seeded torch.Generator

        Usage:
            >>> generator = seed_manager.get_generator()
            >>> torch.randint(0, 100, (10,), generator=generator)
        """
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        return generator

    def get_context_info(self) -> dict:
        """
        Get information about the current seeding context

        Returns:
            Dictionary with seeding information
        """
        info = {
            'seed': self.seed,
            'deterministic': self.deterministic,
            'python_version': f"{random.getstate()[0]}",
            'numpy_version': f"{np.random.get_state()[0]}",
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cudnn_available': torch.backends.cudnn.is_available(),
        }

        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
            })

        return info

    def validate_reproducibility(self, tolerance: float = 1e-6) -> bool:
        """
        Validate that operations are reproducible

        Args:
            tolerance: Numerical tolerance for comparison

        Returns:
            True if reproducible, False otherwise
        """
        logger.info("Validating reproducibility...")

        # Test with simple operations
        self.set_all_seeds()

        # Python random
        r1 = [random.random() for _ in range(10)]
        self.set_all_seeds()
        r2 = [random.random() for _ in range(10)]

        if r1 != r2:
            logger.error("Python random is not reproducible!")
            return False
        logger.debug("✓ Python random reproducible")

        # NumPy
        self.set_all_seeds()
        n1 = np.random.rand(10)
        self.set_all_seeds()
        n2 = np.random.rand(10)

        if not np.allclose(n1, n2, atol=tolerance):
            logger.error("NumPy random is not reproducible!")
            return False
        logger.debug("✓ NumPy random reproducible")

        # PyTorch
        self.set_all_seeds()
        t1 = torch.rand(10)
        self.set_all_seeds()
        t2 = torch.rand(10)

        if not torch.allclose(t1, t2, atol=tolerance):
            logger.error("PyTorch random is not reproducible!")
            return False
        logger.debug("✓ PyTorch random reproducible")

        logger.info("✓ All reproducibility checks passed!")
        return True

    def __repr__(self) -> str:
        return f"SeedManager(seed={self.seed}, deterministic={self.deterministic})"

    def __enter__(self):
        """Context manager entry"""
        self.set_all_seeds()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.deterministic:
            self.restore_env()


def set_global_seed(seed: int = 42, deterministic: bool = True) -> SeedManager:
    """
    Convenience function to set global seed

    Args:
        seed: Random seed value
        deterministic: Enable deterministic mode

    Returns:
        SeedManager instance

    Example:
        >>> seed_manager = set_global_seed(42)
    """
    seed_manager = SeedManager(seed=seed, deterministic=deterministic)
    seed_manager.set_all_seeds()
    return seed_manager


def create_reproducible_dataloader(
    dataset,
    seed: int = 42,
    **dataloader_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with reproducible behavior

    Args:
        dataset: PyTorch Dataset
        seed: Random seed
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader with reproducible worker initialization

    Example:
        >>> loader = create_reproducible_dataloader(
        ...     dataset,
        ...     seed=42,
        ...     batch_size=32,
        ...     num_workers=4
        ... )
    """
    seed_manager = SeedManager(seed=seed)

    # Set worker_init_fn if num_workers > 0
    if dataloader_kwargs.get('num_workers', 0) > 0:
        dataloader_kwargs['worker_init_fn'] = seed_manager.worker_init_fn

    # Set generator for reproducible sampling
    dataloader_kwargs['generator'] = seed_manager.get_generator()

    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Seed Manager - Reproducibility Test")
    print("=" * 60)

    # Test 1: Basic seed setting
    print("\n1. Testing basic seed setting...")
    seed_manager = SeedManager(seed=42, deterministic=True)
    seed_manager.set_all_seeds()
    print(f"   Seed manager created: {seed_manager}")

    # Test 2: Context info
    print("\n2. Getting context information...")
    info = seed_manager.get_context_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Test 3: Reproducibility validation
    print("\n3. Validating reproducibility...")
    is_reproducible = seed_manager.validate_reproducibility()
    print(f"   Reproducibility: {'PASS ✓' if is_reproducible else 'FAIL ✗'}")

    # Test 4: Worker initialization
    print("\n4. Testing worker initialization...")
    for worker_id in range(3):
        seed_manager.worker_init_fn(worker_id)
    print("   Worker initialization complete")

    # Test 5: Generator
    print("\n5. Testing generator...")
    generator = seed_manager.get_generator()
    random_numbers = torch.randint(0, 100, (5,), generator=generator)
    print(f"   Random numbers: {random_numbers.tolist()}")

    # Test 6: Context manager
    print("\n6. Testing context manager...")
    with SeedManager(seed=123) as sm:
        print(f"   Inside context: {sm}")
        val = random.random()
        print(f"   Random value: {val}")
    print("   Context manager exited")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
