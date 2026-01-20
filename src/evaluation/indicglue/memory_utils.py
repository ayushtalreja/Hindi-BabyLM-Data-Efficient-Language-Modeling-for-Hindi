"""
Memory management utilities for IndicGLUE evaluation.

Provides robust CUDA memory cleanup to prevent NVML assertion errors
that occur during evaluation of multiple tasks.
"""

import gc
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def cleanup_cuda_memory(log: Optional[logging.Logger] = None):
    """
    Robust CUDA memory cleanup to prevent NVML assertion errors.

    This function properly synchronizes CUDA operations before freeing memory,
    which prevents the 'NVML_SUCCESS == r INTERNAL ASSERT FAILED' error.

    The error occurs when torch.cuda.empty_cache() is called while CUDA
    operations are still pending. This function ensures all GPU operations
    complete before modifying memory state.

    Args:
        log: Optional logger for debug messages. If None, uses module logger.

    Usage:
        # After model evaluation
        cleanup_cuda_memory(logger)

        # In finally blocks
        try:
            # ... evaluation code ...
        finally:
            cleanup_cuda_memory()
    """
    _log = log or logger

    # Force Python garbage collection first
    # This frees CPU references to GPU tensors, allowing CUDA to reclaim memory
    gc.collect()

    if torch.cuda.is_available():
        # CRITICAL: Synchronize all CUDA streams before memory operations
        # This ensures all pending GPU operations complete before we modify memory state
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            # Non-critical if sync fails - log and continue
            _log.debug(f"CUDA synchronize warning (non-critical): {e}")

        # Now safe to empty the cache
        torch.cuda.empty_cache()

        # Second garbage collection pass to clean up any freed references
        gc.collect()

        _log.debug("CUDA memory cleanup completed successfully")


def move_model_to_cpu(model: torch.nn.Module, log: Optional[logging.Logger] = None) -> bool:
    """
    Safely move a model to CPU before deletion.

    Moving models to CPU before deletion helps prevent CUDA memory issues
    by ensuring all GPU tensors are transferred before Python garbage
    collection runs.

    Args:
        model: PyTorch model to move to CPU
        log: Optional logger for debug messages

    Returns:
        True if successful, False if an error occurred
    """
    _log = log or logger

    try:
        if hasattr(model, 'cpu'):
            model.cpu()
            _log.debug(f"Moved {model.__class__.__name__} to CPU")
            return True
    except Exception as e:
        _log.debug(f"Failed to move model to CPU (non-critical): {e}")
        return False

    return False


def cleanup_dataloader(dataloader, log: Optional[logging.Logger] = None):
    """
    Clean up a DataLoader to release memory.

    DataLoaders can hold references to data that prevent memory from being freed.
    This function helps ensure proper cleanup.

    Args:
        dataloader: PyTorch DataLoader to clean up
        log: Optional logger for debug messages
    """
    _log = log or logger

    if dataloader is None:
        return

    try:
        # Clear any iterator state
        if hasattr(dataloader, '_iterator'):
            dataloader._iterator = None

        # Clear the dataset reference if possible
        # Note: This is safe because we're done with the dataloader
        if hasattr(dataloader, 'dataset'):
            # Don't delete the dataset itself, just clear our reference
            pass

        _log.debug("DataLoader cleanup completed")
    except Exception as e:
        _log.debug(f"DataLoader cleanup warning (non-critical): {e}")
