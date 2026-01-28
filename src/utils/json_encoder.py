"""
Custom JSON Encoder for Hindi BabyLM

This module provides a custom JSON encoder that handles non-serializable types
commonly encountered in machine learning projects, including numpy types,
dataclasses, and datetime objects.
"""

import json
import numpy as np
from datetime import datetime
from dataclasses import is_dataclass, asdict


class DataclassJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles dataclass objects and other non-serializable types"""

    def default(self, obj):
        # Handle dataclass objects
        if is_dataclass(obj):
            return asdict(obj)

        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Handle numpy types
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle other common non-serializable types
        if hasattr(obj, '__dict__'):
            return obj.__dict__

        # Let the base class handle other types or raise TypeError
        return super().default(obj)
