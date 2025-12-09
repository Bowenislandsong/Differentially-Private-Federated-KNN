"""Fixed-point arithmetic utilities for secure computation.

Adapted from FastLloyd: https://github.com/D-Diaa/FastLloyd
"""

import numpy as np

# Modulus for fixed-point arithmetic
MOD = 2 ** 31
SCALE = 2 ** 16


def to_fixed(value):
    """Convert floating point value to fixed-point representation.
    
    Args:
        value: Floating point value or array
        
    Returns:
        Integer value or array in fixed-point representation
    """
    scaled = value * SCALE
    # Use int64 first to avoid overflow, then modulo, then convert to int32
    return np.int32(np.int64(scaled) % MOD)


def to_int(value):
    """Convert value to integer modulo MOD.
    
    Args:
        value: Value or array to convert
        
    Returns:
        Integer value or array modulo MOD
    """
    # Use int64 first to avoid overflow, then modulo, then convert to int32
    return np.int32(np.int64(value) % MOD)


def unscale(value):
    """Convert fixed-point value back to floating point.
    
    Args:
        value: Fixed-point value or array
        
    Returns:
        Floating point value or array
    """
    return np.float64(value) / SCALE
