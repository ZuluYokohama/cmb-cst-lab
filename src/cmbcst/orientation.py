"""Exact pixel-grid symmetries; not arbitrary celestial rotations or resampling."""
from __future__ import annotations
import numpy as np
from .surrogate_topology import checked_field


def d4_view(field, quarter_turns=0, reflected=False):
    """Apply R^k F^f and return contiguous float64 pixels with no interpolation."""
    x = checked_field(field)
    if (not isinstance(quarter_turns, (int, np.integer)) or isinstance(quarter_turns, (bool, np.bool_))
            or not 0 <= quarter_turns <= 3 or not isinstance(reflected, (bool, np.bool_))):
        raise ValueError('need zero through three quarter turns and a Boolean reflection')
    if reflected:
        x = np.flip(x, axis=1)
    return np.ascontiguousarray(np.rot90(x, k=int(quarter_turns)), dtype=np.float64)


def directional_sign(quarter_turns):
    """Quarter turns swapping horizontal and vertical reverse the contrast sign."""
    if type(quarter_turns) is not int or not 0 <= quarter_turns <= 3:
        raise ValueError('need zero through three quarter turns')
    return -1 if quarter_turns % 2 else 1
