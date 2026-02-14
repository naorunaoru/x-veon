"""Backward-compatibility shim — all CFA utilities now live in cfa.py."""
from cfa import XTRANS_PATTERN, make_cfa_mask, make_channel_masks, mosaic

__all__ = ["XTRANS_PATTERN", "make_cfa_mask", "make_channel_masks", "mosaic"]
