"""Continuo — the continuity supervisor for AI filmmaking.

A model-agnostic, post-generation QC layer that screens every generated shot
against a registered asset bible, flags drift frame-accurately, and rewrites the
offending prompt in the grammar of whichever model produced it.
"""

__version__ = "0.1.0"
