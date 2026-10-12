"""Out-of-tree import shims for GIANT 2.0.0 (no edits to the GIANT source).

1. Python 3.10: GIANT uses ``typing.Self`` (3.11+); alias it from typing_extensions.
2. feature_matchers/__init__.py imports roma_matcher unconditionally, which raises
   ImportError when RoMa/torch are absent; pre-register an empty module so the
   later guarded ``try: from ...roma_matcher import RoMaFeatureMatcher`` fails cleanly.
"""
import sys
import types
import typing

if not hasattr(typing, "Self"):
    import typing_extensions
    typing.Self = typing_extensions.Self

_ROMA = "giant.image_processing.feature_matchers.roma_matcher"
sys.modules.setdefault(_ROMA, types.ModuleType(_ROMA))
