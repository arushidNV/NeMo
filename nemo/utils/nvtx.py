# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
from contextlib import contextmanager
from typing import Optional

import torch

from nemo.utils.app_state import AppState

# pylint: disable=C0116


_TRUTHY = {"1", "true", "True", "TRUE", "yes", "on"}


@functools.lru_cache(maxsize=None)
def _nvtx_enabled() -> bool:
    """Check if NVTX range profiling is enabled.

    Returns True if any of the following is true:
      - AppState()._nvtx_ranges is True (legacy flag)
      - Env var NEMO_NVTX is truthy ("1"/"true"/"yes"/"on")
      - Env var RIVA_NVTX is truthy (so the same flag toggles the Riva pipeline
        and the NeMo cache-aware path together)
    Cached after first call. Set the env var BEFORE importing nemo.utils.nvtx.
    """
    if AppState()._nvtx_ranges:
        return True
    for var in ("NEMO_NVTX", "RIVA_NVTX"):
        if os.environ.get(var, "") in _TRUTHY:
            return True
    return False


@functools.lru_cache(maxsize=None)
def _nvtx_sync_enabled() -> bool:
    """If true, NVTX pop performs a CUDA stream sync first.

    Gives wall-clock timing per range at the cost of serializing the pipeline.
    Toggle via env var RIVA_NVTX_SYNC=1 or NEMO_NVTX_SYNC=1. Off by default.
    """
    for var in ("RIVA_NVTX_SYNC", "NEMO_NVTX_SYNC"):
        if os.environ.get(var, "") in _TRUTHY:
            return True
    return False


# Messages associated with active NVTX ranges
_nvtx_range_messages: list[str] = []


def nvtx_range_push(msg: str) -> None:
    if not _nvtx_enabled():
        return

    _nvtx_range_messages.append(msg)
    torch.cuda.nvtx.range_push(msg)


def nvtx_range_pop(msg: Optional[str] = None) -> None:
    if not _nvtx_enabled():
        return

    if not _nvtx_range_messages:
        raise RuntimeError("Attempted to pop NVTX range from empty stack")
    last_msg = _nvtx_range_messages.pop()
    if msg is not None and msg != last_msg:
        raise ValueError(
            f"Attempted to pop NVTX range from stack with msg={msg}, " f"but last range has msg={last_msg}"
        )

    if _nvtx_sync_enabled() and torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


@contextmanager
def nvtx_range(msg: str):
    """Context manager wrapper for NVTX push/pop with exception safety.

    Usage:
        with nvtx_range("Niva_execute"):
            ...
    """
    nvtx_range_push(msg)
    try:
        yield
    finally:
        nvtx_range_pop(msg)


def nvtx_decorator(msg: Optional[str] = None):
    """Decorator that wraps a function in an NVTX range.

    If msg is None, uses the qualified function name as the range label.

    Usage:
        @nvtx_decorator("get_context")
        def get_context(self, ...):
            ...

        @nvtx_decorator()  # auto-labels with qualified name
        def update_cache(self, ...):
            ...
    """

    def deco(fn):
        label = msg if msg is not None else getattr(fn, "__qualname__", fn.__name__)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _nvtx_enabled():
                return fn(*args, **kwargs)
            nvtx_range_push(label)
            try:
                return fn(*args, **kwargs)
            finally:
                nvtx_range_pop(label)

        return wrapper

    return deco


# pylint: enable=C0116
