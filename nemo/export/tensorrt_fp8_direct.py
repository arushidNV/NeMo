# Copyright (c) NVIDIA Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TensorRT FP8 Direct Builder for NeMo models.

Drop-in replacement for tensorrt_fp8_compiler.py that uses triton.py's
DIRECT TRT API builder instead of polygraphy. This gives TRT full freedom
to use auxiliary streams for parallel kernel execution (matching CTC's
build path that achieves 1.20x FP8 speedup via 4 aux streams).

Key differences from tensorrt_fp8_compiler.py:
- Uses direct TRT Builder API (not polygraphy) for engine building
- Adds quant_pre_process() before ModelOpt quantization (matching triton.py)
- No precision_constraints="obey" — TRT has full scheduling freedom
- No builder_optimization_level override — uses TRT default (3)
- No max_aux_streams override — lets TRT heuristic decide
- profiling_verbosity=DETAILED — matching triton.py
- Calibration data preserved for accuracy

FP8 requires:
- Ada Lovelace GPU or newer (compute capability >= 8.9)
- TensorRT 8.6+
- nvidia-modelopt package for FP8 quantization

Usage:
    from nemo.export.tensorrt_fp8_direct import trt_fp8_compile
    
    model = trt_fp8_compile(
        model.encoder,
        base_path="/path/to/encoder",
        args={
            "input_names": ["audio_signal", "length"],
            "input_profiles": [...],
            "build_args": {
                "calibration_audio_dir": "/path/to/calibration_wavs",
            },
        }
    )
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from logging import getLogger
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from nemo.utils.export_utils import add_casts_around_norms, replace_for_export
from nemo.utils.import_utils import safe_import

# Import TensorRT and related packages
trt, trt_imported = safe_import("tensorrt")

# Import ONNX packages
onnx, onnx_imported = safe_import("onnx")
gs, gs_imported = safe_import("onnx_graphsurgeon")

# Import ModelOpt for FP8 quantization
modelopt_quantization, modelopt_imported = safe_import("modelopt.onnx.quantization")

# Import ONNX Runtime for pre-processing
onnxruntime_quant, ort_imported = safe_import("onnxruntime.quantization.shape_inference")

lock_sm = threading.Lock()

# Logger for this module
logger = getLogger("trt_fp8_direct")


# =============================================================================
# Utility Functions (from triton.py)
# =============================================================================

def get_compute_capability() -> Tuple[int, int]:
    """
    Get the compute capability of the current CUDA device.
    
    Returns:
        Tuple of (major, minor) compute capability
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return (props.major, props.minor)
    return (0, 0)


def check_fp8_support(skip_modelopt: bool = False) -> bool:
    """
    Check if FP8 is supported on the current system.
    
    Args:
        skip_modelopt: If True, don't require ModelOpt
    
    Returns:
        True if FP8 is supported, False otherwise
    """
    # Check compute capability (need >= 8.9 for FP8)
    cc = get_compute_capability()
    if cc < (8, 9):
        logger.warning(
            f"FP8 requires compute capability >= 8.9 (Ada Lovelace or newer). "
            f"Current device has {cc[0]}.{cc[1]}"
        )
        return False
    
    # Check TensorRT version
    if not trt_imported:
        logger.warning("TensorRT is not available")
        return False
    
    # Check if TRT has FP8 support
    if not hasattr(trt.BuilderFlag, 'FP8'):
        logger.warning("TensorRT version does not support FP8 (need TRT 8.6+)")
        return False
    
    # Check ModelOpt (unless skip_modelopt is True)
    if not skip_modelopt and not modelopt_imported:
        logger.warning("nvidia-modelopt is not available. Install with: pip install nvidia-modelopt")
        return False
    
    return True


def trt_to_torch_dtype_dict():
    """
    Map of TRT dtype -> Torch dtype
    """
    if not trt_imported:
        return {}
    
    dtype_map = {
        trt.int32: torch.int32,
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    # Add bfloat16 if available
    if hasattr(trt, 'bfloat16'):
        dtype_map[trt.bfloat16] = torch.bfloat16
    # Add fp8 types if available (map to float16 for output tensors)
    if hasattr(trt, 'fp8'):
        dtype_map[trt.fp8] = torch.float16
    return dtype_map


def get_dynamic_axes(profiles):
    """
    Calculate dynamic_axes for onnx.export() from profiles.
    
    Args:
        profiles: List of profile dictionaries with [min, opt, max] shapes
    
    Returns:
        Dictionary mapping input names to list of dynamic axis indices
    """
    dynamic_axes: dict[str, list[int]] = {}
    if not profiles:
        return dynamic_axes
    for profile in profiles:
        for key in profile:
            axes = []
            vals = profile[key]
            for i in range(len(vals[0])):
                if vals[0][i] != vals[2][i]:
                    axes.append(i)
            if len(axes) > 0:
                dynamic_axes[key] = axes
    return dynamic_axes


def make_tensor(d):
    """
    Creates a new tensor from d, returns d if d is already a tensor
    """
    return d if isinstance(d, torch.Tensor) else torch.tensor(d).cuda()


def unroll_input(input_names, input_example):
    """
    Simulates list/tuple unrolling during ONNX export
    """
    unrolled_input = {}
    for name in input_names:
        val = input_example.get(name)
        if val is not None:
            if isinstance(val, list) or isinstance(val, tuple):
                for i in range(len(val)):
                    unrolled_input[f"{name}_{i}"] = make_tensor(val[i])
            else:
                unrolled_input[name] = make_tensor(val)
    return unrolled_input


def parse_groups(
    ret: List[torch.Tensor], output_lists: List[List[int]]
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]:
    """
    Implements parsing of 'output_lists' arg of trt_compile().
    """
    groups: Tuple[Union[torch.Tensor, List[torch.Tensor]], ...] = tuple()
    cur = 0
    for i in range(len(output_lists)):
        gl = output_lists[i]
        assert len(gl) == 0 or len(gl) == 1
        if len(gl) == 0 or gl[0] == 0:
            groups = (*groups, ret[cur])
            cur = cur + 1
        elif gl[0] > 0:
            groups = (*groups, ret[cur : cur + gl[0]])
            cur = cur + gl[0]
        elif gl[0] == -1:
            rev_groups: Tuple[Union[torch.Tensor, List[torch.Tensor]], ...] = tuple()
            rcur = len(ret)
            for rl in range(len(output_lists) - 1, i, -1):
                rgl = output_lists[rl]
                assert len(rgl) == 0 or len(rgl) == 1
                if len(rgl) == 0 or rgl[0] == 0:
                    rcur = rcur - 1
                    rev_groups = (*rev_groups, ret[rcur])
                elif rgl[0] > 0:
                    rcur = rcur - rgl[0]
                    rev_groups = (*rev_groups, ret[rcur : rcur + rgl[0]])
                else:
                    raise ValueError("Two -1 lists in output")
            groups = (*groups, ret[cur:rcur], *rev_groups[::-1])
            break
    return groups


class ShapeError(Exception):
    """
    Exception class to report errors from setting TRT plan input shapes
    """
    pass


# =============================================================================
# TRTEngine Class (for running inference) - Direct TRT API
# =============================================================================

class TRTEngine:
    """
    An auxiliary class to implement running of TRT optimized engines.
    Uses direct TensorRT API (not polygraphy).
    """

    def __init__(self, plan_path, logger=None):
        """
        Loads serialized engine, creates execution context and activates it
        
        Args:
            plan_path: Path to serialized TRT engine
            logger: Optional logger object
        """
        if not trt_imported:
            raise ImportError("TensorRT is required but not available")
        
        self.plan_path = plan_path
        self.logger = logger or getLogger("trt_fp8_direct")
        self.logger.info(f"Loading TensorRT engine: {self.plan_path}")
        
        # Load engine using direct TRT API
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(plan_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {plan_path}")
        
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        self.dtypes = []
        self.cur_profile = 0
        self.input_table = {}
        
        dtype_dict = trt_to_torch_dtype_dict()
        
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_names.append(binding)
            elif self.engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(binding)
                trt_dtype = self.engine.get_tensor_dtype(binding)
                # Handle FP8 output dtype (map to FP16)
                if trt_dtype in dtype_dict:
                    dtype = dtype_dict[trt_dtype]
                else:
                    dtype = torch.float16  # Default fallback
                self.dtypes.append(dtype)
        
        self.logger.info(
            f"Loaded TensorRT engine: {self.plan_path}\n"
            f"Inputs: {self.input_names}\nOutputs: {self.output_names}"
        )

    def allocate_buffers(self, device):
        """
        Allocates outputs to run TRT engine
        """
        ctx = self.context
        
        for i, binding in enumerate(self.output_names):
            shape = list(ctx.get_tensor_shape(binding))
            if binding not in self.tensors or list(self.tensors[binding].shape) != shape:
                t = torch.empty(shape, dtype=self.dtypes[i], device=device).contiguous()
                self.tensors[binding] = t
            t = self.tensors[binding]
            ctx.set_tensor_address(binding, t.data_ptr())

    def set_inputs(self, feed_dict, stream):
        """
        Sets input bindings for TRT engine according to feed_dict
        """
        e = self.engine
        ctx = self.context
        last_profile = self.cur_profile

        def try_set_inputs():
            for binding in self.input_names:
                t = feed_dict.get(self.input_table.get(binding, binding), None)
                if t is not None:
                    t = t.contiguous()
                    shape = t.shape
                    ctx.set_input_shape(binding, shape)
                    ctx.set_tensor_address(binding, t.data_ptr())

        while True:
            try:
                try_set_inputs()
                break
            except ShapeError:
                next_profile = (self.cur_profile + 1) % e.num_optimization_profiles
                if next_profile == last_profile:
                    raise
                self.cur_profile = next_profile
                ctx.set_optimization_profile_async(self.cur_profile, stream)
            except Exception:
                raise
        
        left = ctx.infer_shapes()
        assert len(left) == 0

    def infer(self, stream):
        """
        Runs TRT engine.
        """
        noerror = self.context.execute_async_v3(stream)
        torch.cuda.synchronize()
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        return self.tensors


# =============================================================================
# TrtFP8DirectCompiler Class - Uses Direct TRT API (matching triton.py)
# =============================================================================

class TrtFP8DirectCompiler:
    """
    TensorRT FP8 Compiler using Direct TRT API (not polygraphy).
    
    Matches triton.py's proven build path that gives CTC 4 aux streams:
    - Uses onnx_graphsurgeon for constant folding
    - Uses quant_pre_process() before ModelOpt (matching triton.py)
    - Uses DIRECT TRT Builder API for engine building (not polygraphy)
    - No precision_constraints — TRT has full scheduling freedom
    - No builder_optimization_level override — uses TRT default
    - Calibration data preserved for accuracy
    """

    def __init__(
        self,
        model,
        plan_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        output_lists: Optional[List[List[int]]] = None,
        export_args: Optional[Dict[str, Any]] = None,
        build_args: Optional[Dict[str, Any]] = None,
        input_profiles: Optional[List[Dict]] = None,
        dynamic_batchsize: Optional[Sequence[int]] = None,
        use_cuda_graph: bool = False,
        timestamp: Optional[float] = None,
        fallback: bool = False,
        op_types_to_exclude: Optional[List[str]] = None,
        nodes_to_exclude: Optional[List[str]] = None,
        skip_modelopt: bool = False,
        verbose: bool = False,
        logger=None,
    ):
        """
        Initialize the FP8 Direct TRT Compiler.
        
        Args:
            model: PyTorch model to compile
            plan_path: Path to save the TRT engine (.plan file)
            input_names: List of input tensor names
            output_names: List of output tensor names
            output_lists: Output grouping specification
            export_args: Arguments for torch.onnx.export()
            build_args: Arguments for TRT builder (calibration_audio_dir, etc.)
            input_profiles: List of input shape profiles for dynamic shapes
            dynamic_batchsize: [min, opt, max] batch sizes
            use_cuda_graph: Enable CUDA graph for inference
            timestamp: Timestamp for cache invalidation
            fallback: Fall back to PyTorch if TRT fails
            op_types_to_exclude: Op types to exclude from FP8 quantization
            nodes_to_exclude: ONNX node name regex patterns to exclude from FP8.
            skip_modelopt: Skip ModelOpt FP8 quantization and use native TRT FP8
            verbose: Enable verbose TRT logging
            logger: Logger instance
        """
        # Check FP8 support
        if not check_fp8_support(skip_modelopt=skip_modelopt):
            raise RuntimeError(
                "FP8 is not supported on this system. "
                "Requires Ada Lovelace GPU (compute >= 8.9), TensorRT 8.6+"
            )
        
        self.plan_path = plan_path
        self.precision = "fp8"
        self.return_dict = output_names is not None
        self.output_names = output_names or []
        self.output_lists = output_lists or []
        self.profiles = input_profiles or []
        self.dynamic_batchsize = dynamic_batchsize
        self.export_args = export_args or {}
        self.build_args = build_args or {}
        self.engine: TRTEngine | None = None
        self.use_cuda_graph = use_cuda_graph
        self.fallback = fallback
        self.disabled = False
        self.op_types_to_exclude = op_types_to_exclude or ["Conv"]
        # Match triton.py: no nodes excluded by name — all MatMuls get FP8 quantized
        self.nodes_to_exclude = nodes_to_exclude if nodes_to_exclude is not None else []
        self.skip_modelopt = skip_modelopt
        self.verbose = verbose
        
        self.logger = logger or getLogger("trt_fp8_direct")
        self.argspec = inspect.getfullargspec(model.forward)
        
        # Get input names from function signature if not provided
        if input_names is None:
            input_names = self.argspec.args[1:]
        
        self.defaults = {}
        if self.argspec.defaults is not None:
            for i in range(len(self.argspec.defaults)):
                d = self.argspec.defaults[-i - 1]
                if d is not None:
                    d = make_tensor(d)
                    self.defaults[self.argspec.args[-i - 1]] = d
        
        self.input_names = input_names
        self.old_forward = model.forward
        
        # Force engine rebuild if older than timestamp
        if timestamp is not None and os.path.exists(self.plan_path) and os.path.getmtime(self.plan_path) < timestamp:
            os.remove(self.plan_path)

    def _inputs_to_dict(self, input_example):
        """Convert positional inputs to dictionary."""
        trt_inputs = {}
        for i, inp in enumerate(input_example):
            input_name = self.input_names[i]
            trt_inputs[input_name] = inp
        return trt_inputs

    def _load_engine(self):
        """
        Load TRT plan from disk and activate execution context.
        """
        if not os.path.exists(self.plan_path):
            self.logger.info(f"Engine file not found: {self.plan_path}")
            return
        
        try:
            self.engine = TRTEngine(self.plan_path, self.logger)
            input_table = {}
            for name in self.engine.input_names:
                if name.startswith("__") and name not in self.input_names:
                    orig_name = name[2:]
                else:
                    orig_name = name
                input_table[name] = orig_name
            self.engine.input_table = input_table
            self.logger.info(f"FP8 Direct Engine loaded, inputs: {self.engine.input_table}")
        except Exception as e:
            self.logger.info(f"Exception while loading the engine:\n{e}")

    def forward(self, model, argv, kwargs):
        """
        Main forward method with lazy TRT build.
        """
        args = self.defaults.copy()
        args.update(kwargs)
        if len(argv) > 0:
            args.update(self._inputs_to_dict(argv))

        if self.engine is None and not self.disabled:
            new_forward = model.forward
            model.forward = self.old_forward
            try:
                self._load_engine()
                if self.engine is None:
                    build_args = args.copy()
                    with torch.no_grad():
                        self._build_and_save(model, build_args)
                    self._load_engine()
                    assert self.engine is not None
            except Exception as e:
                if self.fallback:
                    self.logger.info(f"Failed to build FP8 engine: {e}")
                    import traceback
                    traceback.print_exc()
                    self.disabled = True
                else:
                    raise e
            if not self.disabled and not self.fallback:
                for param in model.parameters():
                    del param
                torch.cuda.empty_cache()
            model.forward = new_forward
        
        # Run inference
        try:
            if self.engine is not None:
                with lock_sm:
                    device = torch.cuda.current_device()
                    stream = torch.cuda.Stream(device=device)
                    
                    # Prepare inputs
                    inputs_dict = unroll_input(self.input_names, args)
                    self.engine.set_inputs(inputs_dict, stream.cuda_stream)
                    self.engine.allocate_buffers(device=device)
                    stream.wait_stream(torch.cuda.current_stream())
                    
                    # Run inference
                    ret = self.engine.infer(stream.cuda_stream)
                    
                    if not self.return_dict:
                        ret = list(ret.values())
                        if self.output_lists:
                            ret = parse_groups(ret, self.output_lists)
                        elif len(ret) == 1:
                            ret = ret[0]
                    return ret
        except Exception as e:
            if self.fallback:
                self.logger.info(f"Exception: {e}\nFalling back to PyTorch...")
            else:
                raise e
        
        return self.old_forward(*argv, **kwargs)

    # =========================================================================
    # Calibration helpers (unchanged from tensorrt_fp8_compiler.py)
    # =========================================================================

    def _prepare_mel_for_calibration(self, mel_data, target_shape: tuple):
        """Prepare mel spectrogram for calibration by padding/cropping to target shape."""
        import numpy as np
        
        if mel_data.ndim == 1:
            mel_data = mel_data.reshape(1, -1)
        if mel_data.ndim == 2:
            mel_data = mel_data[np.newaxis, ...]
        
        result = np.zeros(target_shape, dtype=np.float32)
        src_shape = mel_data.shape
        ndim = min(len(src_shape), len(target_shape))
        
        slices_src = []
        slices_dst = []
        for i in range(ndim):
            size = min(src_shape[i], target_shape[i])
            slices_src.append(slice(0, size))
            slices_dst.append(slice(0, size))
        for i in range(ndim, len(target_shape)):
            slices_dst.append(slice(0, 1))
        
        try:
            result[tuple(slices_dst)] = mel_data[tuple(slices_src)]
        except Exception:
            result.fill(mel_data.mean())
        
        return result

    def _get_onnx_input_dtypes(self, onnx_path: str) -> dict:
        """Get input dtypes from an ONNX model."""
        import numpy as np
        
        dtype_map = {}
        try:
            model = onnx.load(onnx_path)
            for inp in model.graph.input:
                name = inp.name
                if inp.type.HasField('tensor_type'):
                    elem_type = inp.type.tensor_type.elem_type
                    onnx_to_numpy = {
                        1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16,
                        5: np.int16, 6: np.int32, 7: np.int64, 9: np.bool_,
                        10: np.float16, 11: np.float64, 12: np.uint32, 13: np.uint64,
                    }
                    dtype_map[name] = onnx_to_numpy.get(elem_type, np.float32)
        except Exception as e:
            self.logger.warning(f"  Failed to get ONNX input dtypes: {e}")
        
        return dtype_map

    def _log_matching_nodes(self, onnx_path: str):
        """Log which ONNX MatMul/Gemm nodes match nodes_to_exclude patterns."""
        import re
        
        if not self.nodes_to_exclude:
            self.logger.warning("  nodes_to_exclude is empty - all MatMul/Gemm nodes will be FP8 quantized")
            return
        
        try:
            model = onnx.load(onnx_path, load_external_data=False)
            matmul_nodes = [n.name for n in model.graph.node if n.op_type in ("MatMul", "Gemm")]
            
            excluded = []
            kept = []
            for node_name in matmul_nodes:
                matched = False
                for pattern in self.nodes_to_exclude:
                    if re.search(pattern, node_name):
                        excluded.append((node_name, pattern))
                        matched = True
                        break
                if not matched:
                    kept.append(node_name)
            
            self.logger.warning(f"  === Node Exclusion Verification ===")
            self.logger.warning(f"  ONNX MatMul/Gemm nodes: {len(matmul_nodes)} total")
            self.logger.warning(f"  Excluded from FP8: {len(excluded)}")
            for name, pattern in excluded[:15]:
                self.logger.warning(f"    EXCLUDE: {name}  (matched: {pattern})")
            if len(excluded) > 15:
                self.logger.warning(f"    ... and {len(excluded) - 15} more")
            self.logger.warning(f"  Kept for FP8: {len(kept)}")
            for name in kept[:10]:
                self.logger.warning(f"    FP8: {name}")
            if len(kept) > 10:
                self.logger.warning(f"    ... and {len(kept) - 10} more")
            
            del model
        except Exception as e:
            self.logger.warning(f"  Could not verify node exclusions: {e}")

    def _build_calib_sample(self, profile: dict, mel_data, log_first: bool = False, input_dtypes: dict = None):
        """Build a single calibration sample dict from mel spectrogram data."""
        import numpy as np
        input_dtypes = input_dtypes or {}
        
        sample = {}
        for input_name, shapes in profile.items():
            opt_shape = tuple(shapes[1])
            input_lower = input_name.lower()
            
            if input_lower == "length" or input_lower.endswith("_len"):
                dtype = input_dtypes.get(input_name, np.int64)
            else:
                dtype = input_dtypes.get(input_name, np.float32)
            
            if "audio" in input_lower or "signal" in input_lower:
                data = self._prepare_mel_for_calibration(mel_data, opt_shape)
                data = data.astype(dtype)
                if log_first:
                    self.logger.info(f"    {input_name}: shape={data.shape}, dtype={data.dtype}")
            elif input_lower == "length" or input_lower.endswith("_len"):
                mel_len = mel_data.shape[-1] if mel_data.ndim >= 2 else mel_data.shape[0]
                val = min(mel_len, opt_shape[-1] if len(opt_shape) > 0 else mel_len)
                data = np.full(opt_shape, val, dtype=dtype)
            elif "cache" in input_lower:
                data = np.zeros(opt_shape, dtype=dtype)
            else:
                data = np.zeros(opt_shape, dtype=dtype)
            
            sample[input_name] = data
        return sample

    def _build_synthetic_calib_sample(self, profile: dict, input_dtypes: dict = None):
        """Build a synthetic calibration sample (fallback when no real data)."""
        import numpy as np
        input_dtypes = input_dtypes or {}
        
        sample = {}
        for input_name, shapes in profile.items():
            opt_shape = tuple(shapes[1])
            input_lower = input_name.lower()
            
            if input_lower == "length" or input_lower.endswith("_len"):
                dtype = input_dtypes.get(input_name, np.int64)
            else:
                dtype = input_dtypes.get(input_name, np.float32)
            
            if input_lower == "length" or input_lower.endswith("_len"):
                min_len = shapes[0][-1] if len(shapes[0]) > 0 else 1
                max_len = shapes[2][-1] if len(shapes[2]) > 0 else opt_shape[-1]
                length_val = np.random.randint(min_len, max_len + 1)
                data = np.full(opt_shape, length_val, dtype=dtype)
            elif "audio" in input_lower or "signal" in input_lower:
                data = np.random.normal(loc=-5.0, scale=3.0, size=opt_shape).astype(np.float32)
                data = np.clip(data, -15.0, 5.0).astype(dtype)
            elif "cache" in input_lower:
                data = np.zeros(opt_shape, dtype=dtype)
            else:
                data = np.random.normal(loc=0.0, scale=1.0, size=opt_shape).astype(dtype)
            
            sample[input_name] = data
        return sample

    def _create_mel_filterbank(self, sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float = None):
        """Create mel filterbank matrix using numpy."""
        import numpy as np
        if fmax is None:
            fmax = sr / 2.0
        
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        
        mel_low = hz_to_mel(fmin)
        mel_high = hz_to_mel(fmax)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        n_freqs = n_fft // 2 + 1
        fft_freqs = np.linspace(0, sr / 2.0, n_freqs)
        
        filterbank = np.zeros((n_mels, n_freqs))
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            for j, freq in enumerate(fft_freqs):
                if left <= freq < center:
                    filterbank[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filterbank[i, j] = (right - freq) / (right - center)
        return filterbank.astype(np.float32)

    def _compute_mel_spectrogram(self, audio, n_fft: int, hop_length: int, mel_fb):
        """Compute log mel spectrogram using numpy."""
        import numpy as np
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        window = np.hanning(n_fft).astype(np.float32)
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            stft[:, i] = np.abs(spectrum).astype(np.float32)
        
        mel_spec = np.dot(mel_fb, stft ** 2)
        log_mel = np.log(mel_spec + 1e-9)
        return log_mel

    def _restore_dynamic_shapes(self, onnx_path: str, profiles: List[Dict]) -> str:
        """Restore dynamic shapes in an ONNX model after ModelOpt quantization."""
        self.logger.info("  Restoring dynamic shapes in quantized ONNX model...")
        
        model = onnx.load(onnx_path)
        profile = profiles[0] if profiles else {}
        
        dynamic_dims = {}
        for input_name, shapes in profile.items():
            min_shape, opt_shape, max_shape = shapes
            dynamic_dims[input_name] = {}
            for dim_idx, (min_d, max_d) in enumerate(zip(min_shape, max_shape)):
                if min_d != max_d:
                    dynamic_dims[input_name][dim_idx] = f"{input_name}_dynamic_axes_{dim_idx}"
        
        self.logger.info(f"  Dynamic dimensions to restore: {dynamic_dims}")
        
        inputs_modified = 0
        for input_tensor in model.graph.input:
            input_name = input_tensor.name
            if input_name not in dynamic_dims:
                continue
            tensor_type = input_tensor.type.tensor_type
            if not tensor_type.HasField('shape'):
                continue
            for dim_idx, dim_name in dynamic_dims[input_name].items():
                if dim_idx < len(tensor_type.shape.dim):
                    dim = tensor_type.shape.dim[dim_idx]
                    old_value = dim.dim_value if dim.HasField('dim_value') else "already_dynamic"
                    dim.ClearField('dim_value')
                    dim.dim_param = dim_name
                    self.logger.info(f"    {input_name}[{dim_idx}]: {old_value} -> {dim_name} (dynamic)")
                    inputs_modified += 1
        
        if inputs_modified > 0:
            try:
                external_data_path = os.path.splitext(onnx_path)[0] + "_data"
                onnx.save(
                    model, onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=os.path.basename(external_data_path)
                )
            except Exception as e:
                self.logger.warning(f"  Failed to save with external data: {e}, trying regular save")
                onnx.save(model, onnx_path)
            self.logger.info(f"  Restored {inputs_modified} dynamic dimensions in ONNX model")
        else:
            self.logger.info("  No dimensions needed restoration (already dynamic)")
        
        return onnx_path

    # =========================================================================
    # DIRECT TRT API ENGINE BUILDER (matching triton.py lines 706-828)
    # =========================================================================

    def _build_trt_engine_direct(
        self,
        onnx_path: str,
        opt_profiles: List[Dict],
        use_fp8: bool = True,
    ) -> bytes:
        """
        Build TRT engine from ONNX file using DIRECT TRT API (matching triton.py).
        
        Key differences from polygraphy-based builder:
        - No precision_constraints — TRT has full scheduling freedom
        - No builder_optimization_level override — uses TRT default (3)
        - profiling_verbosity=DETAILED — matching triton.py
        - No max_aux_streams — lets TRT heuristic decide
        - Direct trt.Builder / trt.OnnxParser — same code path as CTC in triton.py
        """
        self.logger.info("  Building TRT engine with DIRECT TRT API (matching triton.py)")
        self.logger.info(f"  ONNX path: {onnx_path}")
        self.logger.info(f"  Profiles: {len(opt_profiles)}")
        self.logger.info(f"  FP8 enabled: {use_fp8}")
        self.logger.info(f"  Builder: direct TRT API (no polygraphy)")
        self.logger.info(f"  precision_constraints: NOT SET (TRT has full freedom)")
        self.logger.info(f"  builder_optimization_level: NOT SET (TRT default = 3)")
        self.logger.info(f"  max_aux_streams: NOT SET (TRT heuristic decides)")
        
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if self.verbose else trt.Logger.WARNING)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(explicit_batch) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser, \
             builder.create_builder_config() as config:
            
            # === Flags — match triton.py exactly (lines 710-712) ===
            config.set_flag(trt.BuilderFlag.FP16)
            if use_fp8:
                config.set_flag(trt.BuilderFlag.FP8)
            
            # Detailed profiling verbosity (matching triton.py line 711)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            
            # Workspace (matching triton.py line 712)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 * 1024 * 1024)
            
            # NOTE: No precision_constraints — let TRT optimize freely
            # NOTE: No builder_optimization_level — use TRT default (3)
            # NOTE: No max_aux_streams — let TRT heuristic decide (like triton.py)
            
            # === Parse ONNX ===
            parsed = parser.parse_from_file(onnx_path)
            if not parsed:
                for i in range(parser.num_errors):
                    self.logger.error(f"  ONNX Parser error {i}: {parser.get_error(i)}")
                raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")
            
            self.logger.info(f"  ONNX parsed successfully. Network inputs: {network.num_inputs}, outputs: {network.num_outputs}")
            
            # === Optimization profiles ===
            for profile_spec in opt_profiles:
                profile = builder.create_optimization_profile()
                for input_name, shapes in profile_spec.items():
                    min_s, opt_s, max_s = shapes
                    profile.set_shape(input_name, min_s, opt_s, max_s)
                config.add_optimization_profile(profile)
            
            # === INT32 for length inputs (matching triton.py lines 807-819) ===
            inputs_to_update = ["length", "cache_last_channel_len"]
            for idx in range(network.num_inputs):
                inp = network.get_input(idx)
                if inp.name in inputs_to_update:
                    inp.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
                    inp.dtype = trt.DataType.INT32
                    self.logger.info(f"  Set input '{inp.name}' to INT32 LINEAR format")
            
            # === Build engine ===
            self.logger.info("  Building serialized network (this may take several minutes)...")
            serialized = builder.build_serialized_network(network, config)
            
            if serialized is None:
                raise RuntimeError("TRT builder.build_serialized_network() returned None")
            
            engine_size = len(serialized) if hasattr(serialized, '__len__') else 0
            self.logger.info(f"  TRT engine built successfully. Size: {engine_size / (1024*1024):.2f} MB")
            
            return serialized

    # =========================================================================
    # BUILD AND SAVE (ONNX export + quantize + TRT build)
    # =========================================================================

    def _build_and_save(self, model, input_example):
        """
        Export model to ONNX, quantize to FP8, and build TRT engine using direct TRT API.
        
        Steps:
        1. Export PyTorch model to ONNX
        2. Constant folding (onnx_graphsurgeon + quant_pre_process)
        3. FP8 quantization with ModelOpt (with calibration data)
        4. Build TRT engine with DIRECT TRT API (not polygraphy)
        5. Save engine to disk
        """
        if self.engine is not None:
            return

        export_args = self.export_args.copy()
        
        add_casts_around_norms(model)
        replace_for_export(model)

        # Setup profiles for dynamic shapes
        dbs = self.dynamic_batchsize
        if dbs:
            if len(self.profiles) > 0:
                raise ValueError("Both dynamic_batchsize and input_profiles set!")
            if len(dbs) != 3:
                raise ValueError("dynamic_batchsize must have length 3")
            profile = {}
            for input_id, val in input_example.items():
                if isinstance(val, list) or isinstance(val, tuple):
                    for i in range(len(val)):
                        sh = val[i].shape
                        if len(sh) > 0:
                            sh = sh[1:]
                            profile[f"{input_id}_{i}"] = [[dbs[0], *sh], [dbs[1], *sh], [dbs[2], *sh]]
                elif isinstance(val, torch.Tensor):
                    sh = val.shape
                    if len(sh) > 0:
                        sh = sh[1:]
                        profile[input_id] = [[dbs[0], *sh], [dbs[1], *sh], [dbs[2], *sh]]
            self.profiles = [profile]

        # Calculate dynamic axes for ONNX export
        dynamic_axes = get_dynamic_axes(self.profiles)
        if len(dynamic_axes) > 0:
            export_args["dynamic_axes"] = dynamic_axes
            self.logger.info(f"Using dynamic axes: {dynamic_axes}")

        # Use temporary directory for all intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            # =================================================================
            # STEP 1: Export to ONNX
            # =================================================================
            input_names = list(unroll_input(self.input_names, input_example).keys())
            onnx_path = os.path.join(tmpdir, "model.onnx")
            
            self.logger.info(f"STEP 1: Exporting to ONNX")
            self.logger.info(f"  Path: {onnx_path}")
            self.logger.info(f"  Input names: {input_names}")
            self.logger.info(f"  Output names: {self.output_names}")
            
            opset_version = None
            
            torch.onnx.export(
                model,
                (input_example,),
                onnx_path,
                input_names=input_names,
                output_names=self.output_names if self.output_names else None,
                opset_version=opset_version,
                **export_args,
            )
            self.logger.info("  ONNX export successful")
            
            # =================================================================
            # STEP 2: Constant Folding + quant_pre_process (matching triton.py)
            # =================================================================
            model_to_quantize = onnx_path
            
            # Step 2a: onnx_graphsurgeon constant folding (matching triton.py lines 722-737)
            self.logger.info("STEP 2a: Constant folding (onnx_graphsurgeon)")
            try:
                from onnx.external_data_helper import convert_model_to_external_data
                
                tmp_folded = os.path.join(tmpdir, "folded.onnx")
                tmp_folded_data = os.path.join(tmpdir, "folded.onnx_data")
                model_onnx = onnx.load_model(onnx_path)
                graph = gs.import_onnx(model_onnx)
                graph.fold_constants().cleanup()
                model_onnx = gs.export_onnx(graph)
                
                convert_model_to_external_data(
                    model_onnx,
                    all_tensors_to_one_file=True,
                    location=tmp_folded_data,
                    size_threshold=0,
                )
                onnx.save_model(model_onnx, tmp_folded)
                del model_onnx
                model_to_quantize = tmp_folded
                self.logger.info("  Constants folded with onnx_graphsurgeon")
            except Exception as e:
                self.logger.warning(f"  Constant folding failed: {e}, using original ONNX")
                model_to_quantize = onnx_path
            
            # Step 2b: quant_pre_process (matching triton.py lines 757-769)
            self.logger.info("STEP 2b: quant_pre_process (ORT shape inference, matching triton.py)")
            try:
                from onnxruntime.quantization import shape_inference
                
                preprocessed_path = os.path.join(tmpdir, "preprocessed.onnx")
                preprocessed_data = "preprocessed.onnx_data"
                shape_inference.quant_pre_process(
                    input_model_path=model_to_quantize,
                    output_model_path=preprocessed_path,
                    skip_optimization=False,
                    skip_onnx_shape=False,
                    skip_symbolic_shape=True,
                    auto_merge=False,
                    guess_output_rank=False,
                    verbose=0,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    external_data_location=preprocessed_data,
                )
                model_to_quantize = preprocessed_path
                self.logger.info("  quant_pre_process completed")
            except Exception as e:
                self.logger.warning(f"  quant_pre_process failed: {e}, continuing without it")
            
            # =================================================================
            # STEP 3: FP8 Quantization with ModelOpt (with calibration data)
            # =================================================================
            model_for_trt = model_to_quantize
            use_fp8 = True
            
            if self.skip_modelopt:
                self.logger.info("STEP 3: Skipping ModelOpt FP8 quantization (using native TRT FP8)")
            else:
                self.logger.info("STEP 3: FP8 Quantization with ModelOpt (with calibration)")
                
                if not modelopt_imported:
                    raise ImportError("nvidia-modelopt is required for FP8 quantization")
                
                try:
                    import numpy as np
                    import glob
                    
                    quantized_path = os.path.join(tmpdir, "quantized.onnx")
                    
                    if not self.profiles or len(self.profiles) == 0:
                        raise ValueError("No input profiles available")
                    profile = self.profiles[0]
                    
                    # Build calibration_shapes string
                    shape_parts = []
                    for input_name, shapes in profile.items():
                        opt_shape = shapes[1]
                        shape_str = "x".join(str(d) for d in opt_shape)
                        shape_parts.append(f"{input_name}:{shape_str}")
                    calib_shapes_str = ",".join(shape_parts)
                    self.logger.info(f"  Calibration shapes: {calib_shapes_str}")
                    
                    # Collect calibration samples
                    calib_audio_dir = self.build_args.get("calibration_audio_dir", None)
                    calib_data_file = self.build_args.get("calibration_data_file", None)
                    num_calib_samples = self.build_args.get("num_calibration_samples", 100)
                    
                    input_dtypes = self._get_onnx_input_dtypes(model_to_quantize)
                    if input_dtypes:
                        self.logger.info(f"  Detected ONNX input dtypes: {{{', '.join(f'{k}: {v.__name__}' for k, v in input_dtypes.items())}}}")
                    
                    calib_samples = []
                    
                    if calib_audio_dir and os.path.isdir(calib_audio_dir):
                        self.logger.info(f"  Loading calibration data from: {calib_audio_dir}")
                        npy_files = sorted(glob.glob(os.path.join(calib_audio_dir, "*.npy")))
                        wav_files = sorted(glob.glob(os.path.join(calib_audio_dir, "*.wav")))
                        
                        if npy_files:
                            self.logger.info(f"  Found {len(npy_files)} .npy files")
                            for idx, npy_file in enumerate(npy_files[:num_calib_samples]):
                                try:
                                    mel_data = np.load(npy_file)
                                    sample = self._build_calib_sample(profile, mel_data, idx == 0, input_dtypes)
                                    calib_samples.append(sample)
                                except Exception as e:
                                    self.logger.warning(f"    Failed to load {npy_file}: {e}")
                        elif wav_files:
                            self.logger.info(f"  Found {len(wav_files)} .wav files")
                            try:
                                from scipy.io import wavfile
                                from scipy.signal import resample
                                
                                target_sr = 16000
                                n_mels = 80
                                n_fft = 512
                                hop_length = 160
                                mel_fb = self._create_mel_filterbank(target_sr, n_fft, n_mels)
                                
                                for idx, wav_file in enumerate(wav_files[:num_calib_samples]):
                                    try:
                                        sr, audio = wavfile.read(wav_file)
                                        if audio.dtype == np.int16:
                                            audio = audio.astype(np.float32) / 32768.0
                                        elif audio.dtype == np.int32:
                                            audio = audio.astype(np.float32) / 2147483648.0
                                        else:
                                            audio = audio.astype(np.float32)
                                        if audio.ndim > 1:
                                            audio = audio.mean(axis=1)
                                        if sr != target_sr:
                                            num_samples = int(len(audio) * target_sr / sr)
                                            audio = resample(audio, num_samples)
                                        mel_data = self._compute_mel_spectrogram(audio, n_fft, hop_length, mel_fb)
                                        sample = self._build_calib_sample(profile, mel_data, idx == 0, input_dtypes)
                                        calib_samples.append(sample)
                                    except Exception as e:
                                        self.logger.warning(f"    Failed to process {wav_file}: {e}")
                                self.logger.info(f"  Loaded {len(calib_samples)} samples from wav files")
                            except ImportError as e:
                                self.logger.warning(f"  scipy not available: {e}")
                    
                    elif calib_data_file and os.path.isfile(calib_data_file):
                        self.logger.info(f"  Loading from: {calib_data_file}")
                        try:
                            npz_data = np.load(calib_data_file)
                            if "mels" in npz_data:
                                for idx, mel_data in enumerate(npz_data["mels"][:num_calib_samples]):
                                    sample = self._build_calib_sample(profile, mel_data, idx == 0, input_dtypes)
                                    calib_samples.append(sample)
                        except Exception as e:
                            self.logger.warning(f"  Failed to load .npz: {e}")
                    
                    if len(calib_samples) == 0:
                        self.logger.warning("  WARNING: Using SYNTHETIC calibration data!")
                        sample = self._build_synthetic_calib_sample(profile, input_dtypes)
                        calib_samples.append(sample)
                    
                    self.logger.info(f"  Collected {len(calib_samples)} calibration samples")
                    
                    calibration_data = calib_samples[0]
                    self.logger.info(f"  Using first calibration sample:")
                    for input_name, arr in calibration_data.items():
                        self.logger.info(f"    {input_name}: shape = {arr.shape}")
                    
                    # Call ModelOpt quantize
                    self.logger.warning(f"  Quantizing to FP8:")
                    self.logger.warning(f"    op_types_to_exclude: {self.op_types_to_exclude}")
                    self.logger.warning(f"    nodes_to_exclude:    {self.nodes_to_exclude}")
                    
                    self._log_matching_nodes(model_to_quantize)
                    
                    import time
                    quant_start = time.time()
                    self.logger.warning(f"  >>> Starting ModelOpt quantize() at {time.strftime('%H:%M:%S')}")
                    
                    modelopt_quantization.quantize(
                        onnx_path=model_to_quantize,
                        quantize_mode="fp8",
                        output_path=quantized_path,
                        calibration_data=calibration_data,
                        calibration_method="max",
                        calibration_shapes=calib_shapes_str,
                        override_shapes=calib_shapes_str,
                        op_types_to_exclude=self.op_types_to_exclude,
                        nodes_to_exclude=self.nodes_to_exclude,
                        use_external_data_format=True,
                        dq_only=False,
                    )
                    
                    quant_elapsed = time.time() - quant_start
                    self.logger.warning(f"  >>> ModelOpt quantize() completed in {quant_elapsed:.1f}s")
                    
                    # Restore dynamic shapes
                    self.logger.info("  Restoring dynamic shapes...")
                    quantized_path = self._restore_dynamic_shapes(quantized_path, self.profiles)
                    
                    model_for_trt = quantized_path
                    
                    # Optionally save quantized ONNX for debugging
                    save_path = self.build_args.get("save_quantized_onnx", None)
                    if save_path:
                        shutil.copy2(model_for_trt, save_path)
                        self.logger.info(f"  Saved quantized ONNX to: {save_path}")
                    
                except Exception as quant_error:
                    self.logger.error(f"  ModelOpt quantization failed: {quant_error}")
                    import traceback
                    traceback.print_exc()
                    raise quant_error
            
            # =================================================================
            # STEP 4: Build TRT Engine with DIRECT TRT API (matching triton.py)
            # =================================================================
            self.logger.info("STEP 4: Building TensorRT engine (DIRECT TRT API)")
            
            engine_bytes = self._build_trt_engine_direct(
                onnx_path=model_for_trt,
                opt_profiles=self.profiles,
                use_fp8=use_fp8,
            )
            
            # =================================================================
            # STEP 5: Save Engine
            # =================================================================
            if engine_bytes:
                with open(self.plan_path, "wb") as f:
                    if hasattr(engine_bytes, 'tobytes'):
                        f.write(engine_bytes.tobytes())
                    else:
                        f.write(engine_bytes)
                self.logger.info(f"FP8 TRT engine saved to: {self.plan_path}")


# =============================================================================
# Entry Point Functions
# =============================================================================

def trt_fp8_forward(self, *argv, **kwargs):
    """
    Patch function to replace model's forward() with TRT FP8 hook.
    """
    return self._trt_fp8_compiler.forward(self, argv, kwargs)


def trt_fp8_compile(
    model: torch.nn.Module,
    base_path: str,
    args: Optional[Dict[str, Any]] = None,
    submodule: Optional[Union[str, List[str]]] = None,
    logger: Any = None,
) -> torch.nn.Module:
    """
    Compile a PyTorch model (or submodule) to TensorRT with FP8 precision.
    
    Uses DIRECT TRT API builder (matching triton.py) instead of polygraphy.
    This gives TRT full freedom to use auxiliary streams for parallel execution.
    
    Args:
        model: PyTorch model to compile
        base_path: Base path for saving TRT engine(s)
        args: Dictionary of arguments. Key options:
            - input_names: List of input tensor names
            - input_profiles: List of shape profiles [{name: [min, opt, max], ...}]
            - build_args: Dict with calibration settings:
                - calibration_audio_dir: Path to calibration audio files
                - num_calibration_samples: Number of calibration samples (default: 100)
                - save_quantized_onnx: Path to save quantized ONNX for debugging
            - op_types_to_exclude: Op types to exclude from FP8 (default: ["Conv"])
            - nodes_to_exclude: Regex patterns for nodes to keep in FP16
            - skip_modelopt: Skip ModelOpt quantization (default: False)
            - verbose: Enable verbose TRT logging (default: False)
            - fallback: Fall back to PyTorch if TRT fails (default: False)
        submodule: Optional submodule path(s) to compile
        logger: Optional logger for diagnostics
    
    Returns:
        The model with TRT FP8 compilation hooks installed
    """
    default_args: Dict[str, Any] = {
        "build_args": {
            # No builder_optimization_level — use TRT default (3)
            # No precision_constraints — let TRT optimize freely
            # No max_aux_streams — let TRT heuristic decide
        },
        "verbose": False,
    }
    
    default_args.update(args or {})
    args = default_args
    
    # Check FP8 support
    skip_modelopt = args.get("skip_modelopt", False)
    if not check_fp8_support(skip_modelopt=skip_modelopt):
        logger = logger or getLogger("trt_fp8_direct")
        logger.warning(
            "FP8 is not supported on this system. Model will run in PyTorch mode. "
            "FP8 requires: Ada Lovelace GPU (compute >= 8.9), TensorRT 8.6+"
        )
        return model
    
    # Handle timestamp from existing file
    if os.path.exists(base_path):
        timestamp = int(os.path.getmtime(base_path))
        if "timestamp" in args:
            timestamp = max(int(args["timestamp"]), timestamp)
        args["timestamp"] = timestamp

    def wrap(model, path):
        if not hasattr(model, "_trt_fp8_compiler"):
            model.orig_forward = model.forward
            wrapper = TrtFP8DirectCompiler(model, path + ".plan", logger=logger, **args)
            model._trt_fp8_compiler = wrapper
            model.forward = MethodType(trt_fp8_forward, model)

    def find_sub(parent, submodule):
        idx = submodule.find(".")
        if idx != -1:
            parent_name = submodule[:idx]
            parent = getattr(parent, parent_name)
            submodule = submodule[idx + 1:]
            return find_sub(parent, submodule)
        return parent, submodule

    if submodule is not None:
        if isinstance(submodule, str):
            submodule = [submodule]
        for s in submodule:
            parent, sub = find_sub(model, s)
            wrap(getattr(parent, sub), base_path + "." + s)
    else:
        wrap(model, base_path)

    return model
