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
TensorRT FP8 Compiler for NeMo models.

This module provides FP8 quantization and TensorRT compilation for neural network models.
Based on triton.py's proven FP8 flow that works for CTC models.

FP8 requires:
- Ada Lovelace GPU or newer (compute capability >= 8.9)
- TensorRT 8.6+
- nvidia-modelopt package for FP8 quantization (optional with skip_modelopt=True)

Usage:
    from nemo.export.tensorrt_fp8_compiler import trt_fp8_compile
    
    model = trt_fp8_compile(
        model.encoder,
        base_path="/path/to/encoder",
        args={
            "input_names": ["audio_signal", "length"],
            "input_profiles": [...],
        }
    )
"""

from __future__ import annotations

import inspect
import json
import os
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
logger = getLogger("trt_fp8_compile")


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
# Engine Configuration (from triton.py)
# =============================================================================

@dataclass(unsafe_hash=True)
class EngineConfiguration:
    """Configuration for TRT engine building."""
    max_batch: int
    fp16: bool
    bf16: bool
    fp8: bool
    config_flags: int
    max_workspace: int
    opt_profiles: str
    explicit_batch: int


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
        self.logger = logger or getLogger("trt_fp8_compile")
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

    def infer(self, stream, use_cuda_graph=False):
        """
        Runs TRT engine.
        """
        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                import cuda.cudart as cudart
                cudart.cudaGraphLaunch(self.cuda_graph_instance, stream)
                torch.cuda.synchronize()
            else:
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                import cuda.cudart as cudart
                cudart.cudaStreamBeginCapture(
                    stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
                )
                self.context.execute_async_v3(stream)
                graph = cudart.cudaStreamEndCapture(stream)[1]
                self.cuda_graph_instance = cudart.cudaGraphInstantiate(graph, 0)[1]
                self.logger.info("CUDA Graph captured!")
        else:
            noerror = self.context.execute_async_v3(stream)
            torch.cuda.synchronize()
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors


# =============================================================================
# TrtFP8Compiler Class - Based on triton.py's approach
# =============================================================================

class TrtFP8Compiler:
    """
    TensorRT FP8 Compiler class.
    
    Based on triton.py's proven FP8 flow:
    - Uses onnx_graphsurgeon for constant folding
    - Uses quant_pre_process() before ModelOpt
    - Uses direct TRT API for engine building
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
        fp8_exclusion_preset: Optional[str] = None,
        skip_modelopt: bool = False,
        verbose: bool = False,
        logger=None,
    ):
        """
        Initialize the FP8 TRT Compiler.
        
        Args:
            model: PyTorch model to compile
            plan_path: Path to save the TRT engine (.plan file)
            input_names: List of input tensor names
            output_names: List of output tensor names
            output_lists: Output grouping specification
            export_args: Arguments for torch.onnx.export()
            build_args: Arguments for TRT builder
            input_profiles: List of input shape profiles for dynamic shapes
            dynamic_batchsize: [min, opt, max] batch sizes
            use_cuda_graph: Enable CUDA graph for inference
            timestamp: Timestamp for cache invalidation
            fallback: Fall back to PyTorch if TRT fails
            op_types_to_exclude: Op types to exclude from FP8 quantization
            nodes_to_exclude: ONNX node name regex patterns to exclude from FP8.
                Pass [] to quantize all nodes. Pass None for defaults.
                Ignored if fp8_exclusion_preset is set.
            fp8_exclusion_preset: Named preset for nodes_to_exclude. Options:
                - "streaming": batch=1 low-latency — only ff*/linear2 stays FP8.
                - "throughput": batch>=8 — only small attention MatMuls excluded.
                - None: Use nodes_to_exclude directly (default).
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
        
        _FP8_EXCLUSION_PRESETS = {
            "streaming": [
                r"self_attn/linear_pos",
                r"self_attn/linear_out",
                r"self_attn/MatMul_1",
                r"self_attn/MatMul_2",
                r"self_attn/MatMul(?!_)",
                r"self_attn/linear_q",
                r"self_attn/linear_k",
                r"self_attn/linear_v",
                r"feed_forward[12]/linear1",
            ],
            "throughput": [
                r"self_attn/linear_pos",
                r"self_attn/MatMul_1",
                r"self_attn/MatMul_2",
                r"self_attn/MatMul(?!_)",
            ],
        }
        
        if fp8_exclusion_preset is not None:
            preset_key = fp8_exclusion_preset.lower()
            if preset_key not in _FP8_EXCLUSION_PRESETS:
                raise ValueError(
                    f"Unknown fp8_exclusion_preset: '{fp8_exclusion_preset}'. "
                    f"Available presets: {list(_FP8_EXCLUSION_PRESETS.keys())}"
                )
            self.nodes_to_exclude = _FP8_EXCLUSION_PRESETS[preset_key]
        elif nodes_to_exclude is not None:
            self.nodes_to_exclude = nodes_to_exclude
        else:
            self.nodes_to_exclude = []
        self.skip_modelopt = skip_modelopt
        self.verbose = verbose
        
        self.logger = logger or getLogger("trt_fp8_compile")
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
            self.logger.info(f"FP8 Engine loaded, inputs: {self.engine.input_table}")
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
                    self.engine.set_inputs(unroll_input(self.input_names, args), stream.cuda_stream)
                    self.engine.allocate_buffers(device=device)
                    stream.wait_stream(torch.cuda.current_stream())
                    ret = self.engine.infer(stream.cuda_stream, use_cuda_graph=self.use_cuda_graph)
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

    def _prepare_mel_for_calibration(self, mel_data, target_shape: tuple):
        """
        Prepare mel spectrogram for calibration by padding/cropping to target shape.
        
        Args:
            mel_data: Input mel spectrogram, shape can be [n_mels, time] or [batch, n_mels, time]
            target_shape: Target shape from optimization profile, e.g., (batch, n_mels, time)
        
        Returns:
            Mel spectrogram reshaped to target_shape
        """
        import numpy as np
        
        # Ensure mel_data is at least 2D
        if mel_data.ndim == 1:
            mel_data = mel_data.reshape(1, -1)
        
        # Add batch dimension if needed
        if mel_data.ndim == 2:
            mel_data = mel_data[np.newaxis, ...]  # [1, n_mels, time]
        
        # Now mel_data is [batch, n_mels, time]
        # target_shape is typically (batch, n_mels, time) 
        
        result = np.zeros(target_shape, dtype=np.float32)
        
        # Copy data, handling dimension mismatches
        src_shape = mel_data.shape
        
        # Determine how many dimensions to copy
        ndim = min(len(src_shape), len(target_shape))
        
        # Build slices for copying
        slices_src = []
        slices_dst = []
        for i in range(ndim):
            size = min(src_shape[i], target_shape[i])
            slices_src.append(slice(0, size))
            slices_dst.append(slice(0, size))
        
        # Pad remaining dimensions if target has more dims
        for i in range(ndim, len(target_shape)):
            slices_dst.append(slice(0, 1))
        
        try:
            result[tuple(slices_dst)] = mel_data[tuple(slices_src)]
        except Exception:
            # Fallback: just fill with the mean of the mel data
            result.fill(mel_data.mean())
        
        return result

    def _get_onnx_input_dtypes(self, onnx_path: str) -> dict:
        """
        Get input dtypes from an ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
            
        Returns:
            Dict mapping input names to numpy dtypes
        """
        import numpy as np
        
        dtype_map = {}
        try:
            model = onnx.load(onnx_path, load_external_data=False)
            for inp in model.graph.input:
                name = inp.name
                if inp.type.HasField('tensor_type'):
                    elem_type = inp.type.tensor_type.elem_type
                    onnx_to_numpy = {
                        1: np.float32,   # FLOAT
                        2: np.uint8,     # UINT8
                        3: np.int8,      # INT8
                        4: np.uint16,    # UINT16
                        5: np.int16,     # INT16
                        6: np.int32,     # INT32
                        7: np.int64,     # INT64
                        9: np.bool_,     # BOOL
                        10: np.float16,  # FLOAT16
                        11: np.float64,  # DOUBLE
                        12: np.uint32,   # UINT32
                        13: np.uint64,   # UINT64
                    }
                    dtype_map[name] = onnx_to_numpy.get(elem_type, np.float32)
        except Exception as e:
            self.logger.warning(f"  Failed to get ONNX input dtypes: {e}")
        
        return dtype_map

    def _build_calib_sample(self, profile: dict, mel_data, log_first: bool = False, input_dtypes: dict = None):
        """
        Build a single calibration sample dict from mel spectrogram data.
        
        Args:
            profile: Input profile dict {input_name: [min_shape, opt_shape, max_shape]}
            mel_data: Mel spectrogram numpy array
            log_first: If True, log the first sample's shapes
            input_dtypes: Optional dict of {input_name: numpy_dtype} from ONNX model
            
        Returns:
            Dict mapping input names to numpy arrays
        """
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
        """
        Build a synthetic calibration sample (fallback when no real data).
        
        Args:
            profile: Input profile dict {input_name: [min_shape, opt_shape, max_shape]}
            input_dtypes: Optional dict of {input_name: numpy_dtype} from ONNX model
            
        Returns:
            Dict mapping input names to numpy arrays with realistic synthetic values
        """
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
                data = np.clip(data, -15.0, 5.0)
                data = data.astype(dtype)
            elif "cache" in input_lower:
                data = np.zeros(opt_shape, dtype=dtype)
            else:
                data = np.random.normal(loc=0.0, scale=1.0, size=opt_shape).astype(np.float32)
                data = data.astype(dtype)
            
            sample[input_name] = data
        
        return sample

    def _create_mel_filterbank(self, sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float = None):
        """Create mel filterbank matrix using numpy (no librosa/torchaudio needed)."""
        import numpy as np
        
        if fmax is None:
            fmax = sr / 2.0
        
        # Mel scale conversion functions
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        
        # Create mel points
        mel_low = hz_to_mel(fmin)
        mel_high = hz_to_mel(fmax)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # FFT bin frequencies
        n_freqs = n_fft // 2 + 1
        fft_freqs = np.linspace(0, sr / 2.0, n_freqs)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, n_freqs))
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            # Rising slope
            for j, freq in enumerate(fft_freqs):
                if left <= freq < center:
                    filterbank[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filterbank[i, j] = (right - freq) / (right - center)
        
        return filterbank.astype(np.float32)

    def _compute_mel_spectrogram(self, audio, n_fft: int, hop_length: int, mel_fb):
        """Compute log mel spectrogram using numpy (no librosa/torchaudio needed)."""
        import numpy as np
        
        # Pad audio to ensure we get at least one frame
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        
        # Compute STFT using numpy
        # Number of frames
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        
        # Create window
        window = np.hanning(n_fft).astype(np.float32)
        
        # Compute STFT
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            stft[:, i] = np.abs(spectrum).astype(np.float32)
        
        # Apply mel filterbank
        mel_spec = np.dot(mel_fb, stft ** 2)
        
        # Log mel spectrogram
        log_mel = np.log(mel_spec + 1e-9)
        
        return log_mel

    def _restore_dynamic_shapes(self, onnx_path: str, profiles: List[Dict]) -> str:
        """
        Restore dynamic shapes in an ONNX model after ModelOpt quantization.
        
        ModelOpt's override_shapes bakes static dimensions into the output ONNX model.
        This function restores dynamic shapes so TensorRT can build with optimization profiles.
        
        Args:
            onnx_path: Path to the quantized ONNX model with static shapes
            profiles: List of profile dicts defining {input_name: [min, opt, max]}
        
        Returns:
            Path to the ONNX model with dynamic shapes restored (same path, modified in-place)
        """
        self.logger.info("  Restoring dynamic shapes in quantized ONNX model...")
        
        # Load the quantized model
        model = onnx.load(onnx_path)
        
        # Get the first profile to understand which dimensions should be dynamic
        profile = profiles[0] if profiles else {}
        
        # Build a map of which dimensions are dynamic for each input
        # A dimension is dynamic if min != max in the profile
        dynamic_dims = {}  # {input_name: {dim_idx: dim_name}}
        for input_name, shapes in profile.items():
            min_shape, opt_shape, max_shape = shapes
            dynamic_dims[input_name] = {}
            for dim_idx, (min_d, max_d) in enumerate(zip(min_shape, max_shape)):
                if min_d != max_d:
                    # This dimension varies - make it dynamic
                    dynamic_dims[input_name][dim_idx] = f"{input_name}_dynamic_axes_{dim_idx}"
        
        self.logger.info(f"  Dynamic dimensions to restore: {dynamic_dims}")
        
        # Process each input in the model
        inputs_modified = 0
        for input_tensor in model.graph.input:
            input_name = input_tensor.name
            
            if input_name not in dynamic_dims:
                continue
            
            tensor_type = input_tensor.type.tensor_type
            if not tensor_type.HasField('shape'):
                continue
            
            # Restore dynamic dimensions
            for dim_idx, dim_name in dynamic_dims[input_name].items():
                if dim_idx < len(tensor_type.shape.dim):
                    dim = tensor_type.shape.dim[dim_idx]
                    old_value = dim.dim_value if dim.HasField('dim_value') else "already_dynamic"
                    
                    # Clear the fixed value and set symbolic name
                    dim.ClearField('dim_value')
                    dim.dim_param = dim_name
                    
                    self.logger.info(f"    {input_name}[{dim_idx}]: {old_value} -> {dim_name} (dynamic)")
                    inputs_modified += 1
        
        if inputs_modified > 0:
            # Save the modified model back
            # Handle external data format (large models)
            try:
                import os
                external_data_path = os.path.splitext(onnx_path)[0] + "_data"
                onnx.save(
                    model, 
                    onnx_path, 
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

    def _build_trt_engine_from_onnx(
        self,
        onnx_path: str,
        opt_profiles: List[Dict],
        use_fp8: bool = True,
    ) -> bytes:
        """
        Build TRT engine from ONNX file using POLYGRAPHY (matching tensorrt_lazy_compiler.py).
        
        This uses the same proven approach as the working lazy compiler.
        """
        # Import polygraphy components (same as tensorrt_lazy_compiler.py)
        try:
            from polygraphy.backend.trt import CreateConfig, Profile, engine_bytes_from_network, network_from_onnx_path
        except ImportError:
            raise ImportError("Polygraphy is required for TRT engine building. Install with: pip install polygraphy")
        
        # Check for precision override modes
        force_fp32_pure = self.build_args.get("force_fp32_pure", False)
        force_fp16_pure = self.build_args.get("force_fp16_pure", False)
        force_fp32 = self.build_args.get("force_fp32", False)
        force_fp16 = self.build_args.get("force_fp16", False)
        
        if force_fp32_pure or force_fp32:
            self.logger.warning("Building PURE FP32 engine (no FP16/FP8/TF32)")
            use_fp8 = False
        elif force_fp16_pure or force_fp16:
            self.logger.warning("Building FP16+TF32 engine (FP8 disabled)")
            use_fp8 = False
        
        # Build profiles for polygraphy (same format as tensorrt_lazy_compiler.py)
        profiles = []
        for profile_spec in opt_profiles:
            p = Profile()
            for input_name, shapes in profile_spec.items():
                min_shape, opt_shape, max_shape = shapes
                p.add(input_name, min=min_shape, opt=opt_shape, max=max_shape)
            profiles.append(p)
        
        # Build args for polygraphy CreateConfig (matching tensorrt_lazy_compiler.py)
        # Remove custom args that polygraphy doesn't understand
        build_args = self.build_args.copy()
        
        # Remove ALL custom args not recognized by polygraphy CreateConfig
        custom_args_to_remove = [
            # Precision override flags
            "force_fp32_pure",
            "force_fp16_pure", 
            "force_fp32",
            "force_fp16",
            # Calibration args (used by ModelOpt, not TRT)
            "num_calibration_samples",
            "calibration_audio_dir",
            "calibration_data_file",
            # Other custom args
            "use_synthetic_calibration",
            "max_workspace_size",  # Polygraphy uses different param name
            "skip_modelopt",
        ]
        for arg in custom_args_to_remove:
            build_args.pop(arg, None)
        
        # Set precision flags based on mode
        enable_fp8 = False
        if force_fp32_pure or force_fp32:
            # PURE FP32: No precision flags at all
            build_args["tf32"] = False
            build_args["fp16"] = False
            self.logger.info(f"Building TensorRT PURE FP32 engine using polygraphy")
        elif force_fp16_pure or force_fp16:
            build_args["tf32"] = True  # Enable TF32 for numerical stability
            build_args["fp16"] = True  # Enable FP16
            self.logger.info(f"Building TensorRT FP16+TF32 engine using polygraphy")
        else:
            build_args["tf32"] = True  # Enable TF32 for numerical stability
            build_args["fp16"] = True  # Always enable FP16
            if use_fp8:
                enable_fp8 = True
                self.logger.info(f"Building TensorRT FP8+FP16 engine using polygraphy")
            else:
                self.logger.info(f"Building TensorRT FP16 engine using polygraphy")
        
        self.logger.info(f"  ONNX path: {onnx_path}")
        self.logger.info(f"  Profiles: {len(profiles)}")
        self.logger.info(f"  Build args: {build_args}")
        self.logger.info(f"  FP8 enabled: {enable_fp8}")
        
        # Use polygraphy to load network with NATIVE_INSTANCENORM flag (CRITICAL for Conformer)
        self.logger.info("  Loading ONNX with NATIVE_INSTANCENORM flag (polygraphy)")
        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        
        # Build engine using polygraphy
        # NOTE: For FP8, we need to set the builder flag explicitly since polygraphy
        # CreateConfig may not support fp8=True directly
        self.logger.info("  Building engine with polygraphy CreateConfig")
        
        if enable_fp8:
            # FP8 requires setting the builder flag explicitly
            # Also need precision_constraints='prefer' to ensure FP8 precision is honored
            from polygraphy.backend.trt import TrtRunner
            
            # Set precision constraints so TRT honors the QDQ node precisions
            # 'prefer' = use requested precision when possible
            # 'obey' = strictly enforce requested precision (may fail if not possible)
            build_args["precision_constraints"] = "obey"
            self.logger.info("  Setting precision_constraints='prefer' for FP8 QDQ model")
            
            # Try setting fp8 through CreateConfig first (newer polygraphy versions support this)
            try:
                build_args["fp8"] = True
                self.logger.info("  Trying fp8=True in CreateConfig with precision_constraints='prefer'")
                engine_bytes = engine_bytes_from_network(network, config=CreateConfig(profiles=profiles, **build_args))
            except TypeError as e:
                # If fp8 not supported as kwarg, build without it and rely on TRT auto-selection
                self.logger.warning(f"  fp8 kwarg not supported by CreateConfig: {e}")
                self.logger.warning("  Building with FP16+TF32+precision_constraints, TRT will use FP8 from QDQ nodes")
                build_args.pop("fp8", None)
                engine_bytes = engine_bytes_from_network(network, config=CreateConfig(profiles=profiles, **build_args))
        else:
            engine_bytes = engine_bytes_from_network(network, config=CreateConfig(profiles=profiles, **build_args))
        
        if engine_bytes is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        engine_size = len(engine_bytes) if hasattr(engine_bytes, '__len__') else 0
        self.logger.info(f"TensorRT engine built successfully. Size: {engine_size / (1024*1024):.2f} MB")
        
        return engine_bytes

    def _build_and_save(self, model, input_example):
        """
        Export model to ONNX, optionally quantize to FP8, and build TRT engine.
        
        Based on triton.py's build_trt_engine_from_onnx_static() flow.
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
                            profile[f"{input_id}_{i}"] = [
                                [dbs[0], *sh],
                                [dbs[1], *sh],
                                [dbs[2], *sh]
                            ]
                elif isinstance(val, torch.Tensor):
                    sh = val.shape
                    if len(sh) > 0:
                        sh = sh[1:]
                        profile[input_id] = [
                            [dbs[0], *sh],
                            [dbs[1], *sh],
                            [dbs[2], *sh]
                        ]
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
            self.logger.info(f"  Export args: {export_args}")
            
            # Let PyTorch choose the default opset for maximum compatibility
            # ModelOpt FP8 works with opset 17+ (QDQ nodes are supported)
            opset_version = None  # PyTorch will use its default (typically 17-19)
            self.logger.info(f"  Using default opset_version (PyTorch will select compatible version)")
            
            torch.onnx.export(
                model,
                (input_example,),
                onnx_path,
                input_names=input_names,
                output_names=self.output_names if self.output_names else None,
                opset_version=opset_version,
                **export_args,
            )
            self.logger.info("ONNX export successful")
            
            # =================================================================
            # Check for PURE modes - skip ALL processing
            # =================================================================
            force_fp32_pure = self.build_args.get("force_fp32_pure", False)
            force_fp16_pure = self.build_args.get("force_fp16_pure", False)
            force_fp32 = self.build_args.get("force_fp32", False)
            force_fp16 = self.build_args.get("force_fp16", False)
            
            if force_fp32_pure:
                # PURE FP32: Skip EVERYTHING - just ONNX export → TRT build
                self.logger.warning("="*60)
                self.logger.warning("force_fp32_pure=True: PURE FP32 TEST MODE")
                self.logger.warning("  - NO constant folding")
                self.logger.warning("  - NO quantization")
                self.logger.warning("  - NO FP16/TF32 flags")
                self.logger.warning("  - Just raw ONNX → TRT FP32")
                self.logger.warning("="*60)
                self.logger.info("STEP 2: SKIPPED (force_fp32_pure)")
                self.logger.info("STEP 3: SKIPPED (force_fp32_pure)")
                model_for_trt = onnx_path
                use_fp8 = False
            elif force_fp16_pure:
                # PURE FP16: Skip constant folding and quantization, just FP16+TF32
                self.logger.warning("="*60)
                self.logger.warning("force_fp16_pure=True: PURE FP16 TEST MODE")
                self.logger.warning("  - NO constant folding")
                self.logger.warning("  - NO quantization")
                self.logger.warning("  - FP16+TF32 enabled")
                self.logger.warning("  - Just raw ONNX → TRT FP16")
                self.logger.warning("="*60)
                self.logger.info("STEP 2: SKIPPED (force_fp16_pure)")
                self.logger.info("STEP 3: SKIPPED (force_fp16_pure)")
                model_for_trt = onnx_path
                use_fp8 = False
            else:
                # =================================================================
                # STEP 2: Constant Folding (EXACTLY matching tensorrt_lazy_compiler.py)
                # =================================================================
                model_to_quantize = onnx_path  # Default: use original if folding fails
                
                self.logger.info("STEP 2: Constant folding (polygraphy)")
                try:
                    from polygraphy.backend.onnx.loader import fold_constants, onnx_from_path, save_onnx
                    
                    # EXACTLY match tensorrt_lazy_compiler.py lines 626-627:
                    # onnx_model = fold_constants(onnx_from_path(onnx_path), size_threshold=16 * 1000 * 1000)
                    # save_onnx(onnx_model, onnx_path)
                    onnx_model = fold_constants(onnx_from_path(onnx_path), size_threshold=16 * 1000 * 1000)
                    save_onnx(onnx_model, onnx_path)  # Save back to SAME path (like lazy compiler)
                    model_to_quantize = onnx_path
                    self.logger.info("  Constants folded with polygraphy (saved to same path)")
                    
                except Exception as e:
                    self.logger.warning(f"  Constant folding failed: {e}, using original ONNX model")
                    model_to_quantize = onnx_path
                
                # =================================================================
                # STEP 3: FP8 Quantization (from triton.py)
                # =================================================================
                model_for_trt = model_to_quantize
                use_fp8 = True
                
                # Check for precision override modes - skip FP8 quantization
                if force_fp32:
                    self.logger.info("STEP 3: SKIPPED - force_fp32=True, building PURE FP32 engine")
                    use_fp8 = False
                elif force_fp16:
                    self.logger.info("STEP 3: SKIPPED - force_fp16=True, building pure FP16 engine")
                    use_fp8 = False
                elif self.skip_modelopt:
                    self.logger.info("STEP 3: Skipping ModelOpt FP8 quantization (using native TRT FP8)")
                    self.logger.warning("Native TRT FP8 without calibration may have lower accuracy.")
                else:
                    self.logger.info("STEP 3: FP8 Quantization with ModelOpt")
                    
                    # =========================================================
                    # ModelOpt FP8 quantization (simplified per official docs)
                    # No preprocessing needed - ModelOpt handles ONNX directly
                    # =========================================================
                    quantization_succeeded = False
                    
                    if not modelopt_imported:
                        self.logger.error("  ModelOpt not available - cannot do FP8 quantization!")
                        raise ImportError("nvidia-modelopt is required for FP8 quantization")
                    
                    try:
                        import numpy as np
                        import glob
                        
                        quantized_path = os.path.join(tmpdir, "quantized.onnx")
                        
                        # Get profile for shapes
                        if not self.profiles or len(self.profiles) == 0:
                            raise ValueError("No input profiles available - required for FP8 calibration")
                        profile = self.profiles[0]
                        
                        # Build calibration_shapes string (format: "name1:d1xd2xd3,name2:d1xd2")
                        shape_parts = []
                        for input_name, shapes in profile.items():
                            opt_shape = shapes[1]  # Use optimal shape
                            shape_str = "x".join(str(d) for d in opt_shape)
                            shape_parts.append(f"{input_name}:{shape_str}")
                        calib_shapes_str = ",".join(shape_parts)
                        self.logger.info(f"  Calibration shapes: {calib_shapes_str}")
                        
                        # =========================================================
                        # Collect calibration samples (list of dicts)
                        # =========================================================
                        calib_audio_dir = self.build_args.get("calibration_audio_dir", None)
                        calib_data_file = self.build_args.get("calibration_data_file", None)
                        num_calib_samples = self.build_args.get("num_calibration_samples", 100)
                        
                        input_dtypes = self._get_onnx_input_dtypes(model_to_quantize)
                        if input_dtypes:
                            self.logger.info(f"  Detected ONNX input dtypes: {{{', '.join(f'{k}: {v.__name__}' for k, v in input_dtypes.items())}}}")
                        
                        calib_samples = []  # List of {input_name: array} dicts
                        
                        # Option 1: Load from directory (.npy or .wav files)
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
                                self.logger.info(f"  Computing mel spectrograms using scipy (no torchaudio needed)")
                                
                                try:
                                    from scipy.io import wavfile
                                    from scipy.signal import resample
                                    
                                    # Mel spectrogram parameters (standard ASR)
                                    target_sr = 16000
                                    n_mels = 80
                                    n_fft = 512
                                    hop_length = 160
                                    
                                    # Pre-compute mel filterbank
                                    mel_fb = self._create_mel_filterbank(target_sr, n_fft, n_mels)
                                    
                                    for idx, wav_file in enumerate(wav_files[:num_calib_samples]):
                                        try:
                                            sr, audio = wavfile.read(wav_file)
                                            
                                            # Convert to float32 and normalize
                                            if audio.dtype == np.int16:
                                                audio = audio.astype(np.float32) / 32768.0
                                            elif audio.dtype == np.int32:
                                                audio = audio.astype(np.float32) / 2147483648.0
                                            elif audio.dtype == np.uint8:
                                                audio = (audio.astype(np.float32) - 128) / 128.0
                                            else:
                                                audio = audio.astype(np.float32)
                                            
                                            # Convert stereo to mono
                                            if audio.ndim > 1:
                                                audio = audio.mean(axis=1)
                                            
                                            # Resample if needed
                                            if sr != target_sr:
                                                num_samples = int(len(audio) * target_sr / sr)
                                                audio = resample(audio, num_samples)
                                            
                                            # Compute mel spectrogram
                                            mel_data = self._compute_mel_spectrogram(audio, n_fft, hop_length, mel_fb)
                                            
                                            sample = self._build_calib_sample(profile, mel_data, idx == 0, input_dtypes)
                                            calib_samples.append(sample)
                                            
                                            if idx == 0:
                                                self.logger.info(f"    First mel shape: {mel_data.shape}")
                                        except Exception as e:
                                            self.logger.warning(f"    Failed to process {wav_file}: {e}")
                                    
                                    self.logger.info(f"  Loaded {len(calib_samples)} calibration samples from wav files")
                                except ImportError as e:
                                    self.logger.warning(f"  scipy not available: {e}")
                        
                        # Option 2: Load from .npz file
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
                        
                        # Fallback: synthetic data
                        if len(calib_samples) == 0:
                            self.logger.warning("  WARNING: Using SYNTHETIC calibration data!")
                            self.logger.warning("  For better accuracy, provide calibration_audio_dir or calibration_data_file")
                            for _ in range(min(num_calib_samples, 32)):
                                sample = self._build_synthetic_calib_sample(profile, input_dtypes)
                                calib_samples.append(sample)
                        
                        self.logger.info(f"  Collected {len(calib_samples)} calibration samples")
                        
                        # =========================================================
                        # Use first sample for calibration
                        # Note: ModelOpt expects single sample dict, not stacked arrays
                        # Stacking would create wrong ranks for inputs with batch dim != 0
                        # (e.g., cache_last_time has shape [24, batch, 1024, 8])
                        # =========================================================
                        calibration_data = calib_samples[0]
                        self.logger.info(f"  Using first calibration sample:")
                        for input_name, arr in calibration_data.items():
                            self.logger.info(f"    {input_name}: shape={arr.shape}, dtype={arr.dtype}")
                        
                        # =========================================================
                        # Call ModelOpt quantize
                        # =========================================================
                        self.logger.warning(f"  Quantizing to FP8:")
                        self.logger.warning(f"    op_types_to_exclude: {self.op_types_to_exclude}")
                        self.logger.warning(f"    nodes_to_exclude:    {self.nodes_to_exclude}")
                        self.logger.warning(f"  Input model: {model_to_quantize}")
                        
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
                        self.logger.warning(f"  >>> ModelOpt quantize() completed in {quant_elapsed:.1f}s at {time.strftime('%H:%M:%S')}")
                        self.logger.info(f"  FP8 quantized model saved: {quantized_path}")
                        
                        # =========================================================
                        # CRITICAL: Restore dynamic shapes after ModelOpt quantization
                        # ModelOpt's override_shapes bakes static dims into the ONNX
                        # which causes TRT profile mismatch errors.
                        # =========================================================
                        self.logger.info("  Restoring dynamic shapes for TensorRT compatibility...")
                        quantized_path = self._restore_dynamic_shapes(quantized_path, self.profiles)
                        
                        model_for_trt = quantized_path
                        quantization_succeeded = True
                        
                    except Exception as quant_error:
                        self.logger.error(f"  ModelOpt quantization failed: {quant_error}")
                        import traceback
                        traceback.print_exc()
                        raise quant_error
                    
                    # Fall back to native TRT FP8 if quantization failed
                    # COMMENTED OUT FOR TESTING - let it fail instead
                    # if not quantization_succeeded:
                    #     self.logger.warning("  Falling back to native TRT FP8 (uncalibrated)")
                    #     model_for_trt = model_to_quantize
                    if not quantization_succeeded:
                        raise RuntimeError("ModelOpt FP8 quantization failed - no fallback enabled for testing")
            
            # =================================================================
            # STEP 4: Build TRT Engine (DIRECT TRT API from triton.py)
            # =================================================================
            self.logger.info("STEP 4: Building TensorRT engine")
            
            engine_bytes = self._build_trt_engine_from_onnx(
                onnx_path=model_for_trt,
                opt_profiles=self.profiles,
                use_fp8=use_fp8,
            )
            
            # =================================================================
            # STEP 5: Save Engine
            # =================================================================
            if engine_bytes:
                with open(self.plan_path, "wb") as f:
                    # Handle both IHostMemory (TRT) and bytes objects
                    if hasattr(engine_bytes, 'tobytes'):
                        f.write(engine_bytes.tobytes())
                    else:
                        f.write(engine_bytes)
                self.logger.info(f"STEP 5: FP8 TRT engine saved to: {self.plan_path}")


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
    
    Based on triton.py's proven FP8 flow that works for CTC models.
    
    Args:
        model: PyTorch model to compile
        base_path: Base path for saving TRT engine(s)
        args: Dictionary of arguments passed to TrtFP8Compiler. Key options:
            - input_names: List of input tensor names
            - input_profiles: List of shape profiles [{name: [min, opt, max], ...}]
            - op_types_to_exclude: Op types to exclude from FP8 (default: ["Conv"])
            - skip_modelopt: Skip ModelOpt quantization, use native TRT FP8 (default: False)
            - verbose: Enable verbose TRT logging (default: False)
            - fallback: Fall back to PyTorch if TRT fails (default: False)
        submodule: Optional submodule path(s) to compile instead of whole model
        logger: Optional logger for diagnostics
    
    Returns:
        The model with TRT FP8 compilation hooks installed
    
    Example:
        model = trt_fp8_compile(
            model.encoder,
            "/path/to/encoder",
            args={
                "input_names": ["audio_signal", "length"],
                "input_profiles": [{
                    "audio_signal": [[1, 80, 100], [8, 80, 500], [16, 80, 1000]],
                    "length": [[1], [8], [16]],
                }],
            }
        )
    """
    default_args: Dict[str, Any] = {
        "build_args": {
            "builder_optimization_level": 5,
            "precision_constraints": "obey",  # Match tensorrt_lazy_compiler.py
        },
        "verbose": False,
    }
    
    default_args.update(args or {})
    args = default_args
    
    # Check FP8 support
    skip_modelopt = args.get("skip_modelopt", False)
    if not check_fp8_support(skip_modelopt=skip_modelopt):
        logger = logger or getLogger("trt_fp8_compile")
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
            wrapper = TrtFP8Compiler(model, path + ".plan", logger=logger, **args)
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