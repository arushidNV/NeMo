#!/usr/bin/env python3
"""
ONNX Graph Surgery: Replace [linear1/MatMul + SiluMulCast + QuantizeLinear + DequantizeLinear]
with a FusedGemmSiluQuant plugin that outputs FP16 directly.

The plugin absorbs:
    DequantizeLinear(act) -> MatMul -> Add(bias) -> Silu -> QuantizeLinear -> DequantizeLinear
and replaces it with:
    FusedGemmSiluQuant(activation_fp8, weight_fp8, bias) -> output_fp16

This works around a known TensorRT ONNX-parser limitation where custom plugin
FP8 output types are not propagated to downstream nodes, causing DequantizeLinear
type-inference failures ("No matching rules for input operand types").

Usage:
    python onnx_surgery.py input_model.onnx output_model.onnx

The output model should be compiled with:
    trtexec --plugins=libfused_gemm_silu_quant.so --onnx=output_model.onnx ...
"""

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np

try:
    import onnx
    import onnx_graphsurgeon as gs
except ImportError:
    print("ERROR: requires onnx and onnx-graphsurgeon packages")
    print("  pip install onnx onnx-graphsurgeon")
    sys.exit(1)


# Match feed_forward1 or feed_forward2 + linear1 with either / or . (PyTorch uses ., TRT may use _)
# Examples: feed_forward1/linear1/MatMul, layers.0.feed_forward1.linear1.MatMul, /layers_0/feed_forward1/linear1/MatMul
LINEAR1_MATMUL_PATTERN = re.compile(
    r'feed_forward[12][/_.]linear1', re.IGNORECASE
)


def find_matmul_silu_pairs(graph: gs.Graph) -> List[Tuple[gs.Node, gs.Node, gs.Node, gs.Node]]:
    """
    Find [DequantizeLinear -> MatMul -> Add(bias) -> Silu(Sigmoid+Mul) -> QuantizeLinear]
    patterns for feed_forward*/linear1 layers.
    
    Returns list of (matmul_node, bias_add_node, silu_mul_node, quant_node) tuples.
    """
    patterns = []
    
    for node in graph.nodes:
        # Look for MatMul nodes that are linear1 (feed_forward1 or feed_forward2)
        if node.op != "MatMul":
            continue
        if not LINEAR1_MATMUL_PATTERN.search(node.name):
            continue
        
        # Check: MatMul output -> Add (bias)
        matmul_out = node.outputs[0]
        add_node = None
        for consumer in matmul_out.outputs:
            if consumer.op == "Add":
                add_node = consumer
                break
        if add_node is None:
            continue
        
        # Check: Add output -> Sigmoid (part of Silu)
        add_out = add_node.outputs[0]
        sigmoid_node = None
        for consumer in add_out.outputs:
            if consumer.op == "Sigmoid":
                sigmoid_node = consumer
                break
        if sigmoid_node is None:
            continue
        
        # Check: Sigmoid output + Add output -> Mul (Silu = x * sigmoid(x))
        sigmoid_out = sigmoid_node.outputs[0]
        silu_mul_node = None
        for consumer in sigmoid_out.outputs:
            if consumer.op == "Mul":
                silu_mul_node = consumer
                break
        if silu_mul_node is None:
            continue
        
        # Check: Mul output -> QuantizeLinear (FP8 quantize)
        mul_out = silu_mul_node.outputs[0]
        quant_node = None
        for consumer in mul_out.outputs:
            if consumer.op == "QuantizeLinear":
                quant_node = consumer
                break
        if quant_node is None:
            # Pattern without explicit QuantizeLinear (might be fused differently)
            # Still capture the pattern up to the Silu Mul
            print(f"  WARNING: {node.name} has Silu but no QuantizeLinear after it")
            continue
        
        patterns.append((node, add_node, silu_mul_node, quant_node))
    
    return patterns


def _consumers(tensor: gs.Tensor) -> List[gs.Node]:
    """Return list of nodes that consume this tensor (read-only)."""
    return list(tensor.outputs) if hasattr(tensor, "outputs") and tensor.outputs else []


def analyze_quantized_graph(graph: gs.Graph) -> List[dict]:
    """
    Analyze quantized ONNX before surgery: for each replaceable pattern, report
    DQ nodes and whether they are safe to remove (only consumer is the MatMul we replace).
    Does not modify the graph.
    """
    patterns = find_matmul_silu_pairs(graph)
    report = []
    for matmul_node, add_node, silu_mul_node, quant_node in patterns:
        activation_input = matmul_node.inputs[0]
        weight_input = matmul_node.inputs[1]
        dq_act = activation_input.inputs[0] if (len(activation_input.inputs) == 1 and activation_input.inputs[0].op == "DequantizeLinear") else None
        dq_wt = weight_input.inputs[0] if (len(weight_input.inputs) == 1 and weight_input.inputs[0].op == "DequantizeLinear") else None

        act_consumers = _consumers(activation_input)
        wt_consumers = _consumers(weight_input)
        safe_remove_dq_act = (dq_act is not None and len(act_consumers) == 1 and act_consumers[0] is matmul_node)
        safe_remove_dq_wt = (dq_wt is not None and len(wt_consumers) == 1 and wt_consumers[0] is matmul_node)

        report.append({
            "matmul": matmul_node.name,
            "dq_activation": dq_act.name if dq_act else None,
            "dq_weight": dq_wt.name if dq_wt else None,
            "activation_consumers": [n.name for n in act_consumers],
            "weight_consumers": [n.name for n in wt_consumers],
            "safe_remove_dq_activation": safe_remove_dq_act,
            "safe_remove_dq_weight": safe_remove_dq_wt,
        })
    return report


def get_fp8_scale(quant_node: gs.Node) -> float:
    """Extract the FP8 quantization scale from a QuantizeLinear node."""
    # QuantizeLinear has inputs: [input, scale, (optional) zero_point]
    scale_tensor = quant_node.inputs[1]
    if isinstance(scale_tensor, gs.Constant):
        return float(scale_tensor.values)
    else:
        print(f"  WARNING: scale for {quant_node.name} is not a constant, using 1.0")
        return 1.0


def _find_downstream_dq(quant_node: gs.Node) -> Optional[gs.Node]:
    """
    Find the DequantizeLinear node immediately after a QuantizeLinear.
    Returns the DQ node or None if not found.
    """
    q_output = quant_node.outputs[0]
    for consumer in _consumers(q_output):
        if consumer.op == "DequantizeLinear":
            return consumer
    return None


def replace_with_plugin(
    graph: gs.Graph,
    matmul_node: gs.Node,
    add_node: gs.Node,
    silu_mul_node: gs.Node,
    quant_node: gs.Node,
) -> gs.Node:
    """
    Replace the [MatMul -> Add -> Silu -> QuantizeLinear -> DequantizeLinear] chain
    with a FusedGemmSiluQuant plugin node that outputs FP16 directly.
    
    The downstream DequantizeLinear is also absorbed: the plugin output connects
    to whatever the DQ node was feeding, bypassing the Q+DQ pair entirely.
    """
    # Extract inputs:
    # MatMul inputs: [activation (after DequantizeLinear), weight (after DequantizeLinear)]
    # We want the ORIGINAL FP8 inputs (before DequantizeLinear)
    
    # Input 0: activation (FP8)
    # The MatMul input[0] might be a DequantizeLinear output
    activation_input = matmul_node.inputs[0]
    dq_activation = None
    if len(activation_input.inputs) == 1 and activation_input.inputs[0].op == "DequantizeLinear":
        dq_activation = activation_input.inputs[0]
        fp8_activation = dq_activation.inputs[0]  # Original FP8 tensor
    else:
        fp8_activation = activation_input  # Already FP8 or no DQ
    
    # Input 1: weight (FP8)
    weight_input = matmul_node.inputs[1]
    dq_weight = None
    if len(weight_input.inputs) == 1 and weight_input.inputs[0].op == "DequantizeLinear":
        dq_weight = weight_input.inputs[0]
        fp8_weight = dq_weight.inputs[0]  # Original FP8 weight
    else:
        fp8_weight = weight_input
    
    # Input 2: bias (FP16)
    # The Add node has two inputs: one is the MatMul output, the other is the bias
    bias_tensor = None
    for inp in add_node.inputs:
        if inp != matmul_node.outputs[0]:
            bias_tensor = inp
            break
    assert bias_tensor is not None, f"Could not find bias for {add_node.name}"
    
    # Find the downstream DequantizeLinear after the QuantizeLinear
    dq_downstream = _find_downstream_dq(quant_node)
    
    # Determine the plugin output tensor:
    # If DQ exists, use the DQ's output tensor (so all DQ consumers automatically get the plugin output)
    # If no DQ, fall back to the QuantizeLinear's output (but mark as FP16)
    if dq_downstream is not None:
        plugin_output = dq_downstream.outputs[0]
    else:
        plugin_output = quant_node.outputs[0]
        print(f"  WARNING: No DequantizeLinear found after {quant_node.name}; "
              f"plugin will output FP16 to QuantizeLinear output tensor")

    # Preserve shape so ONNX value_info is set
    output_shape = getattr(plugin_output, "shape", None)
    if output_shape is not None:
        output_shape = list(output_shape)

    # Get FP8 output scale (kept as plugin attribute for compatibility)
    output_scale = get_fp8_scale(quant_node)

    # ================================================================
    # Input 3: combined_scale = act_dq_scale * wt_dq_scale
    # This recovers the dequantization that the bypassed DQ nodes would do.
    # Math: (A_fp8 * sA) * (B_fp8 * sB) = sA * sB * (A_fp8 * B_fp8)
    # So we multiply the raw FP8 GEMM accumulator by combined_scale in the epilogue.
    # ================================================================
    
    # Extract activation DQ scale (scalar)
    act_dq_scale_val = 1.0
    if dq_activation is not None and len(dq_activation.inputs) >= 2:
        act_scale_tensor = dq_activation.inputs[1]
        if isinstance(act_scale_tensor, gs.Constant):
            act_dq_scale_val = float(act_scale_tensor.values.flatten()[0])
    
    # Extract weight DQ scale (may be per-channel [K] or [N] or scalar)
    wt_dq_scale_arr = None
    if dq_weight is not None and len(dq_weight.inputs) >= 2:
        wt_scale_tensor = dq_weight.inputs[1]
        if isinstance(wt_scale_tensor, gs.Constant):
            wt_dq_scale_arr = np.array(wt_scale_tensor.values).flatten().astype(np.float32)
    
    if wt_dq_scale_arr is None:
        # Fallback: no weight DQ scale found, use 1.0
        print(f"  WARNING: No weight DQ scale found for {matmul_node.name}, using 1.0")
        # We need to know N for the scale vector size
        # Try to get it from bias shape
        bias_shape = getattr(bias_tensor, "shape", None)
        n_dim = int(bias_shape[0]) if bias_shape is not None else 4096
        wt_dq_scale_arr = np.ones(n_dim, dtype=np.float32)
    
    # Compute combined_scale[n] = act_dq_scale * wt_dq_scale[n]
    combined_scale_data = (act_dq_scale_val * wt_dq_scale_arr).astype(np.float32)
    
    # Create as a Constant tensor in the graph
    combined_scale_name = f"{matmul_node.name}_combined_dq_scale"
    combined_scale_tensor = gs.Constant(
        name=combined_scale_name,
        values=combined_scale_data,
    )
    
    print(f"    combined_scale: act={act_dq_scale_val:.6f} * wt_scale[{len(wt_dq_scale_arr)}] "
          f"-> [{combined_scale_data.min():.6f}, {combined_scale_data.max():.6f}]")

    # Create the plugin node with 4 inputs
    plugin_node = gs.Node(
        op="FusedGemmSiluQuant",
        name=f"fused_{matmul_node.name}",
        inputs=[fp8_activation, fp8_weight, bias_tensor, combined_scale_tensor],
        outputs=[plugin_output],
        attrs={
            "output_scale": output_scale,
            "plugin_namespace": "",
            "plugin_version": "1",
        },
    )

    # Disconnect old nodes (MatMul, Add, Sigmoid, Silu-Mul, QuantizeLinear)
    matmul_node.outputs.clear()
    add_node.outputs.clear()
    silu_mul_node.outputs.clear()
    quant_node.outputs.clear()
    
    # Also disconnect the downstream DQ if we absorbed it
    if dq_downstream is not None:
        dq_downstream.outputs.clear()

    # Remove upstream DequantizeLinear nodes only when they are exclusive to this MatMul
    to_remove = []
    if dq_activation is not None:
        act_consumers = _consumers(activation_input)
        if len(act_consumers) == 1 and act_consumers[0] is matmul_node:
            to_remove.append(dq_activation)
    if dq_weight is not None:
        wt_consumers = _consumers(weight_input)
        if len(wt_consumers) == 1 and wt_consumers[0] is matmul_node:
            to_remove.append(dq_weight)
    # Also remove the downstream DQ
    if dq_downstream is not None:
        to_remove.append(dq_downstream)
    for dq_node in to_remove:
        dq_node.outputs.clear()
    if to_remove:
        graph.nodes = [n for n in graph.nodes if n not in to_remove]

    # Add new node
    graph.nodes.append(plugin_node)
    # Force plugin output shape so export and TRT get correct value_info
    if output_shape is not None:
        plugin_output.shape = output_shape
    # Set plugin output dtype to FP16 (matches getOutputDataTypes returning kHALF)
    if hasattr(plugin_output, "dtype"):
        plugin_output.dtype = onnx.TensorProto.FLOAT16

    return plugin_node


def analyze_post_surgery(graph: gs.Graph) -> List[dict]:
    """
    Analyze ONNX graph after surgery: list every DequantizeLinear with its input
    producer and output consumers. Flag nodes that are orphaned (no consumers).
    Does not modify the graph.
    """
    # tensor name -> list of nodes that consume it
    consumers_of: dict = {}
    for node in graph.nodes:
        for inp in node.inputs:
            if getattr(inp, "name", None):
                consumers_of.setdefault(inp.name, []).append(node)
    # tensor name -> node that produces it (first output of that node)
    producer_of: dict = {}
    for node in graph.nodes:
        for t in node.outputs:
            if getattr(t, "name", None):
                producer_of[t.name] = node
    report = []
    for node in graph.nodes:
        if node.op != "DequantizeLinear":
            continue
        out_names = [getattr(t, "name", None) for t in node.outputs if getattr(t, "name", None)]
        out_name = out_names[0] if out_names else None
        consumers = consumers_of.get(out_name, []) if out_name else []
        inp_producers = []
        for inp in node.inputs:
            iname = getattr(inp, "name", None)
            prod = producer_of.get(iname) if iname else None
            inp_producers.append((iname, prod.name if prod else "(initializer or graph input)"))
        report.append({
            "name": node.name,
            "output": out_name,
            "consumers": [n.name for n in consumers],
            "input_producers": inp_producers,
            "orphaned": len(consumers) == 0,
        })
    return report


def process_model(input_path: str, output_path: str, dry_run: bool = False):
    """Load ONNX model, find patterns, replace with plugin nodes, save."""
    
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path, load_external_data=True)
    graph = gs.import_onnx(model)
    
    print(f"Graph: {len(graph.nodes)} nodes, {len(graph.tensors())} tensors")
    
    # Find replaceable patterns
    patterns = find_matmul_silu_pairs(graph)
    print(f"\nFound {len(patterns)} MatMul+Silu+Quantize patterns to fuse:")

    # Pre-surgery analysis: report DQ consumers and safe-to-remove
    pre_report = analyze_quantized_graph(graph)
    report_by_matmul = {r["matmul"]: r for r in pre_report}
    for matmul, add, silu_mul, quant in patterns:
        scale = get_fp8_scale(quant)
        print(f"  {matmul.name}")
        print(f"    bias: {add.inputs[1].name if add.inputs[1] != matmul.outputs[0] else add.inputs[0].name}")
        print(f"    output_scale: {scale}")
        r = report_by_matmul.get(matmul.name)
        if r:
            print(f"    DQ activation: {r['dq_activation']}; consumers: {r['activation_consumers']}; safe_to_remove: {r['safe_remove_dq_activation']}")
            print(f"    DQ weight:     {r['dq_weight']}; consumers: {r['weight_consumers']}; safe_to_remove: {r['safe_remove_dq_weight']}")
            if not r["safe_remove_dq_activation"] or not r["safe_remove_dq_weight"]:
                print(f"    WARNING: At least one DQ has other consumers; it will NOT be removed (only unplugged from MatMul).")
        # Check for downstream DQ
        dq_down = _find_downstream_dq(quant)
        if dq_down:
            print(f"    Downstream DQ: {dq_down.name} -> consumers: {[n.name for n in _consumers(dq_down.outputs[0])]}")
        else:
            print(f"    WARNING: No downstream DequantizeLinear found after {quant.name}")

    if dry_run:
        print(f"\nDry run -- not modifying model.")
        return
    
    if len(patterns) == 0:
        print("\nNo patterns found. The model may already be optimized or use a different pattern.")
        print("Check that the model has been quantized with ModelOpt FP8.")
        # Diagnostic: show MatMul node names that might be linear1 (to debug naming)
        matmul_names = [n.name for n in graph.nodes if n.op == "MatMul"]
        linear1_like = [n for n in matmul_names if "linear1" in n.lower() or "feed_forward" in n.lower()]
        if linear1_like:
            print("Sample MatMul names in graph (linear1/feed_forward):")
            for name in linear1_like[:15]:
                print(f"  {name}")
            if len(linear1_like) > 15:
                print(f"  ... and {len(linear1_like) - 15} more")
        else:
            print("Sample MatMul names in graph (first 10):")
            for name in matmul_names[:10]:
                print(f"  {name}")
        return
    
    # Replace patterns
    print(f"\nReplacing {len(patterns)} patterns with FusedGemmSiluQuant plugin nodes (FP16 output)...")
    dq_absorbed = 0
    for matmul, add, silu_mul, quant in patterns:
        dq_down = _find_downstream_dq(quant)
        plugin = replace_with_plugin(graph, matmul, add, silu_mul, quant)
        if dq_down:
            dq_absorbed += 1
        print(f"  Replaced: {matmul.name} -> {plugin.name}" +
              (f" (absorbed DQ: {dq_down.name})" if dq_down else " (no downstream DQ)"))
    
    # Cleanup: remove disconnected nodes
    graph.cleanup()
    
    print(f"\nGraph after surgery: {len(graph.nodes)} nodes")
    print(f"  Plugin output type: FP16 (absorbed {dq_absorbed} downstream DequantizeLinear nodes)")
    
    # Export
    model = gs.export_onnx(graph)

    # Set value_info for plugin outputs to FP16
    _set_plugin_output_value_info_fp16(model)

    print(f"Saving to: {output_path}")

    # ONNX 1.18 raises if location is absolute. Use a filename-only relative path.
    external_name = os.path.basename(output_path).replace(".onnx", ".onnx_data")
    location = external_name.lstrip("/\\").split(os.sep)[-1].split("/")[-1] or "model_with_plugin.onnx_data"
    if os.path.isabs(location):
        location = external_name  # fallback

    # If the exported model still has external_data refs (e.g. from graphsurgeon), they might be absolute.
    # Load all external data into memory so save() only uses our relative location.
    try:
        from onnx.external_data_helper import load_external_data_for_model
        load_external_data_for_model(model, os.path.dirname(os.path.abspath(input_path)))
    except Exception:
        pass  # model may already have in-memory data

    print(f"Saving with location={repr(location)} (is_abs={os.path.isabs(location)})")
    onnx.save(model, output_path, save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=location)
    
    print(f"Done. {len(patterns)} patterns replaced ({dq_absorbed} DQ nodes absorbed).")
    # Verify plugin output types in saved ONNX
    try:
        _verify_plugin_output_types_in_onnx(output_path)
    except Exception as e:
        print(f"  (Verification skipped: {e})")
    # Post-surgery analysis: list all DequantizeLinear nodes and flag orphans
    try:
        model_check = onnx.load(output_path, load_external_data=True)
        graph_check = gs.import_onnx(model_check)
        post_report = analyze_post_surgery(graph_check)
        if post_report:
            print(f"\n--- Post-surgery DequantizeLinear analysis ({output_path}) ---")
            for r in post_report:
                orphan = " [ORPHAN - no consumers]" if r["orphaned"] else ""
                print(f"  {r['name']}{orphan}")
                print(f"    output -> consumers: {r['consumers']}")
                print(f"    input producers: {r['input_producers']}")
            print("--- End post-surgery analysis ---")
    except Exception as e:
        print(f"Post-surgery analysis skipped: {e}")
    # Root-cause TRT "broadcast dimensions must be conformable" on /layers.0/Add
    try:
        _diagnose_add_shapes(output_path, add_node_name="/layers.0/Add")
    except Exception as e:
        print(f"Diagnostic skipped: {e}")
    print(f"\nTo build TRT engine:")
    print(f"  trtexec --plugins=libfused_gemm_silu_quant.so --onnx={output_path} ...")


def _set_plugin_output_value_info_fp16(model: "onnx.ModelProto") -> None:
    """
    Set value_info for all FusedGemmSiluQuant output tensors to FP16
    so TensorRT infers the correct type for downstream nodes.
    """
    graph = model.graph
    plugin_output_names = []
    for node in graph.node:
        if node.op_type == "FusedGemmSiluQuant":
            for out_name in node.output:
                if out_name:
                    plugin_output_names.append(out_name)
    fp16_elem = onnx.TensorProto.FLOAT16
    for name in plugin_output_names:
        existing = next((vi for vi in graph.value_info if vi.name == name), None)
        if existing is not None:
            if not existing.type.HasField("tensor_type"):
                existing.type.CopyFrom(onnx.helper.make_tensor_type_proto(fp16_elem, None))
            else:
                existing.type.tensor_type.elem_type = fp16_elem
        else:
            vi = onnx.helper.make_tensor_value_info(name, fp16_elem, None)
            graph.value_info.append(vi)
    # Log so server logs show FP16 was set
    if plugin_output_names:
        print(f"  Set value_info to FLOAT16 for {len(plugin_output_names)} plugin output tensor(s)")


def _verify_plugin_output_types_in_onnx(onnx_path: str) -> None:
    """Load saved ONNX and print elem_type for plugin output tensors (FP16=10)."""
    model = onnx.load(onnx_path, load_external_data=True)
    graph = model.graph
    name_to_elem = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            name_to_elem[vi.name] = vi.type.tensor_type.elem_type
    plugin_outs = []
    for node in graph.node:
        if node.op_type == "FusedGemmSiluQuant":
            for out_name in node.output:
                if out_name:
                    plugin_outs.append((out_name, name_to_elem.get(out_name, None)))
    if not plugin_outs:
        return
    fp16_enum = onnx.TensorProto.FLOAT16  # 10
    ok = all(t[1] == fp16_enum for t in plugin_outs)
    for name, elem in plugin_outs[:3]:
        label = "FP16" if elem == fp16_enum else f"elem_type={elem} (expected FP16={fp16_enum})"
        print(f"  Saved ONNX: {name!r} -> {label}")
    if len(plugin_outs) > 3:
        print(f"  ... and {len(plugin_outs) - 3} more plugin outputs")
    if not ok:
        print("  WARNING: Some plugin outputs are not FP16 in saved ONNX.")


def _diagnose_add_shapes(onnx_path: str, add_node_name: str = "/layers.0/Add") -> None:
    """Load ONNX and print shapes for the failing Add and its inputs (for TRT broadcast error)."""
    model = onnx.load(onnx_path, load_external_data=True)
    graph = model.graph
    # value_info / input / output: name -> shape list
    def _dims(v):
        if v.type.HasField("tensor_type") and v.type.tensor_type.HasField("shape"):
            out = []
            for d in v.type.tensor_type.shape.dim:
                if d.dim_param:
                    out.append(d.dim_param)
                else:
                    out.append(d.dim_value)
            return out
        return None
    vi = {vo.name: _dims(vo) for vo in list(graph.value_info) + list(graph.input)}
    for vo in graph.output:
        vi[vo.name] = _dims(vo)
    # initializers have shape from raw dims
    for init in graph.initializer:
        vi[init.name] = list(init.dims)
    # producer: output_name -> node
    producer = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    # Find Add node by name or by output name (graph.node are ONNX NodeProto: use op_type not op)
    add_node = None
    for node in graph.node:
        if node.name == add_node_name or (node.op_type == "Add" and add_node_name in node.name):
            add_node = node
            break
    if add_node is None:
        for node in graph.node:
            if node.op_type == "Add" and any(add_node_name in o for o in node.output):
                add_node = node
                break
    if add_node is None:
        print(f"Diagnostic: Add node '{add_node_name}' not found in graph.")
        return
    print(f"\n--- Shape diagnostic for TRT Add error (node: {add_node.name}) ---")
    for i, inp_name in enumerate(add_node.input):
        if not inp_name:
            continue
        shape = vi.get(inp_name)
        prod = producer.get(inp_name)
        prod_info = prod.name if prod is not None else "(graph input or initializer)"
        print(f"  input[{i}] '{inp_name}'")
        print(f"    shape: {shape}")
        print(f"    producer: {prod_info}")
    out_name = add_node.output[0] if add_node.output else None
    if out_name:
        print(f"  output '{out_name}' shape: {vi.get(out_name)}")
    print("--- End diagnostic ---\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replace MatMul+Silu+Quantize+DequantizeLinear with FusedGemmSiluQuant TRT plugin (FP16 output)")
    parser.add_argument("input", nargs="?", help="Input quantized ONNX model path")
    parser.add_argument("output", nargs="?", help="Output ONNX model path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only find patterns, don't modify the model")
    parser.add_argument("--diagnose", metavar="ONNX", dest="diagnose_path",
                        help="Run shape diagnostic on an ONNX file (e.g. model_with_plugin.onnx) and exit")
    parser.add_argument("--analyze-quantized", metavar="ONNX", dest="analyze_quant_path",
                        help="Analyze quantized ONNX before surgery: DQ consumers and safe-to-remove; no write")
    parser.add_argument("--analyze-post", metavar="ONNX", dest="analyze_post_path",
                        help="Analyze ONNX after surgery: list all DequantizeLinear nodes and orphans; no write")
    args = parser.parse_args()

    if args.analyze_quant_path:
        print(f"Loading (quantized): {args.analyze_quant_path}")
        model = onnx.load(args.analyze_quant_path, load_external_data=True)
        graph = gs.import_onnx(model)
        pre_report = analyze_quantized_graph(graph)
        print(f"\n--- Pre-surgery analysis: {len(pre_report)} replaceable pattern(s) ---")
        for r in pre_report:
            print(f"  MatMul: {r['matmul']}")
            print(f"    DQ activation: {r['dq_activation']}; consumers: {r['activation_consumers']}; safe_to_remove: {r['safe_remove_dq_activation']}")
            print(f"    DQ weight:     {r['dq_weight']}; consumers: {r['weight_consumers']}; safe_to_remove: {r['safe_remove_dq_weight']}")
        print("--- End analysis ---")
        return
    if args.analyze_post_path:
        print(f"Loading (post-surgery): {args.analyze_post_path}")
        model = onnx.load(args.analyze_post_path, load_external_data=True)
        graph = gs.import_onnx(model)
        post_report = analyze_post_surgery(graph)
        print(f"\n--- Post-surgery DequantizeLinear analysis: {len(post_report)} DQ node(s) ---")
        for r in post_report:
            orphan = " [ORPHAN - no consumers]" if r["orphaned"] else ""
            print(f"  {r['name']}{orphan}")
            print(f"    output -> consumers: {r['consumers']}")
            print(f"    input producers: {r['input_producers']}")
        print("--- End analysis ---")
        return
    if args.diagnose_path:
        _diagnose_add_shapes(args.diagnose_path, add_node_name="/layers.0/Add")
        return
    if not args.input or not args.output:
        parser.error("input and output paths required unless --diagnose/--analyze-quantized/--analyze-post is used")
    process_model(args.input, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
