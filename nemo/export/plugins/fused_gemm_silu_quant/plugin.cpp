/**
 * TensorRT plugin implementation for Fused GEMM + Silu + FP8 Quantize.
 * Dispatches to SM90 (H100) or SM89 (RTX 6000 Ada) based on device.
 */

#include "plugin.h"
#include "kernel_common.h"

#include <cstring>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <cuda_runtime_api.h>

namespace fused_gemm_silu_quant {

// ============================================================================
// Plugin Implementation
// ============================================================================

FusedGemmSiluQuantPlugin::FusedGemmSiluQuantPlugin(float output_scale)
    : output_scale_(output_scale) {}

nvinfer1::IPluginCapability* FusedGemmSiluQuantPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        default:
            return nullptr;
    }
}

nvinfer1::IPluginV3* FusedGemmSiluQuantPlugin::clone() noexcept {
    auto* plugin = new FusedGemmSiluQuantPlugin(output_scale_);
    plugin->K_ = K_;
    plugin->N_ = N_;
    return plugin;
}

// ---- Core ----

char const* FusedGemmSiluQuantPlugin::getPluginName() const noexcept {
    return PLUGIN_NAME;
}

char const* FusedGemmSiluQuantPlugin::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

char const* FusedGemmSiluQuantPlugin::getPluginNamespace() const noexcept {
    return PLUGIN_NAMESPACE;
}

// ---- Build ----

int32_t FusedGemmSiluQuantPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t FusedGemmSiluQuantPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    // Input 0: activation [M, K] FP8
    // Input 1: weight [K, N] FP8
    // Input 2: bias [N] FP16
    // Input 3: combined_scale [N] FP32 (act_dq_scale * wt_dq_scale[n])
    assert(nbInputs == 4);
    assert(nbOutputs == 1);

    // Cache dimensions from weight tensor (static)
    K_ = in[1].desc.dims.d[0];
    N_ = in[1].desc.dims.d[1];

    return 0;
}

bool FusedGemmSiluQuantPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept {
    // pos 0: activation      - FP8 E4M3, linear format
    // pos 1: weight          - FP8 E4M3, linear format
    // pos 2: bias            - FP16, linear format
    // pos 3: combined_scale  - FP32, linear format
    // pos 4: output          - FP16, linear format
    bool isLinear = (inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR);

    switch (pos) {
        case 0:  // activation
            return isLinear && inOut[pos].desc.type == nvinfer1::DataType::kFP8;
        case 1:  // weight
            return isLinear && inOut[pos].desc.type == nvinfer1::DataType::kFP8;
        case 2:  // bias
            return isLinear && inOut[pos].desc.type == nvinfer1::DataType::kHALF;
        case 3:  // combined_scale
            return isLinear && inOut[pos].desc.type == nvinfer1::DataType::kFLOAT;
        case 4:  // output — FP16
            return isLinear && inOut[pos].desc.type == nvinfer1::DataType::kHALF;
        default:
            return false;
    }
}

int32_t FusedGemmSiluQuantPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept {
    assert(nbOutputs == 1);
    // Output is FP16 — plugin absorbs the Q+DQ pair and outputs FP16 directly.
    // This avoids a known TRT ONNX-parser limitation where custom plugin FP8
    // output types are not propagated, causing DequantizeLinear type-inference failure.
    outputTypes[0] = nvinfer1::DataType::kHALF;
    return 0;
}

int32_t FusedGemmSiluQuantPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    assert(nbInputs == 3);
    assert(nbOutputs == 1);

    // Output must match activation rank so TRT binds the same dynamic dims as the rest of the graph.
    // Conformer/encoder uses 3D [batch, seq, K]; reporting 2D [M, N] caused "broadcast dimensions
    // must be conformable" on the residual Add because TRT saw plugin output as 2D vs 3D residual.
    int nd = inputs[0].nbDims;
    if (nd != 2 && nd != 3)
        return -1;  // unsupported rank
    outputs[0].nbDims = nd;
    for (int i = 0; i < nd - 1; ++i)
        outputs[0].d[i] = inputs[0].d[i];  // propagate batch, seq (or M)
    outputs[0].d[nd - 1] = inputs[1].d[1];  // N

    return 0;
}

// ---- Runtime ----

int32_t FusedGemmSiluQuantPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

    // Extract dimensions (support 2D [M,K] or 3D [batch, seq, K] only)
    int const nd = inputDesc[0].dims.nbDims;
    if (nd != 2 && nd != 3)
        return -1;
    int M, K;
    if (nd == 3) {
        M = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
        K = inputDesc[0].dims.d[2];
    } else {
        M = inputDesc[0].dims.d[0];
        K = inputDesc[0].dims.d[1];
    }
    int const N = inputDesc[1].dims.d[1];

    GemmConfig config;
    config.M = M;
    config.N = N;
    config.K = K;
    config.output_scale = output_scale_;

    int status = -1;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return -1;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return -1;
    int arch = prop.major * 10 + prop.minor;

    if (arch >= 90) {
        status = launch_sm90(config, inputs[0], inputs[1], inputs[2],
                            inputs[3], outputs[0], workspace, stream);
    } else if (arch == 89) {
        status = launch_sm89(config, inputs[0], inputs[1], inputs[2],
                            inputs[3], outputs[0], workspace, stream);
    }

    return status;
}

int32_t FusedGemmSiluQuantPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    int const nd = in[0].dims.nbDims;
    K_ = (nd == 3) ? in[0].dims.d[2] : in[0].dims.d[1];
    N_ = in[1].dims.d[1];
    return 0;
}

nvinfer1::IPluginV3* FusedGemmSiluQuantPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept {
    return clone();
}

size_t FusedGemmSiluQuantPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept {
    int const nd = inputs[0].max.nbDims;
    if (nd != 2 && nd != 3)
        return 0;
    int maxM = (nd == 3) ? (inputs[0].max.d[0] * inputs[0].max.d[1]) : inputs[0].max.d[0];
    int K = (nd == 3) ? inputs[0].max.d[2] : inputs[0].max.d[1];
    int N = inputs[1].max.d[1];

    GemmConfig config;
    config.M = maxM;
    config.N = N;
    config.K = K;
    config.output_scale = output_scale_;

    size_t ws90 = get_workspace_size_sm90(config);
    size_t ws89 = get_workspace_size_sm89(config);
    return std::max(ws90, ws89);
}

nvinfer1::PluginFieldCollection const* FusedGemmSiluQuantPlugin::getFieldsToSerialize() noexcept {
    // IPluginV3 serialization: TRT reads data from the PluginField::data pointers.
    // Must point to this instance's output_scale_ (not nullptr).
    serialize_fields_.clear();
    serialize_fields_.push_back({"output_scale", &output_scale_, nvinfer1::PluginFieldType::kFLOAT32, 1});
    serialize_collection_ = {static_cast<int32_t>(serialize_fields_.size()), serialize_fields_.data()};
    return &serialize_collection_;
}

size_t FusedGemmSiluQuantPlugin::getSerializationSize() const noexcept {
    return sizeof(float);  // output_scale_
}

void FusedGemmSiluQuantPlugin::serialize(void* buffer) const noexcept {
    memcpy(buffer, &output_scale_, sizeof(float));
}

// ============================================================================
// Creator Implementation
// ============================================================================

std::vector<nvinfer1::PluginField> FusedGemmSiluQuantCreator::plugin_fields_ = {
    {"output_scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
};

nvinfer1::PluginFieldCollection FusedGemmSiluQuantCreator::field_collection_ = {
    static_cast<int32_t>(FusedGemmSiluQuantCreator::plugin_fields_.size()),
    FusedGemmSiluQuantCreator::plugin_fields_.data()
};

FusedGemmSiluQuantCreator::FusedGemmSiluQuantCreator() {
    namespace_ = PLUGIN_NAMESPACE;
}

char const* FusedGemmSiluQuantCreator::getPluginName() const noexcept {
    return PLUGIN_NAME;
}

char const* FusedGemmSiluQuantCreator::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

char const* FusedGemmSiluQuantCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

void FusedGemmSiluQuantCreator::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns;
}

nvinfer1::IPluginV3* FusedGemmSiluQuantCreator::createPlugin(
    char const* name,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase phase) noexcept {

    float output_scale = 1.0f;

    for (int32_t i = 0; i < fc->nbFields; ++i) {
        auto const& field = fc->fields[i];
        if (strcmp(field.name, "output_scale") == 0) {
            output_scale = *static_cast<const float*>(field.data);
        }
    }

    if (phase == nvinfer1::TensorRTPhase::kBUILD
        || phase == nvinfer1::TensorRTPhase::kRUNTIME) {
        return new FusedGemmSiluQuantPlugin(output_scale);
    }
    return nullptr;
}

nvinfer1::PluginFieldCollection const* FusedGemmSiluQuantCreator::getFieldNames() noexcept {
    return &field_collection_;
}

}  // namespace fused_gemm_silu_quant

// ============================================================================
// Plugin Registration
// ============================================================================
// REGISTER_TENSORRT_PLUGIN relies on static initialization which may not
// trigger reliably when the .so is loaded via ctypes.CDLL / dlopen.
// We provide an explicit C init function that Python calls after loading.
// The static macro is kept as a fallback (e.g. trtexec --plugins=...).

using fused_gemm_silu_quant::FusedGemmSiluQuantCreator;
REGISTER_TENSORRT_PLUGIN(FusedGemmSiluQuantCreator);

extern "C" {
/// Explicit init: call from Python via ctypes after loading the .so.
/// Safe to call multiple times (only registers once).
int initFusedGemmSiluQuantPlugin() {
    static FusedGemmSiluQuantCreator creator;
    auto* registry = getPluginRegistry();
    if (!registry) return -1;
    // registerCreator returns bool; ignore duplicate-registration errors.
    registry->registerCreator(creator, "");
    return 0;
}
}
