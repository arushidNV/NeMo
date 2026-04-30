/**
 * TensorRT IPluginV3 wrapper for the Fused GEMM + Silu + FP8 Quantize kernel.
 *
 * Plugin name: "FusedGemmSiluQuant"
 * Plugin version: "1"
 * Plugin namespace: "fused_fp8"
 *
 * Inputs:
 *   0: activation      [M, K] or [batch, seq, K] in FP8 E4M3
 *   1: weight          [K, N] in FP8 E4M3
 *   2: bias            [N]    in FP16
 *   3: combined_scale  [N]    in FP32 (= act_dq_scale * wt_dq_scale[n])
 *
 * Outputs:
 *   0: output     [M, N] or [batch, seq, N] in FP16 (same rank as input 0; required for TRT shape binding)
 *
 * NOTE: The plugin absorbs both the QuantizeLinear and downstream DequantizeLinear
 * from the ONNX graph, outputting FP16 directly. This works around a known TRT
 * ONNX-parser limitation where custom plugin FP8 output types are not propagated.
 *
 * Attributes:
 *   output_scale: float (FP8 quantization scale for output)
 *
 * This plugin fuses the native TRT layers linear1/MatMul_myl0_* and __myl_SiluMulCast_myl0_*.
 * Dtypes and shapes are aligned with rnnt_fp8_kernel_layers.json (e.g. FP8 [-1,1024]/[-1,4096]
 * GEMM and FP8 [-1,120,4096] Silu output).
 */

#pragma once

#include <NvInferPlugin.h>
#include <string>
#include <vector>

namespace fused_gemm_silu_quant {

// Plugin metadata
static const char* PLUGIN_NAME = "FusedGemmSiluQuant";
static const char* PLUGIN_VERSION = "1";
static const char* PLUGIN_NAMESPACE = "";

// ============================================================================
// IPluginV3 Implementation
// ============================================================================

class FusedGemmSiluQuantPlugin : public nvinfer1::IPluginV3,
                                  public nvinfer1::IPluginV3OneCore,
                                  public nvinfer1::IPluginV3OneBuild,
                                  public nvinfer1::IPluginV3OneRuntime {
public:
    FusedGemmSiluQuantPlugin() = default;
    FusedGemmSiluQuantPlugin(float output_scale);
    ~FusedGemmSiluQuantPlugin() override = default;

    // ---- IPluginV3 ----
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // ---- IPluginV3OneCore ----
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // ---- IPluginV3OneBuild ----
    int32_t getNbOutputs() const noexcept override;

    int32_t configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(
        nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    // ---- IPluginV3OneRuntime ----
    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override;

    size_t getWorkspaceSize(
        nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    // Serialization
    size_t getSerializationSize() const noexcept;
    void serialize(void* buffer) const noexcept;

private:
    float output_scale_ = 1.0f;
    int K_ = 0;  // cached from configurePlugin
    int N_ = 0;

    // CUTLASS workspace (allocated by TRT via getWorkspaceSize)
    void* cutlass_workspace_ = nullptr;

    // Per-instance serialization storage (getFieldsToSerialize must return
    // PluginField entries with non-null data pointers to the actual values).
    std::vector<nvinfer1::PluginField> serialize_fields_;
    nvinfer1::PluginFieldCollection serialize_collection_{};
};

// ============================================================================
// IPluginCreatorV3One Implementation
// ============================================================================

class FusedGemmSiluQuantCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedGemmSiluQuantCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    nvinfer1::IPluginV3* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

private:
    static std::vector<nvinfer1::PluginField> plugin_fields_;
    static nvinfer1::PluginFieldCollection field_collection_;
    std::string namespace_;
};

}  // namespace fused_gemm_silu_quant
