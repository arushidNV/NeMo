/**
 * SM90 (Hopper / H100) entry points for Fused GEMM + DQ Scale + Bias + Silu (FP16 output).
 * Wraps the CUTLASS implementation in kernel.cuh; compiled only for sm_90.
 */

#include "kernel_common.h"
#include "kernel.cuh"

namespace fused_gemm_silu_quant {

size_t get_workspace_size_sm90(const GemmConfig& config) {
    return get_workspace_size(config);
}

int launch_sm90(const GemmConfig& config,
                const void* input,
                const void* weight,
                const void* bias,
                const void* combined_scale,
                void* output,
                void* workspace,
                cudaStream_t stream) {
    auto* a = reinterpret_cast<const ElementA*>(input);
    auto* b = reinterpret_cast<const ElementB*>(weight);
    auto* bias_ptr = reinterpret_cast<const ElementBias*>(bias);
    auto* scale_ptr = reinterpret_cast<const ElementScale*>(combined_scale);
    auto* d = reinterpret_cast<ElementD*>(output);
    auto status = launch(config, a, b, bias_ptr, scale_ptr, d, workspace, stream);
    return (status == cutlass::Status::kSuccess) ? 0 : -1;
}

}  // namespace fused_gemm_silu_quant
