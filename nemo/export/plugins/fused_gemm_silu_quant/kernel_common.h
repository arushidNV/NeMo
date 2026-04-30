/**
 * Common types and launch declarations for Fused GEMM + Silu + FP8 Quantize.
 * Used by the plugin for arch dispatch; implementations are in kernel.cuh (SM90)
 * and kernel_sm89.cuh (SM89 / RTX 6000 Ada).
 */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace fused_gemm_silu_quant {

struct GemmConfig {
    int M;               // batch * seq_len (dynamic)
    int N;               // 4096 (ff intermediate dim)
    int K;               // 1024 (hidden dim)
    float output_scale;  // FP8 output quantization scale (kept for compat, unused)
};

// SM90 (Hopper / H100) – CUTLASS implementation
size_t get_workspace_size_sm90(const GemmConfig& config);
int launch_sm90(const GemmConfig& config,
                const void* input,          // [M, K] FP8 E4M3
                const void* weight,         // [K, N] FP8 E4M3
                const void* bias,           // [N] FP16
                const void* combined_scale, // [N] FP32: act_dq_scale * wt_dq_scale[n]
                void* output,               // [M, N] FP16
                void* workspace,
                cudaStream_t stream);

// SM89 (Ada / RTX 6000 Ada) – custom kernel, no CUTLASS EVT
size_t get_workspace_size_sm89(const GemmConfig& config);
int launch_sm89(const GemmConfig& config,
                const void* input,
                const void* weight,
                const void* bias,
                const void* combined_scale, // [N] FP32
                void* output,
                void* workspace,
                cudaStream_t stream);

}  // namespace fused_gemm_silu_quant
