/**
 * SM89 (Ada / RTX 6000 Ada) fused kernel: FP8 GEMM + bias + Silu -> FP16 output.
 *
 * The plugin absorbs both the QuantizeLinear and the downstream DequantizeLinear
 * from the ONNX graph, outputting FP16 directly so TensorRT does not need to
 * infer an FP8 output type for a custom op (a known TRT ONNX-parser limitation).
 *
 * No CUTLASS EVT; single custom CUDA kernel for portability.
 * Target: compute capability 8.9.
 */

#pragma once

#include "kernel_common.h"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace fused_gemm_silu_quant {

namespace sm89 {

// Tile sizes (tune for SM89)
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;

__device__ __forceinline__ float nv_fp8_e4m3_to_float(const __nv_fp8_e4m3 x) {
    return static_cast<float>(x);
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// A [M,K] row-major, B [K,N] column-major (same as CUTLASS LayoutB), output [M,N] row-major
// output_fp16 = Silu( A*B + bias )
// The scale parameter is retained for interface compatibility but not applied;
// the Q+DQ pair has been absorbed and the FP16 result goes directly to the next layer.
__global__ void fused_gemm_silu_quant_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    float /* scale -- kept for launch interface compat */,
    int M,
    int N,
    int K)
{
    // Block covers output tile [row0:row0+BM, col0:col0+BN]
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    __shared__ float A_tile[BM][BK + 1];   // +1 to avoid bank conflicts
    __shared__ float B_tile[BK][BN + 1];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j] = 0.0f;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_row = (tid / 16) * 4;  // 0,4,8,...,60
    const int thread_col = (tid % 16) * 4; // 0,4,8,...,60

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperatively load A_tile [BM x BK]
        for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
            int r = i / BK, c = i % BK;
            int g_row = row0 + r, g_col = k0 + c;
            if (g_row < M && g_col < K)
                A_tile[r][c] = nv_fp8_e4m3_to_float(A[g_row * K + g_col]);
            else
                A_tile[r][c] = 0.0f;
        }
        // Cooperatively load B_tile [BK x BN]; B is column-major so B(k,n) at k + n*K
        for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
            int r = i / BN, c = i % BN;
            int g_row = k0 + r, g_col = col0 + c;
            if (g_row < K && g_col < N)
                B_tile[r][c] = nv_fp8_e4m3_to_float(B[g_row + g_col * K]);
            else
                B_tile[r][c] = 0.0f;
        }
        __syncthreads();

        // Compute 4x4 block of output
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_vals[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                a_vals[i] = A_tile[thread_row + i][k];
            float b_vals[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                b_vals[j] = B_tile[k][thread_col + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

    // Add bias, Silu, store as FP16
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int out_row = row0 + thread_row + i;
        if (out_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int out_col = col0 + thread_col + j;
            if (out_col >= N) continue;
            float b = __half2float(bias[out_col]);
            float val = acc[i][j] + b;
            val = silu(val);
            output[out_row * N + out_col] = __float2half(val);
        }
    }
}

inline size_t get_workspace_size_sm89(const GemmConfig&) {
    return 0;
}

inline int launch_sm89(const GemmConfig& config,
                      const void* input,
                      const void* weight,
                      const void* bias,
                      const void* /* combined_scale -- unused on SM89 */,
                      void* output,
                      void* /* workspace */,
                      cudaStream_t stream) {
    const auto* A = reinterpret_cast<const __nv_fp8_e4m3*>(input);
    const auto* B = reinterpret_cast<const __nv_fp8_e4m3*>(weight);
    const auto* bias_ptr = reinterpret_cast<const __half*>(bias);
    auto* out = reinterpret_cast<__half*>(output);

    dim3 block(16, 16);  // 256 threads
    dim3 grid((config.N + BN - 1) / BN, (config.M + BM - 1) / BM);

    sm89::fused_gemm_silu_quant_kernel<<<grid, block, 0, stream>>>(
        A, B, bias_ptr, out, config.output_scale,
        config.M, config.N, config.K);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

}  // namespace sm89

// Re-export for kernel_common.h API (no inline so .so exports these symbols)
size_t get_workspace_size_sm89(const GemmConfig& config) {
    return sm89::get_workspace_size_sm89(config);
}

int launch_sm89(const GemmConfig& config,
                      const void* input,
                      const void* weight,
                      const void* bias,
                      const void* combined_scale,
                      void* output,
                      void* workspace,
                      cudaStream_t stream) {
    return sm89::launch_sm89(config, input, weight, bias, combined_scale, output, workspace, stream);
}

}  // namespace fused_gemm_silu_quant
