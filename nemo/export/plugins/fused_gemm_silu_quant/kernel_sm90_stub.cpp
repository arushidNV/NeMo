/**
 * Stub implementations when SM90/CUTLASS is not built (e.g. SM89-only machine).
 * get_workspace_size_sm90 returns 0; launch_sm90 returns -1 (not implemented).
 */

#include "kernel_common.h"
#include <cstddef>
#include <cuda_runtime_api.h>

namespace fused_gemm_silu_quant {

size_t get_workspace_size_sm90(const GemmConfig&) {
    return 0;
}

int launch_sm90(const GemmConfig&, const void*, const void*, const void*, const void*, void*, void*, cudaStream_t) {
    return -1;  // SM90 kernel not built
}

}  // namespace fused_gemm_silu_quant
