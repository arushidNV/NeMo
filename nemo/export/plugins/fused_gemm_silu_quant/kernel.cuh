/**
 * FP8 GEMM + DQ Scale + Bias + Silu Fused Kernel -> FP16 output
 *
 * Replaces the pattern:
 *   DQ(act) -> MatMul <- DQ(wt) -> Add(bias) -> SiLU -> Q -> DQ
 *
 * With a single CUTLASS kernel:
 *   FP8 GEMM -> multiply by combined DQ scale -> add bias -> Silu -> write FP16
 *
 * The combined_scale[n] = act_dq_scale * wt_dq_scale[n] is precomputed and
 * passed as a [N] FP32 vector. This correctly recovers the dequantized values:
 *   result = Silu( (A_fp8 * B_fp8) * combined_scale + bias )
 *
 * Mathematically: (A_fp8 * sA) * (B_fp8 * sB) = sA * sB * (A_fp8 * B_fp8)
 * So we can run the raw FP8 GEMM and multiply the accumulator by the scales.
 *
 * Target: NVIDIA Hopper (SM90), FP8 E4M3 tensor cores
 * Requires: CUTLASS 3.x
 */

#pragma once

#include "kernel_common.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"

#include "cute/tensor.hpp"

namespace fused_gemm_silu_quant {

// ============================================================================
// Type aliases
// ============================================================================

using ElementA    = cutlass::float_e4m3_t;  // Activation: FP8 E4M3
using ElementB    = cutlass::float_e4m3_t;  // Weight: FP8 E4M3
using ElementD    = cutlass::half_t;         // Output: FP16
using ElementBias = cutlass::half_t;         // Bias: FP16
using ElementScale = float;                  // Combined DQ scale: FP32
using ElementAcc  = float;                   // Accumulator: FP32
using ElementCompute = float;                // Epilogue compute: FP32

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // Transposed weight for TN GEMM
using LayoutD = cutlass::layout::RowMajor;

// Alignment
static constexpr int AlignmentA = 16;  // FP8: 128-bit / 1 byte = 16
static constexpr int AlignmentB = 16;
static constexpr int AlignmentD = 8;   // FP16: 128-bit / 2 bytes = 8

// Architecture
using ArchTag   = cutlass::arch::Sm90;
using OpClass   = cutlass::arch::OpClassTensorOp;

// Tile shape
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

// ============================================================================
// Custom Silu Functor
// ============================================================================

template <typename T>
struct SiluActivation {
    CUTLASS_HOST_DEVICE
    T operator()(T x) const {
        T one = T(1);
        return x / (one + cutlass::fast_exp(-x));
    }
};

// ============================================================================
// Epilogue Visitor Tree (EVT) Definition
// ============================================================================
//
// Computes: output_fp16 = Silu(Acc * combined_scale + bias)
//
// Tree structure:
//
//   Sm90Compute<SiluActivation>                    <- Silu(scaled_plus_bias)
//     Sm90Compute<plus>                            <- scaled_acc + bias
//       Sm90Compute<multiplies>                    <- Acc * combined_scale
//         Sm90AccFetch                             <- raw FP8 GEMM accumulator
//         Sm90RowBroadcast<combined_scale, FP32>   <- [N] DQ scale vector
//       Sm90RowBroadcast<bias, FP16>               <- [N] bias vector
//

using namespace cutlass::epilogue::fusion;

// Level 1: Acc * combined_scale
using EVT_ScaleAcc = Sm90EVT<
    Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    Sm90AccFetch,                                        // FP32 GEMM accumulator
    Sm90RowBroadcast<0, TileShape, ElementScale,         // combined_scale [N], FP32
                     cute::Stride<cute::_0, cute::_1, cute::_0>>
>;

// Level 2: scaled_acc + bias
using EVT_ScaleAccPlusBias = Sm90EVT<
    Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    EVT_ScaleAcc,                                        // Acc * combined_scale
    Sm90RowBroadcast<0, TileShape, ElementBias,          // bias [N], FP16
                     cute::Stride<cute::_0, cute::_1, cute::_0>>
>;

// Level 3 (root): Silu(scaled_acc + bias) -> store as FP16
using EVT_SiluScaleBias = Sm90EVT<
    Sm90Compute<SiluActivation, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    EVT_ScaleAccPlusBias
>;

// ============================================================================
// Collective Epilogue
// ============================================================================

using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    void, cutlass::layout::RowMajor, 0,     // No C matrix
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    EVT_SiluScaleBias
>::CollectiveOp;

// ============================================================================
// Collective Mainloop (FP8 GEMM on Hopper tensor cores)
// ============================================================================

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::TensorStorage))
    >,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

// ============================================================================
// Full GEMM Kernel
// ============================================================================

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ============================================================================
// Launch function
// ============================================================================

inline cutlass::Status launch(
    const GemmConfig& config,
    const ElementA* input,
    const ElementB* weight,
    const ElementBias* bias,
    const ElementScale* combined_scale,  // [N] FP32: act_dq_scale * wt_dq_scale[n]
    ElementD* output,
    void* workspace,
    cudaStream_t stream)
{
    auto problem_shape = cute::make_shape(config.M, config.N, config.K, 1);

    // Strides: use the kernel's own types (cute::tuple<int64_t, C<1>, int64_t>)
    // RowMajor [M,K]: stride = (K, 1, M*K)  — leading dim is K
    // ColMajor [K,N]: stride = (K, 1, K*N)  — leading dim is K (stored as transpose)
    // RowMajor [M,N]: stride = (N, 1, M*N)  — leading dim is N
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A;
    cute::get<0>(stride_A) = int64_t(config.K);       // A [M,K] row-major: leading dim = K
    cute::get<2>(stride_A) = int64_t(config.M) * int64_t(config.K);

    StrideB stride_B;
    cute::get<0>(stride_B) = int64_t(config.K);       // B [K,N] col-major: leading dim = K
    cute::get<2>(stride_B) = int64_t(config.K) * int64_t(config.N);

    StrideD stride_D;
    cute::get<0>(stride_D) = int64_t(config.N);       // D [M,N] row-major: leading dim = N
    cute::get<2>(stride_D) = int64_t(config.M) * int64_t(config.N);

    // Construct arguments using named field access.
    // Sm90EVT Arguments use fields: op_0, op_1, op_2 for children.
    // Sm90RowBroadcast<0,...> Arguments use: ptr_row, null_default, dRow.
    //
    // Our EVT tree:
    //   SiluScaleBias { op_0 = ScaleAccPlusBias }  (1-child Sm90EVT)
    //     ScaleAccPlusBias { op_0 = ScaleAcc, op_1 = RowBroadcast(bias) }  (2-child)
    //       ScaleAcc { op_0 = AccFetch, op_1 = RowBroadcast(scale) }  (2-child)
    //
    // Field path to scale:  thread.op_0.op_0.op_1.ptr_row
    // Field path to bias:   thread.op_0.op_1.ptr_row

    // Step 1: Build EVT thread args (default-init, then set broadcast pointers)
    typename CollectiveEpilogue::FusionCallbacks::Arguments thread_args{};
    thread_args.op_0.op_0.op_1.ptr_row = combined_scale;       // RowBroadcast(scale)
    thread_args.op_0.op_0.op_1.null_default = ElementScale(0);
    thread_args.op_0.op_1.ptr_row = bias;                      // RowBroadcast(bias)
    thread_args.op_0.op_1.null_default = ElementBias(0);

    // Step 2: Build epilogue args
    typename Gemm::GemmKernel::EpilogueArguments epilogue_args{};
    epilogue_args.thread = thread_args;
    epilogue_args.ptr_C = nullptr;
    epilogue_args.ptr_D = output;
    epilogue_args.dD = stride_D;

    // Step 3: Build mainloop args
    typename Gemm::GemmKernel::MainloopArguments mainloop_args{};
    mainloop_args.ptr_A = input;
    mainloop_args.dA = stride_A;
    mainloop_args.ptr_B = weight;
    mainloop_args.dB = stride_B;

    // Step 4: Full GEMM arguments
    typename Gemm::Arguments arguments{};
    arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    arguments.problem_shape = problem_shape;
    arguments.mainloop = mainloop_args;
    arguments.epilogue = epilogue_args;
    arguments.hw_info = cutlass::KernelHardwareInfo{};

    Gemm gemm;

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    status = gemm.initialize(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    return gemm.run(stream);
}

/**
 * Get the required workspace size in bytes.
 */
inline size_t get_workspace_size(const GemmConfig& config) {
    auto problem_shape = cute::make_shape(config.M, config.N, config.K, 1);

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A;
    cute::get<0>(stride_A) = int64_t(config.K);
    cute::get<2>(stride_A) = int64_t(config.M) * int64_t(config.K);
    StrideB stride_B;
    cute::get<0>(stride_B) = int64_t(config.K);
    cute::get<2>(stride_B) = int64_t(config.K) * int64_t(config.N);
    StrideD stride_D;
    cute::get<0>(stride_D) = int64_t(config.N);
    cute::get<2>(stride_D) = int64_t(config.M) * int64_t(config.N);

    typename Gemm::Arguments arguments{};
    arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    arguments.problem_shape = problem_shape;
    arguments.mainloop.dA = stride_A;
    arguments.mainloop.dB = stride_B;
    arguments.epilogue.dD = stride_D;

    Gemm gemm;
    return gemm.get_workspace_size(arguments);
}

}  // namespace fused_gemm_silu_quant
