// warp_decode: gate_up kernel for BF16 MoE decode.
//
// Layout (matches sglang fused_moe / bench_bf16_fuse_moe.py):
//   X   : [B, hidden]                  bf16
//   W   : [E, 2 * inter, hidden]       bf16       (rows [0, inter) = gate == W1,
//                                                  rows [inter, 2*inter) = up  == W3)
//   ids : [B, topk]                    int32
//   out : [B * topk, inter]            bf16       (== silu(gate) * up, fused)
//
// Parallelism (warp-per-neuron):
//   gridDim.x = B * topk                 (one CTA per (token, topk_slot))
//   gridDim.y = inter / WARPS_PER_CTA    (each CTA covers WARPS_PER_CTA neurons)
//   blockDim.x = WARPS_PER_CTA * 32
//   Each warp owns one output neuron n \in [0, inter), computes
//       g = sum_k x[token, k] * W[expert, n,        k]
//       u = sum_k x[token, k] * W[expert, n+inter,  k]
//       ou2u(g) * u
//
// Notes:
//   * All 8 warps read x[token] directly from global memory; caches provide reuse.
//   * hidden must be a multiple of 32 * VEC = 256 (VEC = 8 bf16 per uint4 load).
//   * inter must be a multiple of WARPS_PER_CTA (=8 by default).


#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace warp_decode {

constexpr int WARPS_PER_CTA   = 8;
constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;
constexpr int VEC             = 8;   // 8 bf16 == 16 B == uint4
static_assert(VEC == 8, "uint4 loads must contain exactly 8 BF16 values");

// BF16 is the high 16 bits of FP32. Expanding both packed halves directly
// avoids extracting each 16-bit value before converting it to FP32.
__device__ __forceinline__ float2 unpack_bf16_pair(uint32_t packed) {
    return make_float2(__uint_as_float(packed << 16),
                      __uint_as_float(packed & 0xffff0000u));
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // full warp reduction via xor butterfly
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__global__ void moe_gate_up_3d_batched_kernel(
    const __nv_bfloat16* __restrict__ X,        // [B, hidden]
    const __nv_bfloat16* __restrict__ W,        // [E, 2*inter, hidden]
    const int*           __restrict__ topk_ids, // [B, topk]
    __nv_bfloat16*       __restrict__ out,      // [B*topk, inter]
    int B, int E, int topk, int hidden, int inter)
{
    const int flat_tok = blockIdx.x;                 // in [0, B*topk)
    const int token_id = flat_tok / topk;
    const int slot     = flat_tok - token_id * topk;

    const int warp_id  = threadIdx.x >> 5;
    const int lane_id  = threadIdx.x & 31;

    const int neuron_n = blockIdx.y * WARPS_PER_CTA + warp_id;
    // Boundary guard for inter % WARPS_PER_CTA != 0 (currently we assert host-side).
    if (neuron_n >= inter) return;

    const int expert_id = topk_ids[token_id * topk + slot];

    // All warps use the same token row. Read it directly through the cache
    // hierarchy instead of cooperatively staging it in shared memory.
    const __nv_bfloat16* __restrict__ x_ptr = X + (size_t)token_id * hidden;

    // Each warp streams its gate/up weight rows and computes both dot products.
    const size_t expert_stride = (size_t)(2 * inter) * hidden;
    const __nv_bfloat16* __restrict__ gate_ptr =
        W + (size_t)expert_id * expert_stride + (size_t)neuron_n * hidden;
    const __nv_bfloat16* __restrict__ up_ptr   =
        gate_ptr + (size_t)inter * hidden;

    float acc_gate = 0.f;
    float acc_up   = 0.f;

    #pragma unroll 1
    for (int k = lane_id * VEC; k < hidden; k += 32 * VEC) {
        uint4 xv = *reinterpret_cast<const uint4*>(&x_ptr[k]);
        uint4 gv = *reinterpret_cast<const uint4*>(&gate_ptr[k]);
        uint4 uv = *reinterpret_cast<const uint4*>(&up_ptr[k]);

        const uint32_t x_words[VEC / 2] = {xv.x, xv.y, xv.z, xv.w};
        const uint32_t g_words[VEC / 2] = {gv.x, gv.y, gv.z, gv.w};
        const uint32_t u_words[VEC / 2] = {uv.x, uv.y, uv.z, uv.w};

        // Unrolling keeps the packed words in registers. Preserve the original
        // even/odd element order and perform all multiplication/summing in FP32.
        #pragma unroll
        for (int i = 0; i < VEC / 2; ++i) {
            const float2 xf = unpack_bf16_pair(x_words[i]);
            const float2 gf = unpack_bf16_pair(g_words[i]);
            const float2 uf = unpack_bf16_pair(u_words[i]);
            acc_gate += xf.x * gf.x;
            acc_up   += xf.x * uf.x;
            acc_gate += xf.y * gf.y;
            acc_up   += xf.y * uf.y;
        }
    }

    acc_gate = warp_reduce_sum(acc_gate);
    acc_up   = warp_reduce_sum(acc_up);

    if (lane_id == 0) {
        float y = silu_f32(acc_gate) * acc_up;
        out[(size_t)flat_tok * inter + neuron_n] = __float2bfloat16(y);
    }
}

void launch_moe_gate_up_3d_batched(
    const __nv_bfloat16* X,
    const __nv_bfloat16* W,
    const int*           topk_ids,
    __nv_bfloat16*       out,
    int B, int E, int topk, int hidden, int inter,
    cudaStream_t stream)
{
    dim3 grid(B * topk, inter / WARPS_PER_CTA);
    dim3 block(THREADS_PER_CTA);
    moe_gate_up_3d_batched_kernel<<<grid, block, 0, stream>>>(
        X, W, topk_ids, out, B, E, topk, hidden, inter);
}

}  // namespace warp_decode
