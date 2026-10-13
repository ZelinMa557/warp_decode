// warp_decode: down kernel for BF16 MoE decode.
//
// Layout:
//   buffer     : [B, topk, inter]      bf16   (== [B*topk, inter], gate_up output)
//   W_down     : [E, hidden, inter]    bf16
//   topk_ids   : [B, topk]             int32
//   topk_scale : [B, topk]             fp32
//   out        : [B, hidden]           bf16
//
// Parallelism (warp-per-output-group, no shared memory):
//   Each warp owns a group of adjacent output dimensions for one token.
//   Input values are reused across the group's independent FP32 dot products.
//   buffer[token, :, :] fits in L1/L2 (3 KB @ Qwen3-TP8), so no smem staging.
//
//   1D grid over (token, output group). For inter=192, B=1..2 uses
//   2 outputs/warp and 4 warps/CTA; B=3..16 uses 8 outputs/warp and 2 warps/CTA.
//   The generic path uses 8 outputs/warp and 8 warps/CTA.
//
// Constraints:
//   * inter  % 64 == 0                    (half_k = inter/2, bf16x2 vec)
//   * hidden % OUTS_PER_WARP == 0

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>

namespace warp_decode {

constexpr int WARP_SIZE_DOWN       = 32;
constexpr int WARPS_PER_CTA_DOWN   = 8;
constexpr int THREADS_PER_CTA_DOWN = WARPS_PER_CTA_DOWN * WARP_SIZE_DOWN;

// Output grouping for the generic path. The small-batch path uses two outputs
// per warp to expose more parallel work and shorten each warp's instruction stream.
constexpr int OUTS_PER_WARP = 8;

__device__ __forceinline__ float warp_reduce_sum_d(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

template<int STATIC_INTER>
__device__ __forceinline__ float2 widen_bf16_pair_d(__nv_bfloat162 value) {
    if constexpr (STATIC_INTER == 192) {
        // BF16 occupies the high half of FP32: expand both packed operands
        // directly, before any multiplication, without rounding the product.
        const __nv_bfloat162_raw raw = static_cast<__nv_bfloat162_raw>(value);
        uint32_t bits;
        static_assert(sizeof(raw) == sizeof(bits), "A BF16 pair occupies four bytes");
        memcpy(&bits, &raw, sizeof(bits));
        return make_float2(__uint_as_float(bits << 16),
                          __uint_as_float(bits & 0xffff0000u));
    } else {
        return __bfloat1622float2(value);
    }
}

template<int OUTS, int STATIC_INTER = 0>
__global__ void moe_down_3d_batched_kernel(
    const __nv_bfloat16* __restrict__ buffer,     // [B, topk, inter]
    const __nv_bfloat16* __restrict__ W,          // [E, hidden, inter]
    const int*           __restrict__ topk_ids,   // [B, topk]
    const float*         __restrict__ topk_scale, // [B, topk]
    __nv_bfloat16*       __restrict__ out,        // [B, hidden]
    int B, int E, int topk, int hidden, int inter)
{
    // A fixed inter makes all three BF16-pair iterations compile-time known.
    if constexpr (STATIC_INTER != 0) inter = STATIC_INTER;

    const size_t global_tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t global_warp = global_tid / WARP_SIZE_DOWN;
    const int    lane_id     = threadIdx.x & (WARP_SIZE_DOWN - 1);

    const int    groups_per_token = hidden / OUTS;
    const size_t total_warps      = (size_t)B * groups_per_token;
    if (global_warp >= total_warps) return;

    const int token_id  = (int)(global_warp / groups_per_token);
    const int group_idx = (int)(global_warp - (size_t)token_id * groups_per_token);
    const int h_base    = group_idx * OUTS;

    const int half_k = inter / 2;

    // Accumulators (one per output h in this warp's group).
    float acc[OUTS];
    #pragma unroll
    for (int j = 0; j < OUTS; ++j) acc[j] = 0.f;

    #pragma unroll 1
    for (int s = 0; s < topk; ++s) {
        const int   expert_id = topk_ids  [token_id * topk + s];
        const float scale     = topk_scale[token_id * topk + s];

        const __nv_bfloat162* __restrict__ in_ptr =
            reinterpret_cast<const __nv_bfloat162*>(
                buffer + ((size_t)token_id * topk + s) * inter);

        // Precompute weight-row pointers for the OUTS outputs.
        const __nv_bfloat16* w_base =
            W + (size_t)expert_id * hidden * inter + (size_t)h_base * inter;
        const __nv_bfloat162* w_ptrs[OUTS];
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) {
            w_ptrs[j] = reinterpret_cast<const __nv_bfloat162*>(w_base + j * inter);
        }

        // Per-expert local accumulators (allow the compiler to overlap loads).
        float local[OUTS];
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) local[j] = 0.f;

        // 32 lanes cover half_k in (half_k/32) iters. For Qwen3 half_k=96 -> 3 iters.
        #pragma unroll
        for (int i = lane_id; i < half_k; i += WARP_SIZE_DOWN) {
            // Widen before multiplication to avoid rounding each product to BF16.
            const float2 xf = widen_bf16_pair_d<STATIC_INTER>(in_ptr[i]);
            #pragma unroll
            for (int j = 0; j < OUTS; ++j) {
                const float2 wf = widen_bf16_pair_d<STATIC_INTER>(w_ptrs[j][i]);
                local[j] = fmaf(xf.x, wf.x, local[j]);
                local[j] = fmaf(xf.y, wf.y, local[j]);
            }
        }

        #pragma unroll
        for (int j = 0; j < OUTS; ++j) acc[j] += scale * local[j];
    }

    #pragma unroll
    for (int j = 0; j < OUTS; ++j) acc[j] = warp_reduce_sum_d(acc[j]);

    if (lane_id == 0) {
        __nv_bfloat16* out_ptr = out + (size_t)token_id * hidden + h_base;
        if constexpr (STATIC_INTER == 192 && OUTS == 2) {
            // Each group starts at an even BF16 index, so this store is aligned.
            *reinterpret_cast<__nv_bfloat162*>(out_ptr) =
                __floats2bfloat162_rn(acc[0], acc[1]);
        } else if constexpr (STATIC_INTER == 192 && OUTS == 8) {
            const __nv_bfloat162_raw pairs[4] = {
                static_cast<__nv_bfloat162_raw>(__floats2bfloat162_rn(acc[0], acc[1])),
                static_cast<__nv_bfloat162_raw>(__floats2bfloat162_rn(acc[2], acc[3])),
                static_cast<__nv_bfloat162_raw>(__floats2bfloat162_rn(acc[4], acc[5])),
                static_cast<__nv_bfloat162_raw>(__floats2bfloat162_rn(acc[6], acc[7]))
            };
            uint4 packed;
            static_assert(sizeof(pairs) == sizeof(packed), "Eight BF16 outputs occupy 16 bytes");
            memcpy(&packed, pairs, sizeof(packed));
            *reinterpret_cast<uint4*>(out_ptr) = packed;
        } else {
            #pragma unroll
            for (int j = 0; j < OUTS; ++j) {
                out_ptr[j] = __float2bfloat16(acc[j]);
            }
        }
    }
}

void launch_moe_down_3d_batched(
    const __nv_bfloat16* buffer,
    const __nv_bfloat16* W,
    const int*           topk_ids,
    const float*         topk_scale,
    __nv_bfloat16*       out,
    int B, int E, int topk, int hidden, int inter,
    cudaStream_t stream)
{
    const size_t total_warps = (size_t)B * (hidden / OUTS_PER_WARP);
    if (B > 0 && B <= 16 && inter == 192) {
        if (B <= 2) {
            constexpr int SMALL_BATCH_OUTS = 2;
            constexpr int SMALL_BATCH_WARPS = 4;
            const size_t small_total_warps = (size_t)B * (hidden / SMALL_BATCH_OUTS);
            const size_t small_ctas =
                (small_total_warps + SMALL_BATCH_WARPS - 1) / SMALL_BATCH_WARPS;
            moe_down_3d_batched_kernel<SMALL_BATCH_OUTS, 192>
                <<<dim3((unsigned)small_ctas), dim3(SMALL_BATCH_WARPS * WARP_SIZE_DOWN), 0, stream>>>(
                    buffer, W, topk_ids, topk_scale, out, B, E, topk, hidden, inter);
        } else {
            constexpr int SMALL_BATCH_WARPS = 2;
            const size_t small_ctas =
                (total_warps + SMALL_BATCH_WARPS - 1) / SMALL_BATCH_WARPS;
            moe_down_3d_batched_kernel<OUTS_PER_WARP, 192>
                <<<dim3((unsigned)small_ctas), dim3(SMALL_BATCH_WARPS * WARP_SIZE_DOWN), 0, stream>>>(
                    buffer, W, topk_ids, topk_scale, out, B, E, topk, hidden, inter);
        }
        return;
    }

    const size_t ctas = (total_warps + WARPS_PER_CTA_DOWN - 1) / WARPS_PER_CTA_DOWN;

    dim3 grid((unsigned)ctas);
    dim3 block(THREADS_PER_CTA_DOWN);

    moe_down_3d_batched_kernel<OUTS_PER_WARP><<<grid, block, 0, stream>>>(
        buffer, W, topk_ids, topk_scale, out, B, E, topk, hidden, inter);
}

}  // namespace warp_decode
