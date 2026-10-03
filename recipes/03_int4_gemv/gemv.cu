// 4-bit affine GEMV for LLM decode: out (M, N) = x (M, K) @ dequant(W (N, K)).T, M small (1..8 rows).
//
// Decode is memory-bound, so the only number that matters is how close the weight stream gets to the
// bandwidth measured in recipe 01. Two versions show why layout decides that:
//
//   naive : one thread per output. Adjacent threads read rows K/2 bytes apart, so every warp load touches
//           32 different cache lines (uncoalesced). Kept as the baseline to profile against.
//   warp  : one warp per output column n. Lanes read consecutive 16-byte chunks of that row (512 B per warp
//           step, fully coalesced), each chunk being 32 codes of one group. The group's dequantization is
//           folded algebraically:  sum_k (q*s + b) * x  =  s * sum(q*x) + b * sum(x)
//           so the inner loop is just q*x FMAs; scale and bias are applied once per 32 codes.
//
// Every row m is computed in the same order whatever M is (no cross-row work), so a row's bits do not
// depend on the batch: the property speculative-decoding verifiers need (see recipe 06).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

__global__ void gemv_naive_kernel(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,
                                  const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
                                  __nv_bfloat16* __restrict__ out, int M, int N, int K, int gs) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;
    const int KG = K / gs;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const uint32_t word = w[static_cast<size_t>(n) * (K / 8) + k / 8];
        const float q = static_cast<float>((word >> (4 * (k % 8))) & 0xFu);
        const int g = k / gs;
        const float wv = q * __bfloat162float(scales[static_cast<size_t>(n) * KG + g]) +
                         __bfloat162float(biases[static_cast<size_t>(n) * KG + g]);
        acc = fmaf(wv, __bfloat162float(x[static_cast<size_t>(m) * K + k]), acc);
    }
    out[static_cast<size_t>(m) * N + n] = __float2bfloat16(acc);
}

template <int M>
__global__ void gemv_warp_kernel(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,
                                 const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
                                 __nv_bfloat16* __restrict__ out, int N, int K, int gs) {
    const int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;                       // whole warps exit together: n is uniform across a warp
    const int KG = K / gs;
    const int chunks = K / 32;                // 16-byte chunks of 32 codes; gs % 32 == 0 keeps one in one group
    const uint4* wrow = reinterpret_cast<const uint4*>(w + static_cast<size_t>(n) * (K / 8));
    const __nv_bfloat16* srow = scales + static_cast<size_t>(n) * KG;
    const __nv_bfloat16* brow = biases + static_cast<size_t>(n) * KG;

    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    for (int c = lane; c < chunks; c += 32) {
        const uint4 wv = __ldcs(wrow + c);    // weights are read once: stream them past L2
        const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
        const int g = (c * 32) / gs;
        const float s = __bfloat162float(srow[g]);
        const float b = __bfloat162float(brow[g]);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            // x is small and shared by every warp, so it stays in L1/L2; plain cached loads are right here.
            const uint4* xv = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K + c * 32);
            float dot = 0.0f, sx = 0.0f;
#pragma unroll
            for (int v = 0; v < 4; ++v) {     // 4 x (8 bf16 inputs <-> the 8 nibbles of words[v])
                const uint4 xx = xv[v];
                const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xx);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(h[j]);
                    const float q0 = static_cast<float>((words[v] >> (8 * j)) & 0xFu);
                    const float q1 = static_cast<float>((words[v] >> (8 * j + 4)) & 0xFu);
                    dot = fmaf(q0, f.x, dot);
                    dot = fmaf(q1, f.y, dot);
                    sx += f.x + f.y;
                }
            }
            acc[m] = fmaf(s, dot, fmaf(b, sx, acc[m]));
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = acc[m];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) out[static_cast<size_t>(m) * N + n] = __float2bfloat16(v);
    }
}

template <int M>
void launch_warp(const __nv_bfloat16* x, const uint32_t* w, const __nv_bfloat16* s, const __nv_bfloat16* b,
                 __nv_bfloat16* out, int N, int K, int gs, int threads, cudaStream_t stream) {
    const int warps = threads / 32;
    gemv_warp_kernel<M><<<(N + warps - 1) / warps, threads, 0, stream>>>(x, w, s, b, out, N, K, gs);
}

void check_inputs(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& b,
                  int64_t gs) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && s.is_cuda() && b.is_cuda(), "all tensors must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && s.scalar_type() == torch::kBFloat16 &&
                b.scalar_type() == torch::kBFloat16, "x, scales and biases must be bfloat16");
    TORCH_CHECK(w.scalar_type() == torch::kInt32, "packed weights must be int32");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && s.is_contiguous() && b.is_contiguous(),
                "all tensors must be contiguous");
    const int64_t K = x.size(1);
    TORCH_CHECK(w.size(1) * 8 == K, "weights must be (N, K/8)");
    TORCH_CHECK(gs % 32 == 0 && K % gs == 0, "group size must be a multiple of 32 dividing K");
    TORCH_CHECK(s.size(0) == w.size(0) && s.size(1) == K / gs && b.sizes() == s.sizes(), "scales/biases shape");
}

}  // namespace

torch::Tensor gemv(torch::Tensor x, torch::Tensor w, torch::Tensor scales, torch::Tensor biases, int64_t gs,
                   std::string variant, int64_t threads) {
    check_inputs(x, w, scales, biases, gs);
    const int M = x.size(0), K = x.size(1), N = w.size(0);
    auto out = torch::empty({M, N}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto wp = reinterpret_cast<const uint32_t*>(w.data_ptr<int32_t>());
    auto sp = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
    auto bp = reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr<at::BFloat16>());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    if (variant == "naive") {
        dim3 grid((N + 255) / 256, M);
        gemv_naive_kernel<<<grid, 256, 0, stream>>>(xp, wp, sp, bp, op, M, N, K, gs);
    } else if (variant == "warp") {
        TORCH_CHECK(threads % 32 == 0 && threads >= 32 && threads <= 1024, "threads must be a multiple of 32");
        switch (M) {
            case 1: launch_warp<1>(xp, wp, sp, bp, op, N, K, gs, threads, stream); break;
            case 2: launch_warp<2>(xp, wp, sp, bp, op, N, K, gs, threads, stream); break;
            case 3: launch_warp<3>(xp, wp, sp, bp, op, N, K, gs, threads, stream); break;
            case 4: launch_warp<4>(xp, wp, sp, bp, op, N, K, gs, threads, stream); break;
            case 5: launch_warp<5>(xp, wp, sp, bp, op, N, K, gs, threads, stream); break;
            case 6: launch_warp<6>(xp, wp, sp, bp, op, N, K, gs, threads, stream); break;
            case 7: launch_warp<7>(xp, wp, sp, bp, op, N, K, gs, threads, stream); break;
            case 8: launch_warp<8>(xp, wp, sp, bp, op, N, K, gs, threads, stream); break;
            default: TORCH_CHECK(false, "the warp GEMV takes 1..8 rows; larger M wants a tensor-core GEMM");
        }
    } else {
        TORCH_CHECK(false, "variant must be 'naive' or 'warp'");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &gemv, "4-bit affine GEMV", py::arg("x"), py::arg("w"), py::arg("scales"), py::arg("biases"),
          py::arg("gs") = 64, py::arg("variant") = "warp", py::arg("threads") = 256);
}
