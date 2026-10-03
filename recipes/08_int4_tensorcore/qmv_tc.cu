// Tensor-core 4-bit matmul for decode and speculative verify windows: out (M, N) = x (M, K) @ dequant(W).T
//
// What changes from recipe 03's warp GEMV:
//   1. Weights are repacked at load (tc_pack.py) into mma B-fragment order: one 16-byte streaming load per
//      lane per 128-k chunk, and a nibble pair turns into a bf16 register with two integer ops.
//   2. The multiply runs on tensor cores (mma.sync m16n8k16, bf16 in, fp32 out), 16 rows per m tile, so
//      verifying 16 draft rows costs about what one row costs: the weight stream is the same.
//   3. Dequantization is folded per group: acc = fma(sum_x, bias, fma(dot, scale, acc)), with dot the raw
//      code-times-x product from the tensor core and sum_x precomputed per row and group.
//   4. Small layers split K into a shape-only number of slices (split_k in tc_pack.py), and partials are
//      summed in slice order, so every row gets the same bits at any row count.
//   5. Several projections of one input (Q/K/V, gate/up) can be packed as one weight: one launch.
//
// One warp owns NT adjacent 8-column tiles and one K slice; a block holds 4 independent warps.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int WARPS = 4;
constexpr int K_CHUNK = 128;

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Codes at bits [4p, 4p+4) and [16+4p, 20+4p) as a bf16 pair: as bf16 bits, 0x4300 | q is 128 + q
// exactly (q < 16 fits the 7-bit mantissa), and subtracting 128 leaves q exactly.
__device__ __forceinline__ uint32_t pair(uint32_t w, int p) {
    uint32_t v = ((w >> (4 * p)) & 0x000F000Fu) | 0x43004300u;
    __nv_bfloat162 h = *reinterpret_cast<__nv_bfloat162*>(&v);
    h = __hsub2(h, __floats2bfloat162_rn(128.0f, 128.0f));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t word_of(const uint4& u, int v) {
    return v == 0 ? u.x : v == 1 ? u.y : v == 2 ? u.z : u.w;     // v is a constant after unrolling
}

__device__ __forceinline__ uint32_t ld_pair(const __nv_bfloat16* p) {
    return __ldg(reinterpret_cast<const unsigned int*>(p));
}

template <int GS, int NT>
__global__ void __launch_bounds__(WARPS * 32) qmv_tc_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint4* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
        __nv_bfloat16* __restrict__ out, float* __restrict__ part, int M, int N, int npad, int K, int SK) {
    constexpr int GPC = K_CHUNK / GS;          // groups per chunk
    constexpr int SPG = GS / 16;               // k16 steps per group
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int ngroups = npad / (8 * NT);
    const int item = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (item >= ngroups * SK) return;          // warp-uniform
    const int ng = item % ngroups, slice = item / ngroups;
    const int KC = K / K_CHUNK, per = KC / SK, cbeg = slice * per, cend = cbeg + per;
    const int KG = K / GS;
    const int T0 = ng * NT;

    // This lane's two A rows; rows past M are fed zeros and never stored.
    const int r0 = blockIdx.y * 16 + g, r1 = r0 + 8;
    const bool ok0 = r0 < M, ok1 = r1 < M;
    const __nv_bfloat16* x0 = x + static_cast<size_t>(ok0 ? r0 : 0) * K;
    const __nv_bfloat16* x1 = x + static_cast<size_t>(ok1 ? r1 : 0) * K;

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

    uint4 wcur[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) wcur[j] = __ldcs(w + (static_cast<size_t>(T0 + j) * KC + cbeg) * 32 + lane);

    for (int c = cbeg; c < cend; ++c) {
        // Issue the next chunk's weight loads before this chunk's math, so they are in flight meanwhile.
        const bool more = c + 1 < cend;
        uint4 wnext[NT];
        if (more) {
#pragma unroll
            for (int j = 0; j < NT; ++j)
                wnext[j] = __ldcs(w + (static_cast<size_t>(T0 + j) * KC + c + 1) * 32 + lane);
        }
#pragma unroll
        for (int gi = 0; gi < GPC; ++gi) {
            const int grp = c * GPC + gi;
            float d[NT][4];
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) d[j][e] = 0.0f;
#pragma unroll
            for (int st = 0; st < SPG; ++st) {
                const int step = gi * SPG + st;                    // k16 step within the chunk, 0..7
                const int k = c * K_CHUNK + step * 16 + 2 * t;
                uint32_t a[4];
                a[0] = ok0 ? ld_pair(x0 + k) : 0u;
                a[1] = ok1 ? ld_pair(x1 + k) : 0u;
                a[2] = ok0 ? ld_pair(x0 + k + 8) : 0u;
                a[3] = ok1 ? ld_pair(x1 + k + 8) : 0u;
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const uint32_t word = word_of(wcur[j], step >> 1);
                    const int p = (step & 1) * 2;
                    mma_bf16(d[j], a, pair(word, p), pair(word, p + 1));
                }
            }
            const float xs0 = ok0 ? xs[static_cast<size_t>(r0) * KG + grp] : 0.0f;
            const float xs1 = ok1 ? xs[static_cast<size_t>(r1) * KG + grp] : 0.0f;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int n = (T0 + j) * 8 + 2 * t;                // the output fragment's two columns
                const float2 s = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(scales + static_cast<size_t>(grp) * npad + n));
                const float2 b = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(biases + static_cast<size_t>(grp) * npad + n));
                acc[j][0] = __fmaf_rn(xs0, b.x, __fmaf_rn(d[j][0], s.x, acc[j][0]));
                acc[j][1] = __fmaf_rn(xs0, b.y, __fmaf_rn(d[j][1], s.y, acc[j][1]));
                acc[j][2] = __fmaf_rn(xs1, b.x, __fmaf_rn(d[j][2], s.x, acc[j][2]));
                acc[j][3] = __fmaf_rn(xs1, b.y, __fmaf_rn(d[j][3], s.y, acc[j][3]));
            }
        }
        if (more) {
#pragma unroll
            for (int j = 0; j < NT; ++j) wcur[j] = wnext[j];
        }
    }

#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = e < 2 ? r0 : r1;
            const int n = (T0 + j) * 8 + 2 * t + (e & 1);
            if (row >= M) continue;
            if (SK == 1) {
                if (n < N) out[static_cast<size_t>(row) * N + n] = __float2bfloat16(acc[j][e]);
            } else {
                part[(static_cast<size_t>(slice) * M + row) * npad + n] = acc[j][e];
            }
        }
    }
}

// K-slice partials summed in slice order (never atomics), so the result does not depend on scheduling.
__global__ void reduce_kernel(const float* __restrict__ part, __nv_bfloat16* __restrict__ out, int M, int N,
                              int npad, int SK) {
    const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= static_cast<size_t>(M) * N) return;
    const int m = idx / N, n = idx % N;
    float s = 0.0f;
    for (int k = 0; k < SK; ++k) s += part[(static_cast<size_t>(k) * M + m) * npad + n];
    out[idx] = __float2bfloat16(s);
}

// Per row and group, sum(x) in a fixed sequential order: the bias term of the folded dequantization.
__global__ void group_sums_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ xs, int M, int K,
                                  int gs) {
    const int KG = K / gs;
    const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= static_cast<size_t>(M) * KG) return;
    const int m = idx / KG, grp = idx % KG;
    const __nv_bfloat16* p = x + static_cast<size_t>(m) * K + static_cast<size_t>(grp) * gs;
    float s = 0.0f;
    for (int i = 0; i < gs; ++i) s += __bfloat162float(p[i]);
    xs[idx] = s;
}

template <int GS, int NT>
void launch(const __nv_bfloat16* x, const float* xs, const uint4* w, const __nv_bfloat16* s, const __nv_bfloat16* b,
            __nv_bfloat16* out, float* part, int M, int N, int npad, int K, int SK, cudaStream_t stream) {
    const int items = npad / (8 * NT) * SK;
    dim3 grid((items + WARPS - 1) / WARPS, (M + 15) / 16);
    qmv_tc_kernel<GS, NT><<<grid, WARPS * 32, 0, stream>>>(x, xs, w, s, b, out, part, M, N, npad, K, SK);
}

#define TC_CASE(GS_, NT_)                                                                                    \
    if (gs == GS_ && nt == NT_) {                                                                           \
        launch<GS_, NT_>(xp, xsp, wp, sp, bp, op, pp, M, N, npad, K, SK, stream);                           \
        launched = true;                                                                                     \
    }

}  // namespace

torch::Tensor group_sums(torch::Tensor x, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 2,
                "x must be a contiguous 2-D bfloat16 CUDA tensor");
    TORCH_CHECK(x.size(1) % gs == 0, "K must be a multiple of the group size");
    const int M = x.size(0), K = x.size(1);
    auto xs = torch::empty({M, K / gs}, x.options().dtype(torch::kFloat32));
    const size_t total = static_cast<size_t>(M) * (K / gs);
    group_sums_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), xs.data_ptr<float>(), M, K, gs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return xs;
}

torch::Tensor qmv_tc(torch::Tensor x, torch::Tensor words, torch::Tensor scales, torch::Tensor biases, int64_t n,
                     int64_t gs, int64_t sk, int64_t nt) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 2,
                "x must be a contiguous 2-D bfloat16 CUDA tensor");
    TORCH_CHECK(words.is_cuda() && words.scalar_type() == torch::kInt32 && words.is_contiguous() &&
                words.dim() == 4 && words.size(2) == 32 && words.size(3) == 4,
                "words must be the (Npad/8, K/128, 32, 4) int32 layout from tc_pack.pack");
    TORCH_CHECK(scales.scalar_type() == torch::kBFloat16 && biases.scalar_type() == torch::kBFloat16 &&
                scales.is_contiguous() && biases.is_contiguous() && scales.sizes() == biases.sizes(),
                "scales and biases must be contiguous bfloat16 of equal shape");
    const int M = x.size(0), K = x.size(1), npad = words.size(0) * 8, N = n, SK = sk;
    TORCH_CHECK(words.size(1) * K_CHUNK == K, "K mismatch between x and the packed weight");
    TORCH_CHECK(scales.size(0) == K / gs && scales.size(1) == npad, "scales must be (K/gs, Npad)");
    TORCH_CHECK(N <= npad && npad % (8 * nt) == 0, "N/Npad do not fit the column tiling");
    TORCH_CHECK(SK >= 1 && (K / K_CHUNK) % SK == 0, "sk must divide K/128");

    auto out = torch::empty({M, N}, x.options());
    torch::Tensor partials;
    if (SK > 1) partials = torch::empty({SK, M, npad}, x.options().dtype(torch::kFloat32));
    torch::Tensor xs = group_sums(x, gs);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto xsp = xs.data_ptr<float>();
    auto wp = reinterpret_cast<const uint4*>(words.data_ptr<int32_t>());
    auto sp = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
    auto bp = reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr<at::BFloat16>());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    float* pp = SK > 1 ? partials.data_ptr<float>() : nullptr;

    bool launched = false;
    TC_CASE(32, 2) TC_CASE(32, 4) TC_CASE(64, 2) TC_CASE(64, 4) TC_CASE(128, 2) TC_CASE(128, 4)
    TORCH_CHECK(launched, "unsupported gs/nt: gs in {32, 64, 128}, nt in {2, 4}");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (SK > 1) {
        const size_t total = static_cast<size_t>(M) * N;
        reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(pp, op, M, N, npad, SK);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qmv_tc", &qmv_tc, "tensor-core 4-bit matmul on tc_pack's layout", py::arg("x"), py::arg("words"),
          py::arg("scales"), py::arg("biases"), py::arg("n"), py::arg("gs") = 64, py::arg("sk") = 1,
          py::arg("nt") = 4);
    m.def("group_sums", &group_sums, "per-row, per-group sums of x in fp32", py::arg("x"), py::arg("gs") = 64);
}
