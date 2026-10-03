// Streaming kernels that measure the memory system: read-only (what LLM decode does with weights),
// copy (read + write) and write-only. Each uses 16-byte loads and a grid-stride loop.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace {

__global__ void read_kernel(const float4* __restrict__ x, size_t n4, float* __restrict__ out) {
    float acc = 0.0f;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n4;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float4 v = __ldcs(x + i);   // streaming hint: the data is used once, do not keep it in L2
        acc += (v.x + v.y) + (v.z + v.w);
    }
    // The sum keeps the loads alive; one atomic per warp is negligible next to the traffic.
    for (int m = 16; m; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);
}

__global__ void copy_kernel(const float4* __restrict__ x, float4* __restrict__ y, size_t n4) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n4;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        __stcs(y + i, __ldcs(x + i));
}

__global__ void write_kernel(float4* __restrict__ y, size_t n4, float value) {
    const float4 v = make_float4(value, value, value, value);
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n4;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        __stcs(y + i, v);
}

void check(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
                "expects a contiguous float32 CUDA tensor");
    TORCH_CHECK(t.numel() % 4 == 0, "numel must be a multiple of 4");
}

}  // namespace

torch::Tensor stream_read(torch::Tensor x, int64_t blocks, int64_t threads) {
    check(x);
    auto out = torch::zeros({1}, x.options());
    read_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()), x.numel() / 4, out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void stream_copy(torch::Tensor x, torch::Tensor y, int64_t blocks, int64_t threads) {
    check(x);
    check(y);
    TORCH_CHECK(x.numel() == y.numel(), "x and y must have the same size");
    copy_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()), reinterpret_cast<float4*>(y.data_ptr<float>()),
        x.numel() / 4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void stream_write(torch::Tensor y, double value, int64_t blocks, int64_t threads) {
    check(y);
    write_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<float4*>(y.data_ptr<float>()), y.numel() / 4, static_cast<float>(value));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read", &stream_read, "read-only stream");
    m.def("copy", &stream_copy, "copy stream");
    m.def("write", &stream_write, "write-only stream");
}
