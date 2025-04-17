#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// 声明 CUDA kernel（你 .cu 文件中写的 __global__ 函数）
__global__ void dot_topk_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ topk_scores,
    int* __restrict__ topk_indices,
    int N, int D, int K
);

// 真正的绑定函数
void dot_topk_launcher(
    at::Tensor x, at::Tensor y,
    at::Tensor topk_scores, at::Tensor topk_indices,
    int K
) {
    int N = x.size(0);
    int D = x.size(1);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem_size = threads * (sizeof(float) + sizeof(int));

    dot_topk_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        topk_scores.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        N, D, K
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_topk_launcher", &dot_topk_launcher, "Dot TopK CUDA kernel launcher");
}
