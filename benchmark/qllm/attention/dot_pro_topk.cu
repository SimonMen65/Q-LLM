#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA kernel
__global__ void dot_topk_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ scores,
    int* __restrict__ indices,
    int N, int D
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float sum = 0.0f;
    for (int j = 0; j < D; ++j) {
        sum += x[idx * D + j] * y[j];
    }

    scores[idx] = sum;
    indices[idx] = idx;
}

// launcher (called from binding.cpp)
void dot_topk_kernel_launcher(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& scores,
    at::Tensor& indices,
    int N, int D, int K
) {
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    dot_topk_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        scores.data_ptr<float>(),
        indices.data_ptr<int>(),
        N, D
    );
}
