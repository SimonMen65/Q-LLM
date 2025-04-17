from torch.utils.cpp_extension import load
import torch
import os

# Define CUDA kernel code
cuda_code = """
#include <float.h>

extern "C" __global__ void dot_topk_kernel(
    const float* __restrict__ x, // [N, D]
    const float* __restrict__ y, // [D]
    float* __restrict__ topk_scores,  // [K]
    int* __restrict__ topk_indices,   // [K]
    int N,
    int D,
    int K
) {
    extern __shared__ float shared_data[];
    float* scores = shared_data;
    int* indices = (int*)&scores[N];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float sum = 0.0f;
    for (int j = 0; j < D; ++j) {
        sum += x[idx * D + j] * y[j];
    }
    scores[idx] = sum;
    indices[idx] = idx;

    __syncthreads();

    // Only thread 0 does the top-k selection (for small N)
    if (threadIdx.x == 0) {
        for (int i = 0; i < K; ++i) {
            int max_idx = i;
            for (int j = i + 1; j < N; ++j) {
                if (scores[j] > scores[max_idx]) {
                    max_idx = j;
                }
            }
            float tmp_score = scores[i];
            int tmp_index = indices[i];
            scores[i] = scores[max_idx];
            indices[i] = indices[max_idx];
            scores[max_idx] = tmp_score;
            indices[max_idx] = tmp_index;

            topk_scores[i] = scores[i];
            topk_indices[i] = indices[i];
        }
    }
}
"""

# Write CUDA code to file
cuda_filename = os.path.join(os.path.dirname(__file__), "dot_topk_kernel.cu")
with open(cuda_filename, "w") as f:
    f.write(cuda_code)

# Load CUDA kernel module
cuda_module = load(
    name="dot_topk_kernel",
    sources=[cuda_filename],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

def dot_topk_cuda(x: torch.Tensor, y: torch.Tensor, topk: int):
    assert x.is_cuda and y.is_cuda
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    N, D = x.shape
    assert y.shape[0] == D
    assert topk <= N

    topk_scores = torch.empty((topk,), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((topk,), dtype=torch.int32, device="cuda")

    threads = 1024
    blocks = 1
    shared_mem_size = N * (4 + 4)  # float32 + int32

    cuda_module.dot_topk_kernel(
        x.contiguous(), y.contiguous(), topk_scores, topk_indices,
        N, D, topk,
        grid=(blocks,), block=(threads,), shared_mem=shared_mem_size
    )

    return topk_indices, topk_scores

// 封装 kernel 调用
void dot_topk_kernel_launcher(const at::Tensor& x, const at::Tensor& y,
                              at::Tensor& topk_scores, at::Tensor& topk_indices,
                              int N, int D, int K) {
    int threads = 1024;
    int blocks = 1;
    size_t shared_mem_size = N * (sizeof(float) + sizeof(int));
    dot_topk_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        topk_scores.data_ptr<float>(), topk_indices.data_ptr<int>(),
        N, D, K);
}
