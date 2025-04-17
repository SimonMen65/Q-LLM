#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void dot_topk_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ topk_scores,
    int* __restrict__ topk_indices,
    int N, int D, int K
) {
    extern __shared__ float shared[];
    float* scores = shared;
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

void dot_topk_launcher(const at::Tensor& x, const at::Tensor& y,
                       at::Tensor& topk_scores, at::Tensor& topk_indices, int K) {
    int N = x.size(0);
    int D = x.size(1);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem = N * (sizeof(float) + sizeof(int));  // be cautious with size

    dot_topk_kernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        topk_scores.data_ptr<float>(), topk_indices.data_ptr<int>(),
        N, D, K
    );
}
