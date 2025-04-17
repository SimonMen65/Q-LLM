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

    // Simple selection sort for top-k on a single block (only valid for small N)
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