#include <torch/extension.h>

void dot_topk_launcher(
    const torch::Tensor& x,
    const torch::Tensor& y,
    torch::Tensor& topk_scores,
    torch::Tensor& topk_indices,
    int K
);

void dot_topk_launcher(
    const torch::Tensor& x,
    const torch::Tensor& y,
    torch::Tensor& topk_scores,
    torch::Tensor& topk_indices,
    int K
) {
    int N = x.size(0);
    int D = x.size(1);
    int threads = 1024;
    int blocks = 1;
    int shared_mem = N * (sizeof(float) + sizeof(int));

    dot_topk_kernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        topk_scores.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        N, D, K
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_topk_launcher", &dot_topk_launcher, "Dot + TopK Kernel");
}
