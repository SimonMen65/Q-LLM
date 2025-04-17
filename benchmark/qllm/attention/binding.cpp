#include <torch/extension.h>

// 声明你在 .cu 文件中定义的 launcher
void dot_topk_launcher(const at::Tensor& x, const at::Tensor& y,
                       at::Tensor& topk_scores, at::Tensor& topk_indices, int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_topk_launcher", &dot_topk_launcher, "Dot TopK CUDA kernel launcher");
}