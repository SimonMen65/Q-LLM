#include <torch/extension.h>

// 声明 CUDA 函数（你 .cu 文件中实现的）
void dot_topk_kernel_launcher(const at::Tensor& x, const at::Tensor& y,
                              at::Tensor& topk_scores, at::Tensor& topk_indices,
                              int N, int D, int K);

// Python 绑定函数
void dot_topk_launcher(const at::Tensor& x, const at::Tensor& y,
                       at::Tensor& topk_scores, at::Tensor& topk_indices, int K) {
    int N = x.size(0);
    int D = x.size(1);
    dot_topk_kernel_launcher(x, y, topk_scores, topk_indices, N, D, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_topk_launcher", &dot_topk_launcher, "Dot TopK launcher");
}
