import torch
from torch.utils.cpp_extension import load

dot_product_topk = load(
    name="dot_product_topk",
    sources=["/home/smen/Q-LLM/benchmark/qllm/attention/dot_product_topk.cu"],
    extra_cflags=["-O3"],
    verbose=True
)

class DotProductTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, query_c, query_q, question_weight, topk):
        ctx.save_for_backward(data, query_c, query_q)
        ctx.question_weight = question_weight
        ctx.topk = topk
        return dot_product_topk_cuda(data, query_c, query_q, question_weight, topk)

    @staticmethod
    def backward(ctx, grad_indices, grad_values):
        # 如需支持反向传播需实现，此处暂略
        return None, None, None, None, None

def dot_product_topk_wrapper(data, query_c, query_q=None, question_weight=0.5, topk=1):
    return DotProductTopKFunction.apply(data, query_c, query_q, question_weight, topk)