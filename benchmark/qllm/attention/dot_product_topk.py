import torch
from torch.utils.cpp_extension import load

dot_product_topk = load(
    name="dot_product_topk",
    sources=["/home/smen/Q-LLM/benchmark/qllm/attention/dot_product_topk.cu"],
    extra_cflags=["-O3"],
    verbose=True,
    with_cuda=True
)

class DotProductTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, query_c, query_q, question_weight, topk):
        # 通过模块调用C++函数
        return dot_product_topk.dot_product_topk(  # 注意这里改为通过模块调用
            data, 
            query_c,
            query_q,
            question_weight,
            topk
        )
    
    @staticmethod
    def backward(ctx, grad_indices, grad_values):
        return None, None, None, None, None

def dot_product_topk_wrapper(data, query_c, query_q=None, question_weight=0.5, topk=1):
    return DotProductTopKFunction.apply(
        data.contiguous(), 
        query_c.contiguous(),
        query_q.contiguous() if query_q is not None else None,
        question_weight,
        topk
    )