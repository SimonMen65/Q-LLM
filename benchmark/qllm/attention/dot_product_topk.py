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
    @staticmethod
    def forward(ctx, data, query_c, query_q, question_weight, topk):
        orig_dtype = data.dtype
        
        # 自动转换BFloat16到Float32进行计算
        if orig_dtype == torch.bfloat16:
            data = data.float()
            query_c = query_c.float()
            query_q = query_q.float() if query_q is not None else None
        
        indices, values = dot_product_topk.dot_product_topk(
            data, query_c, query_q, question_weight, topk
        )
        
        # 转换回原始类型
        if orig_dtype == torch.bfloat16:
            return indices, values.bfloat16()
        return indices, values
    
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