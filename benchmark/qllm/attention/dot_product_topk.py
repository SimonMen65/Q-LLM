import torch
from torch.utils.cpp_extension import load

dot_product_topk_module = load(
    name="dot_product_topk",
    sources=["/home/smen/Q-LLM/benchmark/qllm/attention/dot_product_topk.cu"],
    extra_cflags=["-O3"],
    verbose=True
)

class DotProductTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, query_c, query_q, question_weight, topk):
        assert data.dim() == 2, f"Data must be 2D [length, hidden], got {data.shape}"  # data is 2D
        assert query_c.dim() == 2, f"Query must be 2D [batch, hidden], got {query_c.shape}"  # query_c is 2D

        # 保存原始类型
        orig_dtype = data.dtype
        
        # 自动转换BFloat16到Float32
        if orig_dtype == torch.bfloat16:
            data = data.to(torch.float32)
            query_c = query_c.to(torch.float32)
            if query_q is not None:
                query_q = query_q.to(torch.float32)

        # 确保内存连续
        data = data.contiguous()
        query_c = query_c.contiguous()
        query_q = query_q.contiguous() if query_q is not None else None

        return [], []

        # 调用CUDA扩展
        indices, values = dot_product_topk_module.dot_product_topk(
            data, query_c, query_q, question_weight, topk
        )

        # 转换回原始类型
        if orig_dtype == torch.bfloat16:
            values = values.to(orig_dtype)
        
        ctx.save_for_backward(data, query_c, query_q)
        ctx.question_weight = question_weight
        ctx.topk = topk
        ctx.orig_dtype = orig_dtype
        
        return indices, values

    @staticmethod
    def backward(ctx, grad_indices, grad_values):
        # 反向传播暂不实现
        return None, None, None, None, None

def dot_product_topk_wrapper(data, query_c, query_q=None, question_weight=0.5, topk=1):
    return DotProductTopKFunction.apply(
        data, 
        query_c,
        query_q,
        question_weight,
        topk
    )