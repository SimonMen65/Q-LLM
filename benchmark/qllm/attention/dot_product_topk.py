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
        
        # 修改2：通过模块调用具体函数
        return dot_product_topk_module.dot_product_topk(
            data, 
            query_c,
            query_q if query_q is not None else None,  # 处理None值
            question_weight,
            topk
        )

    @staticmethod
    def backward(ctx, grad_indices, grad_values):
        return None, None, None, None, None

# 修改3：添加参数校验
def dot_product_topk_wrapper(data, query_c, query_q=None, question_weight=0.5, topk=1):
    assert data.dim() == 2, f"Data must be 2D tensor, got {data.shape}"
    assert query_c.dim() == 2, f"Query must be 2D tensor, got {query_c.shape}"
    if query_q is not None:
        assert query_q.dim() == 2, f"Query_q must be 2D tensor, got {query_q.shape}"
    
    return DotProductTopKFunction.apply(
        data.contiguous(),
        query_c.contiguous(),
        query_q.contiguous() if query_q is not None else None,
        question_weight,
        topk
    )