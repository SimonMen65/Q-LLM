#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int ITEMS_PER_THREAD = 4;

template<>
__global__ void batched_dot_product_topk_kernel<at::BFloat16>(
    const at::BFloat16* __restrict__ data,
    const at::BFloat16* __restrict__ query_c,
    const at::BFloat16* __restrict__ query_q,
    float question_weight,
    int64_t* __restrict__ topk_indices,
    at::BFloat16* __restrict__ topk_values,
    int num_units,
    int hidden_size,
    int data_length,
    int topk
) {
    extern __shared__ float shared_mem[];
    
    const int unit_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int total_heads = gridDim.y;
    
    if (unit_id >= num_units) return;
    
    float* s_query_c = shared_mem;
    float* s_query_q = s_query_c + hidden_size;

    // 加载查询到共享内存（转换为float）
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        s_query_c[i] = static_cast<float>(
            query_c[unit_id * total_heads * hidden_size + head_id * hidden_size + i]
        );
        if (query_q != nullptr) {
            s_query_q[i] = static_cast<float>(
                query_q[unit_id * total_heads * hidden_size + head_id * hidden_size + i]
            );
        }
    }
    __syncthreads();

    float local_scores[ITEMS_PER_THREAD];
    int local_indices[ITEMS_PER_THREAD];
    
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int data_idx = threadIdx.x + item * blockDim.x;
        if (data_idx >= data_length) {
            local_scores[item] = -INFINITY;
            continue;
        }
        
        float sum_c = 0, sum_q = 0;
        const at::BFloat16* data_ptr = data + (unit_id * data_length + data_idx) * hidden_size;
        
        #pragma unroll
        for (int i = 0; i < hidden_size; ++i) {
            const float data_val = static_cast<float>(data_ptr[i]);
            sum_c += data_val * s_query_c[i];
            if (query_q != nullptr) {
                sum_q += data_val * s_query_q[i];
            }
        }
        
        local_scores[item] = sum_c + (query_q ? question_weight * sum_q : 0);
        local_indices[item] = data_idx;
    }

    typedef cub::BlockReduce<cub::KeyValuePair<int, float>, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        cub::KeyValuePair<int, float> thread_data(
            local_indices[item], local_scores[item]
        );
        
        cub::KeyValuePair<int, float> block_max = BlockReduce(temp_storage).Reduce(
            thread_data,
            [](const cub::KeyValuePair<int, float>& a, 
               const cub::KeyValuePair<int, float>& b) {
                return (a.value > b.value) ? a : b;
            }
        );
        
        if (threadIdx.x == 0) {
            const int output_idx = unit_id * topk + head_id * topk * num_units;
            for (int k = 0; k < topk; ++k) {
                if (block_max.value > static_cast<float>(topk_values[output_idx + k])) {
                    for (int m = topk-1; m > k; --m) {
                        topk_values[output_idx + m] = topk_values[output_idx + m-1];
                        topk_indices[output_idx + m] = topk_indices[output_idx + m-1];
                    }
                    topk_values[output_idx + k] = static_cast<at::BFloat16>(block_max.value);
                    topk_indices[output_idx + k] = block_max.key;
                    break;
                }
            }
        }
    }
}

std::pair<torch::Tensor, torch::Tensor> dot_product_topk_cuda(
    torch::Tensor data,
    torch::Tensor query_c,
    c10::optional<torch::Tensor> query_q,
    float question_weight,
    int topk
) {
    TORCH_CHECK(data.dim() == 2, "Data must be 2D tensor");
    TORCH_CHECK(query_c.dim() == 2, "Query must be 2D tensor");
    
    const int num_units = query_c.size(0);
    const int num_heads = query_c.size(1);
    const int hidden_size = data.size(1);
    const int data_length = data.size(0) / num_units;
    
    auto options = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(data.device());
    auto topk_indices = torch::full({num_units, num_heads, topk}, -1, options);
    
    options = options.dtype(data.dtype());
    auto topk_values = torch::full({num_units, num_heads, topk}, 
                                 -std::numeric_limits<float>::infinity(), options);
    
    const dim3 blocks(num_units, num_heads);

    // 关键修改：将共享内存计算移入类型分派宏内部
    AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, 
    at::ScalarType::BFloat16,
    data.scalar_type(), "dot_product_topk", ([&] {
        // 现在scalar_t在此作用域内有效
        const int smem_size = 2 * hidden_size * sizeof(scalar_t);
        
        auto data_ptr = data.data_ptr<scalar_t>();
        auto query_c_ptr = query_c.data_ptr<scalar_t>();
        auto query_q_ptr = query_q.has_value() ? query_q->data_ptr<scalar_t>() : nullptr;
        
        batched_dot_product_topk_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK, smem_size>>>(
            data_ptr,
            query_c_ptr,
            query_q_ptr,
            question_weight,
            topk_indices.data_ptr<int64_t>(),
            topk_values.data_ptr<scalar_t>(),
            num_units,
            hidden_size,
            data_length,
            topk
        );
    }));
    
    return {topk_indices, topk_values};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_product_topk", &dot_product_topk_cuda, "Batched dot product topk");
}