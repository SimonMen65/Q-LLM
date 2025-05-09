#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int ITEMS_PER_THREAD = 4;

// template<typename scalar_t>
// __global__ void batched_dot_product_topk_kernel(
//     const scalar_t* __restrict__ data,
//     const scalar_t* __restrict__ query_c,
//     const scalar_t* __restrict__ query_q,
//     float question_weight,
//     int64_t* __restrict__ topk_indices,
//     scalar_t* __restrict__ topk_values,
//     int num_units,
//     int hidden_size,
//     int data_length,
//     int topk
// ) {
//     extern __shared__ char shared_mem[];
    
//     const int unit_id = blockIdx.x;
//     const int head_id = blockIdx.y;
//     const int total_heads = gridDim.y;
    
//     if (unit_id >= num_units) return;
    
//     scalar_t* s_query_c = reinterpret_cast<scalar_t*>(shared_mem);
//     scalar_t* s_query_q = s_query_c + hidden_size;
    
//     for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
//         s_query_c[i] = query_c[unit_id * total_heads * hidden_size + head_id * hidden_size + i];
//         if (query_q != nullptr) {
//             s_query_q[i] = query_q[unit_id * total_heads * hidden_size + head_id * hidden_size + i];
//         }
//     }
//     __syncthreads();
    

//     scalar_t local_scores[ITEMS_PER_THREAD];
//     int local_indices[ITEMS_PER_THREAD];
    
//     for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
//         const int data_idx = threadIdx.x + item * blockDim.x;
//         if (data_idx >= data_length) {
//             local_scores[item] = -INFINITY;
//             continue;
//         }
        
//         scalar_t sum_c = 0, sum_q = 0;
//         const scalar_t* data_ptr = data + (unit_id * data_length + data_idx) * hidden_size;
        
//         #pragma unroll
//         for (int i = 0; i < hidden_size; ++i) {
//             sum_c += data_ptr[i] * s_query_c[i];
//             if (query_q != nullptr) {
//                 sum_q += data_ptr[i] * s_query_q[i];
//             }
//         }
        
//         local_scores[item] = sum_c + (query_q ? question_weight * sum_q : 0);
//         local_indices[item] = data_idx;
//     }
    

//     typedef cub::BlockReduce<cub::KeyValuePair<int, scalar_t>, THREADS_PER_BLOCK> BlockReduce;
//     __shared__ typename BlockReduce::TempStorage temp_storage;
    
//     for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
//         cub::KeyValuePair<int, scalar_t> thread_data(
//             local_indices[item], local_scores[item]
//         );
        
//         cub::KeyValuePair<int, scalar_t> block_max = BlockReduce(temp_storage).Reduce(
//             thread_data,
//             [](const cub::KeyValuePair<int, scalar_t>& a, 
//                const cub::KeyValuePair<int, scalar_t>& b) {
//                 return (a.value > b.value) ? a : b;
//             }
//         );
        
//         if (threadIdx.x == 0) {
//             const int output_idx = unit_id * topk + head_id * topk * num_units;
//             for (int k = 0; k < topk; ++k) {
//                 if (block_max.value > topk_values[output_idx + k]) {
//                     for (int m = topk-1; m > k; --m) {
//                         topk_values[output_idx + m] = topk_values[output_idx + m-1];
//                         topk_indices[output_idx + m] = topk_indices[output_idx + m-1];
//                     }
//                     topk_values[output_idx + k] = block_max.value;
//                     topk_indices[output_idx + k] = block_max.key;
//                     break;
//                 }
//             }
//         }
//     }
// }

// } // namespace

// std::pair<torch::Tensor, torch::Tensor> dot_product_topk_cuda(
//     torch::Tensor data,
//     torch::Tensor query_c,
//     c10::optional<torch::Tensor> query_q,
//     float question_weight,
//     int topk
// ) {

//     TORCH_CHECK(data.dim() == 2, 
//         "Data must be 2D tensor, but got ", data.dim(), "D tensor instead");
//     TORCH_CHECK(query_c.dim() == 2, 
//         "Query must be 2D tensor, but got ", query_c.dim(), "D tensor instead");
    
//     const int num_units = query_c.size(0);
//     const int num_heads = query_c.size(1);
//     const int hidden_size = data.size(1);
//     const int data_length = data.size(0) / num_units;
    
//     auto options = torch::TensorOptions()
//         .dtype(torch::kInt64)
//         .device(data.device());
//     auto topk_indices = torch::full({num_units, topk}, -1, options);
    
//     options = options.dtype(data.dtype());
//     auto topk_values = torch::full({num_units, num_heads, topk}, 
//                                  -std::numeric_limits<float>::infinity(), options);
    
//     const dim3 blocks(num_units, num_heads);

//     // 关键修改：将共享内存计算移入类型分派宏内部
//     AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "dot_product_topk", ([&] {
//         // 现在scalar_t在此作用域内有效
//         const int smem_size = 2 * hidden_size * sizeof(scalar_t);
        
//         auto data_ptr = data.data_ptr<scalar_t>();
//         auto query_c_ptr = query_c.data_ptr<scalar_t>();
//         auto query_q_ptr = query_q.has_value() ? query_q->data_ptr<scalar_t>() : nullptr;
        
//         batched_dot_product_topk_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK, smem_size>>>(
//             data_ptr,
//             query_c_ptr,
//             query_q_ptr,
//             question_weight,
//             topk_indices.data_ptr<int64_t>(),
//             topk_values.data_ptr<scalar_t>(),
//             num_units,
//             hidden_size,
//             data_length,
//             topk
//         );
//     }));
    
//     return {topk_indices, topk_values};
// }

template<typename scalar_t>
__global__ void simple_topk_kernel(
    const scalar_t* __restrict__ data,
    int64_t* __restrict__ topk_indices,
    scalar_t* __restrict__ topk_values,
    int num_units,
    int hidden_size,
    int data_length,
    int topk
) {
    const int unit_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int data_idx = unit_id * data_length + thread_id;

    // Check if the thread is within bounds
    if (data_idx >= num_units * data_length) return;

    scalar_t value = data[data_idx];

    // Local array to store topk values and indices
    __shared__ scalar_t shared_values[THREADS_PER_BLOCK];
    __shared__ int64_t shared_indices[THREADS_PER_BLOCK];

    shared_values[thread_id] = value;
    shared_indices[thread_id] = data_idx;

    __syncthreads();

    // Perform a simple top-k selection (finding the maximum value in this block)
    for (int i = 0; i < THREADS_PER_BLOCK; ++i) {
        if (shared_values[i] > shared_values[thread_id]) {
            scalar_t temp_value = shared_values[i];
            int64_t temp_index = shared_indices[i];

            shared_values[i] = shared_values[thread_id];
            shared_indices[i] = shared_indices[thread_id];

            shared_values[thread_id] = temp_value;
            shared_indices[thread_id] = temp_index;
        }
    }

    // Write the top-k values and indices to global memory
    if (thread_id < topk) {
        topk_values[unit_id * topk + thread_id] = shared_values[thread_id];
        topk_indices[unit_id * topk + thread_id] = shared_indices[thread_id];
    }
}

} // namespace

std::pair<torch::Tensor, torch::Tensor> simple_topk_cuda(
    torch::Tensor data,
    int topk
) {
    TORCH_CHECK(data.dim() == 2, 
        "Data must be 2D tensor, but got ", data.dim(), "D tensor instead");

    const int num_units = data.size(0);
    const int hidden_size = data.size(1);
    const int data_length = data.size(0);

    auto options = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(data.device());
    auto topk_indices = torch::full({num_units, topk}, -1, options);

    options = options.dtype(data.dtype());
    auto topk_values = torch::full({num_units, topk}, 
                                 -std::numeric_limits<float>::infinity(), options);

    const dim3 blocks(num_units);

    // Launch a simple topk kernel
    AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "simple_topk", ([&] {
        const int smem_size = THREADS_PER_BLOCK * sizeof(scalar_t);
        
        auto data_ptr = data.data_ptr<scalar_t>();
        
        simple_topk_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK, smem_size>>>(
            data_ptr,
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
    m.def("simple_topk", &simple_topk_cuda, "Simplified topk kernel for testing");
}