#ifndef SWAPPER_MT_SWAPPER_H
#define SWAPPER_MT_SWAPPER_H

#include <string>
#include <vector>
#include <tuple>

#include <cuda_runtime.h>

using blk_idx_pairs = std::vector<std::tuple<size_t, size_t> >;

void swap_block_first(const blk_idx_pairs &blks_to_swap_in, const blk_idx_pairs &blks_to_swap_out,
                      const size_t num_attn_layers,
                      void *kv_cache_gpu_ptr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
                      const std::vector<size_t> &gpu_strides,
                      void *kv_cache_cpu_ptr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
                      const std::vector<size_t> &cpu_strides,
                      cudaStream_t *stream_in_ptr, cudaStream_t *stream_out_ptr,
                      std::vector<size_t> &cached_sizes_vec, std::vector<size_t> &cached_fail_idx_vec,
                      std::vector<void *> &prealloc_d2h_dsts_vec, std::vector<void *> &prealloc_d2h_srcs_vec,
                      std::vector<void *> &prealloc_h2d_dsts_vec, std::vector<void *> &prealloc_h2d_srcs_vec);

void swap(const blk_idx_pairs &blks_to_swap_in, const blk_idx_pairs &blks_to_swap_out, const size_t num_attn_layers,
          void *kv_cache_gpu_ptr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
          const std::vector<size_t> &gpu_strides,
          void *kv_cache_cpu_ptr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
          const std::vector<size_t> &cpu_strides,
          cudaStream_t *stream_in_ptr, cudaStream_t *stream_out_ptr);

void swap_naive(const blk_idx_pairs &blks_to_swap_in, const blk_idx_pairs &blks_to_swap_out,
                const size_t num_attn_layers,
                void *kv_cache_gpu_ptr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
                const std::vector<size_t> &gpu_strides,
                void *kv_cache_cpu_ptr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
                const std::vector<size_t> &cpu_strides,
                cudaStream_t *stream_in_ptr, cudaStream_t *stream_out_ptr);

void swapper_thread(const std::string &data_path, const std::string &notify_path, const size_t num_attn_layers,
                    void *kv_cache_gpu_ptr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
                    const std::vector<size_t> &gpu_strides,
                    void *kv_cache_cpu_ptr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
                    const std::vector<size_t> &cpu_strides,
                    cudaStream_t *stream_in_ptr, cudaStream_t *stream_out_ptr, const bool block_first);

void start_swapper_thread(
    const std::string &data_path, const std::string &notify_path, const size_t num_attn_layers,
    const size_t kv_cache_gpu_addr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
    const std::vector<size_t> &gpu_strides,
    const size_t kv_cache_cpu_addr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
    const std::vector<size_t> &cpu_strides, const bool block_first);

void start_swapper_fork(
    const size_t stream_in_ptr, const size_t stream_out_ptr,
    const std::string &data_path, const std::string &notify_path, const size_t num_attn_layers,
    const size_t kv_cache_gpu_addr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
    const std::vector<size_t> &gpu_strides,
    const size_t kv_cache_cpu_addr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
    const std::vector<size_t> &cpu_strides);

inline size_t calculate_size(const std::vector<size_t> &sizes, const size_t start_dim = 0) {
    if (start_dim >= sizes.size()) {
        return 0;
    }
    size_t size = 1;
    for (size_t i = start_dim; i < sizes.size(); ++i) {
        size *= sizes[i];
    }
    return size;
}

inline void *ptr_offset(const void *base_ptr, const size_t elem_size, const std::vector<size_t> &strides,
                        const std::vector<size_t> &idx) {
    size_t offset_elem = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        offset_elem += idx[i] * strides[i];
    }
    void *ptr = static_cast<char *>(const_cast<void *>(base_ptr)) + offset_elem * elem_size;
    return ptr;
}

#endif //SWAPPER_MT_SWAPPER_H
