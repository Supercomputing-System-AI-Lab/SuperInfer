#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <array>

#include <zmq.hpp>
#include <msgpack.hpp>
#include <cuda_runtime.h>

#include "swapper.h"

#define CHECK_CUDA(call)                                                   \
    {                                                                      \
        const cudaError_t error = call;                                    \
        if (error != cudaSuccess) {                                        \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "; \
            std::cerr << "code: " << error                                 \
                      << ", reason: " << cudaGetErrorString(error)         \
                      << std::endl;                                        \
            exit(1);                                                       \
        }                                                                  \
    }

void swap_block_first(const blk_idx_pairs &blks_to_swap_in, const blk_idx_pairs &blks_to_swap_out,
                      const size_t num_attn_layers,
                      void *kv_cache_gpu_ptr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
                      const std::vector<size_t> &gpu_strides,
                      void *kv_cache_cpu_ptr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
                      const std::vector<size_t> &cpu_strides,
                      cudaStream_t *stream_in_ptr, cudaStream_t *stream_out_ptr,
                      // alloc memory
                      std::vector<size_t> &cached_sizes_vec, std::vector<size_t> &cached_fail_idx_vec,
                      std::vector<void *> &prealloc_d2h_dsts_vec, std::vector<void *> &prealloc_d2h_srcs_vec,
                      std::vector<void *> &prealloc_h2d_dsts_vec, std::vector<void *> &prealloc_h2d_srcs_vec) {
    const size_t blk_size_in_bytes = calculate_size(cpu_sizes, 1) * cpu_elem_size;
    std::cout << "blk_size_in_bytes: " << blk_size_in_bytes << std::endl;

    const size_t d2h_count = blks_to_swap_out.size();
    const size_t h2d_count = blks_to_swap_in.size();
    const size_t max_count = std::max(d2h_count, h2d_count);

    // extend space
    if (cached_sizes_vec.size() < max_count) {
        cached_sizes_vec.resize(max_count, blk_size_in_bytes);
    }
    if (cached_fail_idx_vec.size() < max_count) {
        cached_fail_idx_vec.resize(max_count);
    }
    if (prealloc_d2h_dsts_vec.size() < d2h_count) {
        prealloc_d2h_dsts_vec.reserve(d2h_count);
    }
    if (prealloc_d2h_srcs_vec.size() < d2h_count) {
        prealloc_d2h_srcs_vec.reserve(d2h_count);
    }
    if (prealloc_h2d_dsts_vec.size() < h2d_count) {
        prealloc_h2d_dsts_vec.reserve(h2d_count);
    }
    if (prealloc_h2d_srcs_vec.size() < h2d_count) {
        prealloc_h2d_srcs_vec.reserve(h2d_count);
    }

    // d2h first
    auto &d2h_sizes_vec = cached_sizes_vec;
    auto &fail_idx_vec = cached_fail_idx_vec;
    auto &d2h_dsts_vec = prealloc_d2h_dsts_vec;
    auto &d2h_srcs_vec = prealloc_d2h_srcs_vec;
    d2h_dsts_vec.clear();
    d2h_srcs_vec.clear();
    for (const auto &blk_idx_pair: blks_to_swap_out) {
        const auto [blk_idx_gpu, blk_idx_cpu] = blk_idx_pair;
        void *dst_ptr = ptr_offset(kv_cache_cpu_ptr, cpu_elem_size, cpu_strides, {blk_idx_cpu});
        void *src_ptr = ptr_offset(kv_cache_gpu_ptr, gpu_elem_size, gpu_strides, {blk_idx_gpu});
        d2h_dsts_vec.emplace_back(dst_ptr);
        d2h_srcs_vec.emplace_back(src_ptr);
    }
    constexpr cudaMemcpyAttributes d2h_attr = {
        .srcAccessOrder = cudaMemcpySrcAccessOrderAny,
        .srcLocHint = {
            .type = cudaMemLocationTypeDevice,
            .id = 0,
        },
        .dstLocHint = {
            .type = cudaMemLocationTypeHostNumaCurrent,
        },
        // .flags = cudaMemcpyFlagPreferOverlapWithCompute,
    };
    cudaMemcpyBatchAsync(d2h_dsts_vec.data(), d2h_srcs_vec.data(), d2h_sizes_vec.data(),
                         d2h_count, d2h_attr, fail_idx_vec.data(), *stream_out_ptr);

    // h2d then
    auto &h2d_sizes_vec = cached_sizes_vec;
    // auto &fail_idx_vec = cached_fail_idx_vec;
    auto &h2d_dsts_vec = prealloc_h2d_dsts_vec;
    auto &h2d_srcs_vec = prealloc_h2d_srcs_vec;
    h2d_dsts_vec.clear();
    h2d_srcs_vec.clear();
    for (const auto &blk_idx_pair: blks_to_swap_in) {
        const auto [blk_idx_cpu, blk_idx_gpu] = blk_idx_pair;
        void *dst_ptr = ptr_offset(kv_cache_gpu_ptr, gpu_elem_size, gpu_strides, {blk_idx_gpu});
        void *src_ptr = ptr_offset(kv_cache_cpu_ptr, cpu_elem_size, cpu_strides, {blk_idx_cpu});
        h2d_dsts_vec.emplace_back(dst_ptr);
        h2d_srcs_vec.emplace_back(src_ptr);
    }
    constexpr cudaMemcpyAttributes h2d_attr = {
        .srcAccessOrder = cudaMemcpySrcAccessOrderAny,
        .srcLocHint = {
            .type = cudaMemLocationTypeHostNumaCurrent,
        },
        .dstLocHint = {
            .type = cudaMemLocationTypeDevice,
            .id = 0,
        },
        // .flags = cudaMemcpyFlagPreferOverlapWithCompute,
    };
    cudaMemcpyBatchAsync(h2d_dsts_vec.data(), h2d_srcs_vec.data(), h2d_sizes_vec.data(),
                         h2d_count, h2d_attr, fail_idx_vec.data(), *stream_in_ptr);

    CHECK_CUDA(cudaStreamSynchronize(*stream_out_ptr));
    CHECK_CUDA(cudaStreamSynchronize(*stream_in_ptr));
}

void swap(const blk_idx_pairs &blks_to_swap_in, const blk_idx_pairs &blks_to_swap_out, const size_t num_attn_layers,
          void *kv_cache_gpu_ptr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
          const std::vector<size_t> &gpu_strides,
          void *kv_cache_cpu_ptr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
          const std::vector<size_t> &cpu_strides,
          cudaStream_t *stream_in_ptr = nullptr, cudaStream_t *stream_out_ptr = nullptr
) {
    // pitch
    const size_t pitch_cpu = cpu_strides[1] * cpu_elem_size;
    const size_t pitch_gpu = gpu_strides[1] * gpu_elem_size;
    const size_t blk_size_in_bytes = calculate_size(cpu_sizes, 3) * cpu_elem_size;
    const size_t width = blk_size_in_bytes;
    const size_t height = num_attn_layers * 2;
    // std::cout << "pitch_cpu: " << pitch_cpu << ", pitch_gpu: " << pitch_gpu
    //           << ", blk_size_in_bytes: " << blk_size_in_bytes
    //           << ", width: " << width << ", height: " << height << std::endl;
    // std::cout << "Start swapping in C++." << std::endl;

    for (const auto &blk_idx_pair: blks_to_swap_out) {
        const auto [blk_idx_gpu, blk_idx_cpu] = blk_idx_pair;
        void *dst_ptr = ptr_offset(kv_cache_cpu_ptr, cpu_elem_size, cpu_strides, {0, 0, blk_idx_cpu});
        const auto dpitch = pitch_cpu;
        const auto src_ptr = ptr_offset(kv_cache_gpu_ptr, gpu_elem_size, gpu_strides, {0, 0, blk_idx_gpu});
        const auto spitch = pitch_gpu;
        CHECK_CUDA(
            cudaMemcpy2DAsync(dst_ptr, dpitch, src_ptr, spitch, width, height, cudaMemcpyDeviceToHost, *
                stream_out_ptr
            ));
    }

    for (const auto &blk_idx_pair: blks_to_swap_in) {
        const auto [blk_idx_cpu, blk_idx_gpu] = blk_idx_pair;
        void *dst_ptr = ptr_offset(kv_cache_gpu_ptr, gpu_elem_size, gpu_strides, {0, 0, blk_idx_gpu});
        const auto dpitch = pitch_gpu;
        const auto src_ptr = ptr_offset(kv_cache_cpu_ptr, cpu_elem_size, cpu_strides, {0, 0, blk_idx_cpu});
        const auto spitch = pitch_cpu;
        CHECK_CUDA(
            cudaMemcpy2DAsync(dst_ptr, dpitch, src_ptr, spitch, width, height, cudaMemcpyHostToDevice, *
                stream_in_ptr));
    }

    CHECK_CUDA(cudaStreamSynchronize(*stream_out_ptr));
    CHECK_CUDA(cudaStreamSynchronize(*stream_in_ptr));
    // std::cout << "End swapping in C++." << std::endl;
}


void swap_naive(const blk_idx_pairs &blks_to_swap_in, const blk_idx_pairs &blks_to_swap_out,
                const size_t num_attn_layers,
                void *kv_cache_gpu_ptr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
                const std::vector<size_t> &gpu_strides,
                void *kv_cache_cpu_ptr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
                const std::vector<size_t> &cpu_strides,
                cudaStream_t *stream_in_ptr = nullptr, cudaStream_t *stream_out_ptr = nullptr
) {
    const size_t blk_size_in_bytes = calculate_size(cpu_sizes, 3) * cpu_elem_size;
    auto stream_ptr = stream_in_ptr;

    // out
    for (size_t layer_idx = 0; layer_idx < num_attn_layers; ++layer_idx) {
        // K cache
        for (const auto &blk_idx_pair: blks_to_swap_out) {
            const auto [blk_idx_gpu, blk_idx_cpu] = blk_idx_pair;
            void *dst_ptr = ptr_offset(kv_cache_cpu_ptr, cpu_elem_size, cpu_strides, {0, layer_idx, blk_idx_cpu});
            const auto src_ptr = ptr_offset(kv_cache_gpu_ptr, gpu_elem_size, gpu_strides, {0, layer_idx, blk_idx_gpu});
            CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, blk_size_in_bytes, cudaMemcpyDeviceToHost, *stream_ptr));
        }
        // V cache
        for (const auto &blk_idx_pair: blks_to_swap_out) {
            const auto [blk_idx_gpu, blk_idx_cpu] = blk_idx_pair;
            void *dst_ptr = ptr_offset(kv_cache_cpu_ptr, cpu_elem_size, cpu_strides, {1, layer_idx, blk_idx_cpu});
            const auto src_ptr = ptr_offset(kv_cache_gpu_ptr, gpu_elem_size, gpu_strides, {1, layer_idx, blk_idx_gpu});
            CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, blk_size_in_bytes, cudaMemcpyDeviceToHost, *stream_ptr));
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(*stream_ptr));
    //in
    for (size_t layer_idx = 0; layer_idx < num_attn_layers; ++layer_idx) {
        // K cache
        for (const auto &blk_idx_pair: blks_to_swap_in) {
            const auto [blk_idx_cpu, blk_idx_gpu] = blk_idx_pair;
            void *dst_ptr = ptr_offset(kv_cache_gpu_ptr, gpu_elem_size, gpu_strides, {0, layer_idx, blk_idx_gpu});
            const auto src_ptr = ptr_offset(kv_cache_cpu_ptr, cpu_elem_size, cpu_strides, {0, layer_idx, blk_idx_cpu});
            CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, blk_size_in_bytes, cudaMemcpyHostToDevice, *stream_ptr));
        }
        // V cache
        for (const auto &blk_idx_pair: blks_to_swap_in) {
            const auto [blk_idx_cpu, blk_idx_gpu] = blk_idx_pair;
            void *dst_ptr = ptr_offset(kv_cache_gpu_ptr, gpu_elem_size, gpu_strides, {1, layer_idx, blk_idx_gpu});
            const auto src_ptr = ptr_offset(kv_cache_cpu_ptr, cpu_elem_size, cpu_strides, {1, layer_idx, blk_idx_cpu});
            CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, blk_size_in_bytes, cudaMemcpyHostToDevice, *stream_ptr));
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(*stream_ptr));
}

// // kernel of vllm v0
// void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
//                  const blk_idx_pairs &blks_to_swap_in, const blk_idx_pairs &blks_to_swap_out,
//                  const torch::Tensor& block_mapping) {
//     torch::Device src_device = src.device();
//     torch::Device dst_device = dst.device();
//     cudaMemcpyKind memcpy_type;
//     if (src_device.is_cuda() && dst_device.is_cuda()) {
//         TORCH_CHECK(src_device.index() == dst_device.index(),
//                     "src and dst must be on the same GPU");
//         memcpy_type = cudaMemcpyDeviceToDevice;
//     } else if (src_device.is_cuda() && dst_device.is_cpu()) {
//         memcpy_type = cudaMemcpyDeviceToHost;
//     } else if (src_device.is_cpu() && dst_device.is_cuda()) {
//         memcpy_type = cudaMemcpyHostToDevice;
//     } else {
//         TORCH_CHECK(false, "Invalid device combination");
//     }
//
//     // NOTE(youkaichao): keep in mind that `block_mapping` should be
//     // a cpu tensor, otherwise every `item` call will require a gpu-cpu
//     // synchronization.
//     TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");
//
//     char* src_ptr = static_cast<char*>(src.data_ptr());
//     char* dst_ptr = static_cast<char*>(dst.data_ptr());
//
//     const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
//     const at::cuda::OptionalCUDAGuard device_guard(
//         src_device.is_cuda() ? src_device : dst_device);
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     // NOTE(woosuk): This can be slow if the number of blocks is large.
//     const int64_t num_blocks = block_mapping.size(0);
//     for (size_t i = 0; i < num_blocks; i++) {
//         int64_t src_block_number = block_mapping[i][0].item<int64_t>();
//         int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
//         int64_t src_offset = src_block_number * block_size_in_bytes;
//         int64_t dst_offset = dst_block_number * block_size_in_bytes;
//         cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
//                         block_size_in_bytes, memcpy_type, stream);
//     }
// }

void swapper_thread(const std::string &data_path, const std::string &notify_path, const size_t num_attn_layers,
                    void *kv_cache_gpu_ptr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
                    const std::vector<size_t> &gpu_strides,
                    void *kv_cache_cpu_ptr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
                    const std::vector<size_t> &cpu_strides,
                    cudaStream_t *stream_in_ptr = nullptr, cudaStream_t *stream_out_ptr = nullptr,
                    const bool block_first = true
) {
    if (stream_in_ptr == nullptr) {
        cudaStream_t stream_in;
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_in, cudaStreamNonBlocking));
        stream_in_ptr = &stream_in;
    }
    if (stream_out_ptr == nullptr) {
        cudaStream_t stream_out;
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_out, cudaStreamNonBlocking));
        stream_out_ptr = &stream_out;
    }

    zmq::context_t context(1);
    zmq::socket_t receiver(context, zmq::socket_type::pull);
    try {
        receiver.bind(data_path);
    } catch (const zmq::error_t &e) {
        std::cerr << "Failed to bind to path " << data_path << ": " << e.what() << std::endl;
        return;
    }

    zmq::socket_t notifier(context, zmq::socket_type::push);
    try {
        notifier.connect(notify_path);
    } catch (const zmq::error_t &e) {
        std::cerr << "Failed to connect to path " << notify_path << ": " << e.what() << std::endl;
        return;
    }

    bool USE_NAIVE_SWAP_KERNEL = std::getenv("USE_NAIVE_SWAP_KERNEL") != nullptr;

    std::cout << "starting swapper loop ..." << std::endl;

    zmq::message_t msg(100000); // pre-allocate 1MB buffer
    // zmq::message_t notify_msg(sizeof(size_t)); // to send back time used
    std::array<blk_idx_pairs, 2> data; // [2, N, 2], [0]: swap in, [1]: swap out

    constexpr size_t prealloc_size = 20000;
    data[0].reserve(prealloc_size);
    data[1].reserve(prealloc_size);

    // for block-fist swapper
    std::vector<size_t> cached_sizes_vec;
    std::vector<size_t> cached_fail_idx_vec;
    std::vector<void *> prealloc_d2h_dsts_vec;
    std::vector<void *> prealloc_d2h_srcs_vec;
    std::vector<void *> prealloc_h2d_dsts_vec;
    std::vector<void *> prealloc_h2d_srcs_vec;
    if (block_first) {
        const size_t blk_size_in_bytes = calculate_size(cpu_sizes, 1) * cpu_elem_size;
        cached_sizes_vec.resize(prealloc_size, blk_size_in_bytes);
        cached_fail_idx_vec.resize(prealloc_size);
        prealloc_d2h_dsts_vec.reserve(prealloc_size);
        prealloc_d2h_srcs_vec.reserve(prealloc_size);
        prealloc_h2d_dsts_vec.reserve(prealloc_size);
        prealloc_h2d_srcs_vec.reserve(prealloc_size);
    }

    while (true) {
        // std::cout << "waiting for message ..." << std::endl;

        // receiving msgspec from python
        // msg.rebuild(); // clear previous message
        auto recvd = receiver.recv(msg, zmq::recv_flags::none);

        auto time_beg = std::chrono::high_resolution_clock::now();

        if (!recvd) {
            std::cerr << "Failed to receive message." << std::endl;
            continue;
        }

        // std::cout << "Message received. Size: " << msg.size() << " bytes." << std::endl;

        // unpacking msgpack
        data[0].clear();
        data[1].clear();
        try {
            auto oh = msgpack::unpack(static_cast<const char *>(msg.data()), msg.size());
            auto obj = oh.get();
            obj.convert(data);
            // std::cout << "Received " << data[0].size() << " blocks to swap in and "
            // << data[1].size() << " blocks to swap out." << std::endl;
        } catch (const zmq::error_t &e) {
            std::cerr << "Failed to convert message." << std::endl;
            continue;
        }

        // check data
        if (data[0].empty() && data[1].empty()) {
            std::cerr << "Received empty swap lists. Skipping." << std::endl;
            continue;
        }

        // do swapping
        if (USE_NAIVE_SWAP_KERNEL) {
            std::cout << "Using naive swap kernel." << std::endl;
            swap_naive(data[0], data[1], num_attn_layers,
                       kv_cache_gpu_ptr, gpu_elem_size, gpu_sizes, gpu_strides,
                       kv_cache_cpu_ptr, cpu_elem_size, cpu_sizes, cpu_strides,
                       stream_in_ptr, stream_out_ptr);
        } else if (block_first) {
            swap_block_first(data[0], data[1], num_attn_layers,
                             kv_cache_gpu_ptr, gpu_elem_size, gpu_sizes, gpu_strides,
                             kv_cache_cpu_ptr, cpu_elem_size, cpu_sizes, cpu_strides,
                             stream_in_ptr, stream_out_ptr,
                             cached_sizes_vec, cached_fail_idx_vec,
                             prealloc_d2h_dsts_vec, prealloc_d2h_srcs_vec,
                             prealloc_h2d_dsts_vec, prealloc_h2d_srcs_vec);
        } else {
            swap(data[0], data[1], num_attn_layers,
                 kv_cache_gpu_ptr, gpu_elem_size, gpu_sizes, gpu_strides,
                 kv_cache_cpu_ptr, cpu_elem_size, cpu_sizes, cpu_strides,
                 stream_in_ptr, stream_out_ptr);
        }

        auto time_end = std::chrono::high_resolution_clock::now();
        auto time_used_us = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_beg).count();
        auto time_used_ms = static_cast<double>(time_used_us) / 1000;

        // std::cout << "Converted message. "
        //           << "Received " << data[0].size() << " blocks to swap in and "
        //           << data[1].size() << " blocks to swap out." << std::endl;
        // std::cout << "Swapping done in " << time_used_ms << " ms." << std::endl;
        // std::memcpy(notify_msg.data(), &time_used_us, sizeof(time_used_us));

        zmq::message_t notify_msg(&time_used_us, sizeof(time_used_us));

        // done
        notifier.send(notify_msg, zmq::send_flags::none);
        // std::cout << "Notification sent." << std::endl;
    }
}

void start_swapper_thread(
    const std::string &data_path, const std::string &notify_path, const size_t num_attn_layers,
    const size_t kv_cache_gpu_addr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
    const std::vector<size_t> &gpu_strides,
    const size_t kv_cache_cpu_addr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
    const std::vector<size_t> &cpu_strides, const bool block_first
) {
    auto kv_cache_gpu_ptr = reinterpret_cast<void *>(kv_cache_gpu_addr);
    auto kv_cache_cpu_ptr = reinterpret_cast<void *>(kv_cache_cpu_addr);
    std::thread swapper(swapper_thread, data_path, notify_path, num_attn_layers,
                        kv_cache_gpu_ptr, gpu_elem_size, gpu_sizes, gpu_strides,
                        kv_cache_cpu_ptr, cpu_elem_size, cpu_sizes, cpu_strides,
                        nullptr, nullptr, block_first);
    swapper.detach();
    std::cout << "Swapper thread started in the background." << std::endl;
}

void start_swapper_fork(
    const size_t stream_in_ptr, const size_t stream_out_ptr,
    const std::string &data_path, const std::string &notify_path, const size_t num_attn_layers,
    const size_t kv_cache_gpu_addr, const size_t gpu_elem_size, const std::vector<size_t> &gpu_sizes,
    const std::vector<size_t> &gpu_strides,
    const size_t kv_cache_cpu_addr, const size_t cpu_elem_size, const std::vector<size_t> &cpu_sizes,
    const std::vector<size_t> &cpu_strides
) {
    auto kv_cache_gpu_ptr = reinterpret_cast<void *>(kv_cache_gpu_addr);
    auto kv_cache_cpu_ptr = reinterpret_cast<void *>(kv_cache_cpu_addr);
    auto pid = fork();
    if (pid < 0) {
        std::cerr << "Failed to fork process." << std::endl;
        return;
    }
    if (pid == 0) {
        // Child process
        cudaStream_t *stream_in_c_ptr = reinterpret_cast<cudaStream_t *>(stream_in_ptr);
        cudaStream_t *stream_out_c_ptr = reinterpret_cast<cudaStream_t *>(stream_out_ptr);
        swapper_thread(data_path, notify_path, num_attn_layers,
                       kv_cache_gpu_ptr, gpu_elem_size, gpu_sizes, gpu_strides,
                       kv_cache_cpu_ptr, cpu_elem_size, cpu_sizes, cpu_strides,
                       stream_in_c_ptr, stream_out_c_ptr);
        std::exit(0); // Ensure child process exits after thread function completes
    } else {
        // Parent process
        std::cout << "Started swapper process with PID" << pid << std::endl;
    }
}
