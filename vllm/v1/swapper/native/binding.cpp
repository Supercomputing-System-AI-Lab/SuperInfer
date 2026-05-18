#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "swapper.h"

namespace py = pybind11;

PYBIND11_MODULE(swapper_mt, m) {
    m.def("start_swapper_thread", &start_swapper_thread,
          "Starts the swapper thread in the background.",
          py::arg("data_path"), py::arg("notify_path"), py::arg("num_attn_layers"),
          py::arg("kv_cache_gpu_addr"), py::arg("gpu_elem_size"), py::arg("gpu_sizes"), py::arg("gpu_strides"),
          py::arg("kv_cache_cpu_addr"), py::arg("cpu_elem_size"), py::arg("cpu_sizes"), py::arg("cpu_strides"),
          py::arg("block_first"));
    m.def("start_swapper_fork", &start_swapper_fork,
          "Starts the swapper process in the background.",
          py::arg("stream_in_ptr"), py::arg("stream_out_ptr"),
          py::arg("data_path"), py::arg("notify_path"), py::arg("num_attn_layers"),
          py::arg("kv_cache_gpu_addr"), py::arg("gpu_elem_size"), py::arg("gpu_sizes"), py::arg("gpu_strides"),
          py::arg("kv_cache_cpu_addr"), py::arg("cpu_elem_size"), py::arg("cpu_sizes"), py::arg("cpu_strides"));
}
