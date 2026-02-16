#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include "sampling_cuda_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sampling_cuda", &farthest_point_sampling_cuda, "farthest_point_sampling_cuda");
}
