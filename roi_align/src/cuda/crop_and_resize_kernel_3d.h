#ifndef _CropAndResize_Kernel_3D
#define _CropAndResize_Kernel_3D

//#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#ifdef __cplusplus
extern "C" {
#endif

void crop_and_resize_3d_cuda_forward(
    torch::Tensor image,
    torch::Tensor boxes,           // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,    // range in [0, batch_size) // int
    const float extrapolation_value,
    const int crop_width,
    const int crop_length,
    const int crop_height,
    torch::Tensor crops
);


void crop_and_resize_3d_cuda_backward(
    torch::Tensor grads,
    torch::Tensor boxes,      // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,    // range in [0, batch_size) // int
    torch::Tensor grads_image // resize to [bsize, c, wc, lc, hc]
);

#ifdef __cplusplus
}
#endif

#endif
