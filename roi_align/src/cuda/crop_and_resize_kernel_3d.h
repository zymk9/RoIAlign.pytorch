#ifndef _CropAndResize_Kernel_3D
#define _CropAndResize_Kernel_3D

//#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#ifdef __cplusplus
extern "C" {
#endif

void CropAndResizeLaucher3D(
    const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length, 
    int crop_height, int depth, float extrapolation_value, 
    float *crops_ptr, cudaStream_t stream);

void CropAndResizeBackpropImageLaucher3D(
    const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length,
    int crop_height, int depth, float *grads_image_ptr, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif