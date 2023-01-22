#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
//#include <THC/THC.h>
#include "cuda/crop_and_resize_kernel_3d.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) AT_ASSERTM(x.dim() == 5, #x " must have 5 dimensions")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Float, #x " must be float Tensor")
#define CHECK_INT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Int, #x " must be int Tensor")
//using namespace at;


namespace torch {
void crop_and_resize_3d_gpu_forward(
    torch::Tensor image,
    torch::Tensor boxes,           // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_width,
    const int crop_length,
    const int crop_height,
    torch::Tensor crops
) {
    CHECK_INPUT(image);     CHECK_FLOAT(image);     CHECK_DIMS(image);
    CHECK_INPUT(boxes);     CHECK_FLOAT(boxes);
    CHECK_INPUT(box_index); CHECK_INT(box_index);
    CHECK_INPUT(crops);     CHECK_FLOAT(crops);

    const int batch_size    = image.size(0);
    const int depth         = image.size(1);
    const int image_width   = image.size(2);
    const int image_length  = image.size(3);
    const int image_height  = image.size(4);

    const int num_boxes     = boxes.size(0);

    // init output space
//    THCTensor_resize(state, crops, {num_boxes, depth, crop_height, crop_width});

    crops.resize_({num_boxes, depth, crop_width, crop_length, crop_height});
    crops.zero_();
//    THCudaTensor_resize4d(state, crops, num_boxes, depth, crop_height, crop_width);
//    THCudaTensor_zero(state, crops);



//    auto state = globalContext().getTHCState();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();// THCState_getCurrentStream(state);

    CropAndResizeLaucher3D(
        image.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_width, image_length, image_height,
        crop_width, crop_length, crop_height, depth, extrapolation_value,
        crops.data<float>(),
        stream
    );
}


void crop_and_resize_3d_gpu_backward(
    torch::Tensor grads,
    torch::Tensor boxes,      // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,    // range in [0, batch_size)
    torch::Tensor grads_image // resize to [bsize, c, wc, lc, hc]
) {
    CHECK_INPUT(grads);     CHECK_FLOAT(grads);
    CHECK_INPUT(boxes);     CHECK_FLOAT(boxes);
    CHECK_INPUT(box_index); CHECK_INT(box_index);
    CHECK_INPUT(grads_image); CHECK_FLOAT(grads_image); CHECK_DIMS(grads_image);

    // shape
    const int batch_size    = grads_image.size(0);
    const int depth         = grads_image.size(1);
    const int image_width   = grads_image.size(2);
    const int image_length  = grads_image.size(3);
    const int image_height  = grads_image.size(4);

    const int num_boxes     = grads.size(0);
    const int crop_width    = grads.size(2);
    const int crop_length   = grads.size(3);
    const int crop_height   = grads.size(4);

    // init output space
    grads_image.zero_();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CropAndResizeBackpropImageLaucher3D(
        grads.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_width, image_length, image_height,
        crop_width, crop_length, crop_height, depth,
        grads_image.data<float>(),
        stream
    );
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &torch::crop_and_resize_3d_gpu_forward,
      "crop_and_resize_3d_gpu_forward");
  m.def(
      "backward",
      &torch::crop_and_resize_3d_gpu_backward,
      "crop_and_resize_3d_gpu_backward");
}
