namespace torch {
//cuda tensors
void crop_and_resize_3d_gpu_forward(
    torch::Tensor image,
    torch::Tensor boxes,           // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,    // range in [0, batch_size) // int
    const double spatial_scale,
    const double extrapolation_value,
    const int crop_width,
    const int crop_length,
    const int crop_height,
    torch::Tensor crops
);

void crop_and_resize_3d_gpu_backward(
    torch::Tensor grads,
    torch::Tensor boxes,      // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,    // range in [0, batch_size) // int
    const double spatial_scale,
    torch::Tensor grads_image // resize to [bsize, c, wc, lc, hc]
);
}