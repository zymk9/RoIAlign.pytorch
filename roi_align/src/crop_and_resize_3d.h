namespace at {
struct Tensor;
} // namespace at
namespace torch {
void crop_and_resize_3d_forward(
    at::Tensor image,
    at::Tensor boxes,      // [x1, y1, z1, x2, y2, z2]
    at::Tensor box_index,    // range in [0, batch_size) // int tensor
    const double spatial_scale,
    const double extrapolation_value,
    const int crop_width,
    const int crop_length,
    const int crop_height,
    at::Tensor crops
);

void crop_and_resize_3d_backward(
    at::Tensor grads,
    at::Tensor boxes,      // [x1, y1, z1, x2, y2, z2]
    at::Tensor box_index,    // range in [0, batch_size) // int
    const double spatial_scale,
    at::Tensor grads_image // resize to [bsize, c, wc, lc, hc]
);
}