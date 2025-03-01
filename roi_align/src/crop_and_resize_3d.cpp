#include <torch/extension.h>
//#include <TH/TH.h>
#include <stdio.h>
#include <math.h>

namespace torch {
template <typename scalar_t>
void CropAndResizePerBox3D(
    const scalar_t * image_data, 
    const int batch_size,
    const int depth,
    const int image_width,
    const int image_length,
    const int image_height,

    const scalar_t * boxes_data, 
    const int * box_index_data,
    const int start_box, 
    const int limit_box,

    scalar_t * crops_data,
    const int crop_width,
    const int crop_length,
    const int crop_height,
    const scalar_t spatial_scale,
    const scalar_t extrapolation_value
) {
    const int image_channel_elements = image_height * image_width * image_length;
    const int image_elements = depth * image_channel_elements;

    const int channel_elements = crop_height * crop_width * crop_length;
    const int crop_elements = depth * channel_elements;

    int b;
    #pragma omp parallel for
    for (b = start_box; b < limit_box; ++b) {
        const scalar_t * box = boxes_data + b * 6;
        const scalar_t x1 = box[0] * spatial_scale;
        const scalar_t y1 = box[1] * spatial_scale;
        const scalar_t z1 = box[2] * spatial_scale;
        const scalar_t x2 = box[3] * spatial_scale;
        const scalar_t y2 = box[4] * spatial_scale;
        const scalar_t z2 = box[5] * spatial_scale;

        const int b_in = box_index_data[b];
        if (b_in < 0 || b_in >= batch_size) {
            printf("Error: batch_index %d out of range [0, %d)\n", b_in, batch_size);
            exit(-1);
        }

        const scalar_t width_scale = (crop_width > 1) ? (x2 - x1) / (crop_width - 1) : 0;
        const scalar_t length_scale = (crop_length > 1) ? (y2 - y1) / (crop_length - 1) : 0;
        const scalar_t height_scale = (crop_height > 1) ? (z2 - z1) / (crop_height - 1) : 0;

        for (int x = 0; x < crop_width; ++x) {
            const scalar_t in_x = (crop_width > 1) ? x1 + x * width_scale : 0.5 * (x1 + x2);

            if (in_x < 0 || in_x > image_width - 1) {
                for (int y = 0; y < crop_length; ++y) {
                    for (int z = 0; z < crop_height; ++z) {
                        const int idx = z + crop_height * (y + crop_length * x);
                        for (int d = 0; d < depth; ++d) {
                            crops_data[crop_elements * b + channel_elements * d + idx] = extrapolation_value;
                        }
                    }
                }
                continue;
            }

            const int left_x_index = floorf(in_x);
            const int right_x_index = ceilf(in_x);
            const scalar_t x_lerp = in_x - left_x_index;

            for (int y = 0; y < crop_length; ++y) {
                const scalar_t in_y = (crop_length > 1) ? y1 + y * length_scale : 0.5 * (y1 + y2);

                if (in_y < 0 || in_y > image_length - 1) {
                    for (int z = 0; z < crop_height; ++z) {
                        const int idx = z + crop_height * (y + crop_length * x);
                        for (int d = 0; d < depth; ++d) {
                            // crops(b, y, x, d) = extrapolation_value;
                            crops_data[crop_elements * b + channel_elements * d + idx] = extrapolation_value;
                        }
                    }
                    continue;
                }
                
                const int top_y_index = floorf(in_y);
                const int bottom_y_index = ceilf(in_y);
                const scalar_t y_lerp = in_y - top_y_index;

                for (int z = 0; z < crop_height; ++z) {
                    const scalar_t in_z = (crop_height > 1) ? z1 + z * height_scale : 0.5 * (z1 + z2);

                    if (in_z < 0 || in_z > image_height - 1) {
                        const int idx = z + crop_height * (y + crop_length * x);
                        for (int d = 0; d < depth; ++d) {
                            crops_data[crop_elements * b + channel_elements * d + idx] = extrapolation_value;
                        }
                        continue;
                    }
                
                    const int near_z_index = floorf(in_z);
                    const int far_z_index = ceilf(in_z);
                    const scalar_t z_lerp = in_z - near_z_index;

                    const int idx = z + crop_height * (y + crop_length * x);
                    for (int d = 0; d < depth; ++d) {   
                        const scalar_t *pimage = image_data + b_in * image_elements + d * image_channel_elements;

                        const scalar_t near_top_left = pimage[near_z_index + image_height * (top_y_index + image_length * left_x_index)];
                        const scalar_t near_top_right = pimage[near_z_index + image_height * (top_y_index + image_length * right_x_index)];
                        const scalar_t near_bottom_left = pimage[near_z_index + image_height * (bottom_y_index + image_length * left_x_index)];
                        const scalar_t near_bottom_right = pimage[near_z_index + image_height * (bottom_y_index + image_length * right_x_index)];

                        const scalar_t far_top_left = pimage[far_z_index + image_height * (top_y_index + image_length * left_x_index)];
                        const scalar_t far_top_right = pimage[far_z_index + image_height * (top_y_index + image_length * right_x_index)];
                        const scalar_t far_bottom_left = pimage[far_z_index + image_height * (bottom_y_index + image_length * left_x_index)];
                        const scalar_t far_bottom_right = pimage[far_z_index + image_height * (bottom_y_index + image_length * right_x_index)];

                        const scalar_t near_top = near_top_left + (near_top_right - near_top_left) * x_lerp;
                        const scalar_t near_bottom = near_bottom_left + (near_bottom_right - near_bottom_left) * x_lerp;

                        const scalar_t far_top = far_top_left + (far_top_right - far_top_left) * x_lerp;
                        const scalar_t far_bottom = far_bottom_left + (far_bottom_right - far_bottom_left) * x_lerp;

                        const scalar_t top = near_top + (far_top - near_top) * z_lerp;
                        const scalar_t bottom = near_bottom + (far_bottom - near_bottom) * z_lerp;

                        crops_data[crop_elements * b + channel_elements * d + idx] = top + (bottom - top) * y_lerp;
                    }
                }   // end for z
            }   // end for y
        }   // end for x
    }   // end for b

}

#define CHECK_CUDA(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) AT_ASSERTM(x.dim() == 5, #x " must have 5 dimensions")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Float || \
    x.type().scalarType() == torch::ScalarType::Double, #x " must be float or double Tensor")
#define CHECK_INT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Int, #x " must be int Tensor")

void crop_and_resize_3d_forward(
    torch::Tensor image,
    torch::Tensor boxes,      // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,    // range in [0, batch_size)
    const double spatial_scale,
    const double extrapolation_value,
    const int crop_width,
    const int crop_length,
    const int crop_height,
    torch::Tensor crops
) {
    CHECK_INPUT(image);     CHECK_FLOAT(image);     CHECK_DIMS(image);
    CHECK_INPUT(boxes);     CHECK_FLOAT(boxes); //TODO: check dims for other arguments required.
    CHECK_INPUT(box_index); CHECK_INT(box_index);
    CHECK_INPUT(crops);     CHECK_FLOAT(crops);

    const int batch_size    = image.size(0);
    const int depth         = image.size(1);
    const int image_width   = image.size(2);
    const int image_length  = image.size(3);
    const int image_height  = image.size(4);

    const int num_boxes     = boxes.size(0);

    crops.resize_({num_boxes, depth, crop_width, crop_length, crop_height});
    crops.zero_();

    // crop_and_resize for each box
    AT_DISPATCH_FLOATING_TYPES(image.type(), "crop_and_resize_3d_forward", [&] {
        CropAndResizePerBox3D<scalar_t>(
            image.data<scalar_t>(),
            batch_size,
            depth,
            image_width,
            image_length,
            image_height,

            boxes.data<scalar_t>(),
            box_index.data<int>(),
            0,
            num_boxes,

            crops.data<scalar_t>(),
            crop_width,
            crop_length,
            crop_height,
            spatial_scale,
            extrapolation_value
        );
    });

}


template <typename scalar_t>
void CropAndResizePerBox3DBackward(
    const scalar_t * grads_data,
    const scalar_t * boxes_data,
    const int * box_index_data,
    scalar_t * grads_image_data,
    
    const int batch_size,
    const int depth,
    const int image_width,
    const int image_length,
    const int image_height,

    const int num_boxes,
    const int crop_width,
    const int crop_length,
    const int crop_height,
    const scalar_t spatial_scale
) {
    // n_elements
    const int image_channel_elements = image_height * image_width * image_length;
    const int image_elements = depth * image_channel_elements;

    const int channel_elements = crop_height * crop_width * crop_length;
    const int crop_elements = depth * channel_elements;

    for (int b = 0; b < num_boxes; ++b) {
        const scalar_t * box = boxes_data + b * 6;
        const scalar_t x1 = box[0] * spatial_scale;
        const scalar_t y1 = box[1] * spatial_scale;
        const scalar_t z1 = box[2] * spatial_scale;
        const scalar_t x2 = box[3] * spatial_scale;
        const scalar_t y2 = box[4] * spatial_scale;
        const scalar_t z2 = box[5] * spatial_scale;

        const int b_in = box_index_data[b];
        if (b_in < 0 || b_in >= batch_size) {
            printf("Error: batch_index %d out of range [0, %d)\n", b_in, batch_size);
            exit(-1);
        }

        const scalar_t width_scale = (crop_width > 1) ? (x2 - x1) / (crop_width - 1) : 0;
        const scalar_t length_scale = (crop_length > 1) ? (y2 - y1) / (crop_length - 1) : 0;
        const scalar_t height_scale = (crop_height > 1) ? (z2 - z1) / (crop_height - 1) : 0;

        for (int x = 0; x < crop_width; ++x) {
            const scalar_t in_x = (crop_width > 1) ? x1 + x * width_scale : 0.5 * (x1 + x2);
            if (in_x < 0 || in_x > image_width - 1) {
                continue;
            }

            const int left_x_index = floorf(in_x);
            const int right_x_index = ceilf(in_x);
            const scalar_t x_lerp = in_x - left_x_index;

            for (int y = 0; y < crop_length; ++y) {
                const scalar_t in_y = (crop_length > 1) ? y1 + y * length_scale : 0.5 * (y1 + y2);
                if (in_y < 0 || in_y > image_length - 1) {
                    continue;
                }

                const int top_y_index = floorf(in_y);
                const int bottom_y_index = ceilf(in_y);
                const scalar_t y_lerp = in_y - top_y_index;

                for (int z = 0; z < crop_height; ++z) {
                    const scalar_t in_z = (crop_height > 1) ? z1 + z * height_scale : 0.5 * (z1 + z2);
                    if (in_z < 0 || in_z > image_height - 1) {
                        continue;
                    }

                    const int near_z_index = floorf(in_z);
                    const int far_z_index = ceilf(in_z);
                    const scalar_t z_lerp = in_z - near_z_index;

                    for (int d = 0; d < depth; ++d) {
                        scalar_t *pimage = grads_image_data + b_in * image_elements + d * image_channel_elements;
                        const int idx = z + crop_height * (y + crop_length * x);
                        const scalar_t grad_val = grads_data[crop_elements * b + channel_elements * d + idx];

                        const scalar_t dtop = (1 - y_lerp) * grad_val;
                        const scalar_t dnear_top = (1 - z_lerp) * dtop;
                        const scalar_t dfar_top = z_lerp * dtop;
                        pimage[near_z_index + image_height * (top_y_index + image_length * left_x_index)] += (1 - x_lerp) * dnear_top;
                        pimage[near_z_index + image_height * (top_y_index + image_length * right_x_index)] += x_lerp * dnear_top;
                        pimage[far_z_index + image_height * (top_y_index + image_length * left_x_index)] += (1 - x_lerp) * dfar_top;
                        pimage[far_z_index + image_height * (top_y_index + image_length * right_x_index)] += x_lerp * dfar_top;

                        const scalar_t dbottom = y_lerp * grad_val;
                        const scalar_t dnear_bottom = (1 - z_lerp) * dbottom;
                        const scalar_t dfar_bottom = z_lerp * dbottom;
                        pimage[near_z_index + image_height * (bottom_y_index + image_length * left_x_index)] += (1 - x_lerp) * dnear_bottom;
                        pimage[near_z_index + image_height * (bottom_y_index + image_length * right_x_index)] += x_lerp * dnear_bottom;
                        pimage[far_z_index + image_height * (bottom_y_index + image_length * left_x_index)] += (1 - x_lerp) * dfar_bottom;
                        pimage[far_z_index + image_height * (bottom_y_index + image_length * right_x_index)] += x_lerp * dfar_bottom;
                    }   // end d
                }   // end z
            }   // end y
        }   // end x
    }   // end b
}


void crop_and_resize_3d_backward(
    torch::Tensor grads,
    torch::Tensor boxes,      // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,    // range in [0, batch_size)
    const double spatial_scale,
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
//    THFloatTensor_zero(grads_image);

    AT_DISPATCH_FLOATING_TYPES(grads.type(), "crop_and_resize_3d_backward", ([&] {
        CropAndResizePerBox3DBackward<scalar_t>(
            grads.data<scalar_t>(),
            boxes.data<scalar_t>(),
            box_index.data<int>(),
            grads_image.data<scalar_t>(),
            batch_size,
            depth,
            image_width,
            image_length,
            image_height,
            num_boxes,
            crop_width,
            crop_length,
            crop_height,
            spatial_scale
        );
    }));
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &torch::crop_and_resize_3d_forward,
      "crop_and_resize_3d_forward");
  m.def(
      "backward",
      &torch::crop_and_resize_3d_backward,
      "crop_and_resize_3d_backward");
}
