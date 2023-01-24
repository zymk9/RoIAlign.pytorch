#include <math.h>
#include <stdio.h>
#include "crop_and_resize_kernel_3d.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)


template <typename scalar_t>
__global__
void CropAndResizeKernel3D(
    const int nthreads, const scalar_t *image_ptr, const scalar_t *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length,
    int crop_height, int depth, scalar_t spatial_scale, scalar_t extrapolation_value, 
    scalar_t *crops_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NCWLH: out_idx = h + crop_height * (l + crop_length * (w + crop_width * (d + depth * b)))
        int idx = out_idx;
        const int z = idx % crop_height;
        idx /= crop_height;
        const int y = idx % crop_length;
        idx /= crop_length;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int d = idx % depth;
        const int b = idx / depth;

        const scalar_t x1 = boxes_ptr[b * 6] * spatial_scale;
        const scalar_t y1 = boxes_ptr[b * 6 + 1] * spatial_scale;
        const scalar_t z1 = boxes_ptr[b * 6 + 2] * spatial_scale;
        const scalar_t x2 = boxes_ptr[b * 6 + 3] * spatial_scale;
        const scalar_t y2 = boxes_ptr[b * 6 + 4] * spatial_scale;
        const scalar_t z2 = boxes_ptr[b * 6 + 5] * spatial_scale;

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const scalar_t height_scale = (crop_height > 1) ? (z2 - z1) / (crop_height - 1) : 0;
        const scalar_t length_scale = (crop_length > 1) ? (y2 - y1) / (crop_length - 1) : 0;
        const scalar_t width_scale = (crop_width > 1) ? (x2 - x1) / (crop_width - 1) : 0;

        const scalar_t in_z = (crop_height > 1) ? z1 + z * height_scale : 0.5 * (z1 + z2);
        if (in_z < 0 || in_z > image_height - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const scalar_t in_y = (crop_length > 1) ? y1 + y * length_scale : 0.5 * (y1 + y2);
        if (in_y < 0 || in_y > image_length - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const scalar_t in_x = (crop_width > 1) ? x1 + x * width_scale : 0.5 * (x1 + x2);
        if (in_x < 0 || in_x > image_width - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const int near_z_index = floor(in_z);
        const int far_z_index = ceil(in_z);
        const scalar_t z_lerp = in_z - near_z_index;

        const int top_y_index = floor(in_y);
        const int bottom_y_index = ceil(in_y);
        const scalar_t y_lerp = in_y - top_y_index;

        const int left_x_index = floor(in_x);
        const int right_x_index = ceil(in_x);
        const scalar_t x_lerp = in_x - left_x_index;

        const scalar_t *pimage = image_ptr + (b_in * depth + d) * image_height * image_width * image_length;
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
        crops_ptr[out_idx] = top + (bottom - top) * y_lerp;
    }
}


template <typename scalar_t>
__global__
void CropAndResizeBackpropImageKernel3D(
    const int nthreads, const scalar_t *grads_ptr, const scalar_t *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length,
    int crop_height, int depth, scalar_t spatial_scale, scalar_t *grads_image_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NCWLH: out_idx = h + crop_height * (l + crop_length * (w + crop_width * (d + depth * b)))
        int idx = out_idx;
        const int z = idx % crop_height;
        idx /= crop_height;
        const int y = idx % crop_length;
        idx /= crop_length;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int d = idx % depth;
        const int b = idx / depth;

        const scalar_t x1 = boxes_ptr[b * 6] * spatial_scale;
        const scalar_t y1 = boxes_ptr[b * 6 + 1] * spatial_scale;
        const scalar_t z1 = boxes_ptr[b * 6 + 2] * spatial_scale;
        const scalar_t x2 = boxes_ptr[b * 6 + 3] * spatial_scale;
        const scalar_t y2 = boxes_ptr[b * 6 + 4] * spatial_scale;
        const scalar_t z2 = boxes_ptr[b * 6 + 5] * spatial_scale;

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const scalar_t height_scale = (crop_height > 1) ? (z2 - z1) / (crop_height - 1) : 0;
        const scalar_t length_scale = (crop_length > 1) ? (y2 - y1) / (crop_length - 1) : 0;
        const scalar_t width_scale = (crop_width > 1) ? (x2 - x1) / (crop_width - 1) : 0;

        const scalar_t in_z = (crop_height > 1) ? z1 + z * height_scale : 0.5 * (z1 + z2);
        if (in_z < 0 || in_z > image_height - 1)
        {
            continue;
        }

        const scalar_t in_y = (crop_length > 1) ? y1 + y * length_scale : 0.5 * (y1 + y2);
        if (in_y < 0 || in_y > image_length - 1)
        {
            continue;
        }

        const scalar_t in_x = (crop_width > 1) ? x1 + x * width_scale : 0.5 * (x1 + x2);
        if (in_x < 0 || in_x > image_width - 1)
        {
            continue;
        }

        const int near_z_index = floor(in_z);
        const int far_z_index = ceil(in_z);
        const scalar_t z_lerp = in_z - near_z_index;

        const int top_y_index = floor(in_y);
        const int bottom_y_index = ceil(in_y);
        const scalar_t y_lerp = in_y - top_y_index;

        const int left_x_index = floor(in_x);
        const int right_x_index = ceil(in_x);
        const scalar_t x_lerp = in_x - left_x_index;

        scalar_t *pimage = grads_image_ptr + (b_in * depth + d) * image_height * image_width * image_length;
        const scalar_t dtop = (1 - y_lerp) * grads_ptr[out_idx];
        const scalar_t dnear_top = (1 - z_lerp) * dtop;
        atomicAdd(
            pimage + near_z_index + image_height * (top_y_index + image_length * left_x_index), 
            (1 - x_lerp) * dnear_top
        );
        atomicAdd(
            pimage + near_z_index + image_height * (top_y_index + image_length * right_x_index),
            x_lerp * dnear_top
        );

        const scalar_t dfar_top = z_lerp * dtop;
        atomicAdd(
            pimage + far_z_index + image_height * (top_y_index + image_length * left_x_index), 
            (1 - x_lerp) * dfar_top
        );
        atomicAdd(
            pimage + far_z_index + image_height * (top_y_index + image_length * right_x_index), 
            x_lerp * dfar_top
        );

        const scalar_t dbottom = y_lerp * grads_ptr[out_idx];
        const scalar_t dnear_bottom = (1 - z_lerp) * dbottom;
        atomicAdd(
            pimage + near_z_index + image_height * (bottom_y_index + image_length * left_x_index), 
            (1 - x_lerp) * dnear_bottom
        );
        atomicAdd(
            pimage + near_z_index + image_height * (bottom_y_index + image_length * right_x_index),
            x_lerp * dnear_bottom
        );

        const scalar_t dfar_bottom = z_lerp * dbottom;
        atomicAdd(
            pimage + far_z_index + image_height * (bottom_y_index + image_length * left_x_index), 
            (1 - x_lerp) * dfar_bottom
        );
        atomicAdd(
            pimage + far_z_index + image_height * (bottom_y_index + image_length * right_x_index), 
            x_lerp * dfar_bottom
        );
    }
}


template <typename scalar_t>
void CropAndResizeLaucher3D(
    const scalar_t *image_ptr, const scalar_t *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length,
    int crop_height, int depth, scalar_t spatial_scale, scalar_t extrapolation_value, 
    scalar_t *crops_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * crop_length * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeKernel3D<scalar_t><<<block_count, thread_per_block, 0, stream>>>(
            total_count, image_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_width, image_length, image_height,
            crop_width, crop_length, crop_height, depth, spatial_scale,
            extrapolation_value, crops_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


template <typename scalar_t>
void CropAndResizeBackpropImageLaucher3D(
    const scalar_t *grads_ptr, const scalar_t *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length, 
    int crop_height, int depth, scalar_t spatial_scale, 
    scalar_t *grads_image_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * crop_length * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeBackpropImageKernel3D<scalar_t><<<block_count, thread_per_block, 0, stream>>>(
            total_count, grads_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_width, image_length, image_height,
            crop_width, crop_length, crop_height, depth, spatial_scale, grads_image_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


void crop_and_resize_3d_cuda_forward(
    torch::Tensor image,
    torch::Tensor boxes,            // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,        // range in [0, batch_size) // int
    const double spatial_scale,
    const double extrapolation_value,
    const int crop_width,
    const int crop_length,
    const int crop_height,
    torch::Tensor crops
) {
    const int batch_size    = image.size(0);
    const int depth         = image.size(1);
    const int image_width   = image.size(2);
    const int image_length  = image.size(3);
    const int image_height  = image.size(4);

    const int num_boxes     = boxes.size(0);

    crops.resize_({num_boxes, depth, crop_width, crop_length, crop_height});
    crops.zero_();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();// THCState_getCurrentStream(state);

    AT_DISPATCH_FLOATING_TYPES(image.type(), "crop_and_resize_3d_cuda_forward", ([&] {
        CropAndResizeLaucher3D<scalar_t>(
            image.data<scalar_t>(),
            boxes.data<scalar_t>(),
            box_index.data<int>(),
            num_boxes, batch_size, image_width, image_length, image_height,
            crop_width, crop_length, crop_height, depth,
            spatial_scale, extrapolation_value,
            crops.data<scalar_t>(),
            stream
        );
    }));
}


void crop_and_resize_3d_cuda_backward(
    torch::Tensor grads,
    torch::Tensor boxes,      // [x1, y1, z1, x2, y2, z2]
    torch::Tensor box_index,    // range in [0, batch_size) // int
    const double spatial_scale,
    torch::Tensor grads_image // resize to [bsize, c, wc, lc, hc]
) {
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

    AT_DISPATCH_FLOATING_TYPES(grads.type(), "crop_and_resize_3d_cuda_backward", ([&] {
        CropAndResizeBackpropImageLaucher3D<scalar_t>(
            grads.data<scalar_t>(),
            boxes.data<scalar_t>(),
            box_index.data<int>(),
            num_boxes, batch_size, image_width, image_length, image_height,
            crop_width, crop_length, crop_height, depth, spatial_scale,
            grads_image.data<scalar_t>(),
            stream
        );
    }));
}
