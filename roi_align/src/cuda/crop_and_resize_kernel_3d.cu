#include <math.h>
#include <stdio.h>
#include "crop_and_resize_kernel_3d.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)


__global__
void CropAndResizeKernel3D(
    const int nthreads, const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length,
    int crop_height, int depth, float extrapolation_value, float *crops_ptr)
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

        const float x1 = boxes_ptr[b * 6];
        const float y1 = boxes_ptr[b * 6 + 1];
        const float z1 = boxes_ptr[b * 6 + 2];
        const float x2 = boxes_ptr[b * 6 + 3];
        const float y2 = boxes_ptr[b * 6 + 4];
        const float z2 = boxes_ptr[b * 6 + 5];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale = (crop_height > 1) ? (z2 - z1) / (crop_height - 1) : 0;
        const float length_scale = (crop_length > 1) ? (y2 - y1) / (crop_length - 1) : 0;
        const float width_scale = (crop_width > 1) ? (x2 - x1) / (crop_width - 1) : 0;

        const float in_z = (crop_height > 1) ? z1 + z * height_scale : 0.5 * (z1 + z2);
        if (in_z < 0 || in_z > image_height - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const float in_y = (crop_length > 1) ? y1 + y * length_scale : 0.5 * (y1 + y2);
        if (in_y < 0 || in_y > image_length - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const float in_x = (crop_width > 1) ? x1 + x * width_scale : 0.5 * (x1 + x2);
        if (in_x < 0 || in_x > image_width - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const int near_z_index = floorf(in_z);
        const int far_z_index = ceilf(in_z);
        const float z_lerp = in_z - near_z_index;

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        const float *pimage = image_ptr + (b_in * depth + d) * image_height * image_width * image_length;
        const float near_top_left = pimage[near_z_index + image_height * (top_y_index + image_length * left_x_index)];
        const float near_top_right = pimage[near_z_index + image_height * (top_y_index + image_length * right_x_index)];
        const float near_bottom_left = pimage[near_z_index + image_height * (bottom_y_index + image_length * left_x_index)];
        const float near_bottom_right = pimage[near_z_index + image_height * (bottom_y_index + image_length * right_x_index)];

        const float far_top_left = pimage[far_z_index + image_height * (top_y_index + image_length * left_x_index)];
        const float far_top_right = pimage[far_z_index + image_height * (top_y_index + image_length * right_x_index)];
        const float far_bottom_left = pimage[far_z_index + image_height * (bottom_y_index + image_length * left_x_index)];
        const float far_bottom_right = pimage[far_z_index + image_height * (bottom_y_index + image_length * right_x_index)];

        const float near_top = near_top_left + (near_top_right - near_top_left) * x_lerp;
        const float near_bottom = near_bottom_left + (near_bottom_right - near_bottom_left) * x_lerp;

        const float far_top = far_top_left + (far_top_right - far_top_left) * x_lerp;
        const float far_bottom = far_bottom_left + (far_bottom_right - far_bottom_left) * x_lerp;

        const float top = near_top + (far_top - near_top) * z_lerp;
        const float bottom = near_bottom + (far_bottom - near_bottom) * z_lerp;
        crops_ptr[out_idx] = top + (bottom - top) * y_lerp;
    }
}

__global__
void CropAndResizeBackpropImageKernel3D(
    const int nthreads, const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length,
    int crop_height, int depth, float *grads_image_ptr)
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

        const float x1 = boxes_ptr[b * 6];
        const float y1 = boxes_ptr[b * 6 + 1];
        const float z1 = boxes_ptr[b * 6 + 2];
        const float x2 = boxes_ptr[b * 6 + 3];
        const float y2 = boxes_ptr[b * 6 + 4];
        const float z2 = boxes_ptr[b * 6 + 5];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale = (crop_height > 1) ? (z2 - z1) / (crop_height - 1) : 0;
        const float length_scale = (crop_length > 1) ? (y2 - y1) / (crop_length - 1) : 0;
        const float width_scale = (crop_width > 1) ? (x2 - x1) / (crop_width - 1) : 0;

        const float in_z = (crop_height > 1) ? z1 + z * height_scale : 0.5 * (z1 + z2);
        if (in_z < 0 || in_z > image_height - 1)
        {
            continue;
        }

        const float in_y = (crop_length > 1) ? y1 + y * length_scale : 0.5 * (y1 + y2);
        if (in_y < 0 || in_y > image_length - 1)
        {
            continue;
        }

        const float in_x = (crop_width > 1) ? x1 + x * width_scale : 0.5 * (x1 + x2);
        if (in_x < 0 || in_x > image_width - 1)
        {
            continue;
        }

        const int near_z_index = floorf(in_z);
        const int far_z_index = ceilf(in_z);
        const float z_lerp = in_z - near_z_index;

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        float *pimage = grads_image_ptr + (b_in * depth + d) * image_height * image_width * image_length;
        const float dtop = (1 - y_lerp) * grads_ptr[out_idx];
        const float dnear_top = (1 - z_lerp) * dtop;
        atomicAdd(
            pimage + near_z_index + image_height * (top_y_index + image_length * left_x_index), 
            (1 - x_lerp) * dnear_top
        );
        atomicAdd(
            pimage + near_z_index + image_height * (top_y_index + image_length * right_x_index),
            x_lerp * dnear_top
        );

        const float dfar_top = z_lerp * dtop;
        atomicAdd(
            pimage + far_z_index + image_height * (top_y_index + image_length * left_x_index), 
            (1 - x_lerp) * dfar_top
        );
        atomicAdd(
            pimage + far_z_index + image_height * (top_y_index + image_length * right_x_index), 
            x_lerp * dfar_top
        );

        const float dbottom = y_lerp * grads_ptr[out_idx];
        const float dnear_bottom = (1 - z_lerp) * dbottom;
        atomicAdd(
            pimage + near_z_index + image_height * (bottom_y_index + image_length * left_x_index), 
            (1 - x_lerp) * dnear_bottom
        );
        atomicAdd(
            pimage + near_z_index + image_height * (bottom_y_index + image_length * right_x_index),
            x_lerp * dnear_bottom
        );

        const float dfar_bottom = z_lerp * dbottom;
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


void CropAndResizeLaucher3D(
    const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length,
    int crop_height, int depth, float extrapolation_value, 
    float *crops_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * crop_length * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeKernel3D<<<block_count, thread_per_block, 0, stream>>>(
            total_count, image_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_width, image_length, image_height,
            crop_width, crop_length, crop_height, depth, extrapolation_value, crops_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


void CropAndResizeBackpropImageLaucher3D(
    const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_width,
    int image_length, int image_height, int crop_width, int crop_length, 
    int crop_height, int depth, float *grads_image_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * crop_length * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeBackpropImageKernel3D<<<block_count, thread_per_block, 0, stream>>>(
            total_count, grads_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_width, image_length, image_height,
            crop_width, crop_length, crop_height, depth, grads_image_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}
