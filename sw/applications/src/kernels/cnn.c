// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "cnn.h"
#include "conv2d.h"
#include "network.h"
#include "printf.h"


void pad_image(double *image, double *padded_image, uint16_t H, uint16_t W, uint16_t padding) {

    // get the image dimensions
    uint16_t dim_in_x = H;
    uint16_t dim_in_y = W;
    // printf("Original Image dimensions: %u x %u\n", dim_in_x, dim_in_y);
    // INFO: print the original image
    // for(int i = 0; i < dim_in_x; i++) {
    //     printf("|");
    //     for(int j = 0; j < dim_in_y; j++) {
    //         printf("%.2f|", image[i*dim_in_y + j]);
    //     }
    //     printf("\n");
    // }
    // determine the dimensions of the padded image
    // we assume symmetric padding
    uint16_t padding_x_left = padding;
    uint16_t padding_x_right = padding;
    uint16_t padding_y_top = padding;
    uint16_t padding_y_bottom = padding;

    uint16_t dim_padded_x = dim_in_x + padding_x_left + padding_x_right;
    uint16_t dim_padded_y = dim_in_y + padding_y_top + padding_y_bottom;
    // printf("Padded Image dimensions: %d x %d\n", dim_padded_x, dim_padded_y);
    // pad the image
    for (uint16_t y = 0; y < dim_padded_y; y++) {
        for (uint16_t x = 0; x < dim_padded_x; x++) {
            if (y < padding_y_top || y >= dim_in_y + padding_y_top || x < padding_x_left || x >= dim_in_x + padding_x_left) {
                padded_image[y * dim_padded_x + x] = 0;
            } else {
                padded_image[y * dim_padded_x + x] = image[(y - padding_y_top) * dim_in_x + (x - padding_x_left)];
            }
        }
    }

    // INFO: print the padded image
    // for(int i = 0; i < dim_padded_x; i++) {
    //     printf("|");
    //     for(int j = 0; j < dim_padded_y; j++) {
    //         printf("%.2f|", t_padded_image[i*dim_padded_y + j]);
    //         padded_image[i*dim_padded_y + j] = t_padded_image[i*dim_padded_y + j];
    //     }
    //     printf("\n");
    // }
}

// void conv2d_fp64(kernel_fp64* k){
//     // Input feature map (Ci x H x W) --> for MNIST: 1 x 28 x 28
//     // First layer: kernel (Ci x Kh x Kw) --> for MNIST: 1 x 5 x 5 (1 input channel, 5x5 kernel)
//     // Output feature map of the first layer: (Co x H x W) --> for MNIST: 16 x 28 x 28 (16 output channels)
//     // Weights of the first layer: (Co x Ci x Kh x Kw) --> for MNIST: 16 x 1 x 5 x 5 (16 output channels, 1 input channel, 5x5 kernel)
//     // Bias of the first layer: (Co) --> for MNIST: 16 (16 output channels)

//     // Parallelization/Pipelining parameters
//     const uint32_t compute_id = snrt_cluster_compute_core_idx();
//     const uint32_t compute_num =
//         (snrt_cluster_compute_core_num()) ? snrt_cluster_compute_core_num() : 1;
// }