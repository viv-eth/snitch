// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "cnn.h"
#include "conv2d.h"
#include "network.h"
#include "printf.h"
#include "math.h"


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

void conv2d_fp64(double *padded_image, double *weights, double *biases, uint16_t bias_stride, uint16_t ci, uint16_t co, 
                uint16_t H, uint16_t W, uint16_t K, uint16_t padding, uint16_t stride, 
                uint16_t dim_out_x, uint16_t dim_out_y, uint16_t weight_row_stride, double *output, uint16_t output_stride) {
    // Input feature map (Ci x H x W) --> for MNIST: 1 x 28 x 28
    // First layer: kernel (Ci x Kh x Kw) --> for MNIST: 1 x 5 x 5 (1 input channel, 5x5 kernel)
    // Output feature map of the first layer: (Co x H x W) --> for MNIST: 16 x 28 x 28 (16 output channels)
    // Weights of the first layer: (Co x Ci x Kh x Kw) --> for MNIST: 16 x 1 x 5 x 5 (16 output channels, 1 input channel, 5x5 kernel)
    // Biases of the first layer: (Co) --> for MNIST: 16 (16 output channels)
    // Output feature map of the first layer: (Co x H x W) --> for MNIST: 16 x 28 x 28 (16 output channels)

    // accumulator for convolution of the input feature map with the kernel
    double total = 0;
    double weight;
    double data;
    // accumulator for kernel indices
    double kt = 0;
    // image indices
    uint16_t pos_x = 0;
    uint16_t pos_y = 0;

    // INFO: parallelize over the output channels
    for (uint16_t co_idx = 0; co_idx < co; co_idx++) {
        for(uint16_t out_w = 0; out_w < dim_out_y; out_w++) {
            for(uint16_t out_h = 0; out_h < dim_out_x; out_h++) {
                total = 0;
                for(uint16_t ci_idx = 0; ci_idx < ci; ci_idx++) {
                    kt = 0;
                    for(uint16_t kh_idx = 0; kh_idx < K; kh_idx++) {
                        for(uint16_t kw_idx = 0; kw_idx < K; kw_idx++) {
                            weight = weights[kh_idx * K + kw_idx + co_idx * weight_row_stride];
                            pos_x = out_h * stride + kh_idx;
                            pos_y = out_w * stride + kw_idx;
                            data = padded_image[ci_idx * H * W + pos_y * W + pos_x];
                            kt += weight * data;
                            // if(data != 0) {
                            //     printf("data[%u][%u] = %.4f\n", pos_y, pos_x, data);
                            //     printf("weight[%u][%u] = %.4f\n", kh_idx, kw_idx, weight);
                            //     printf("kt = %.4f\n", kt);
                            // } // WORKS
                        }

                        total += kt;
                        // if(total != 0) {
                        //     printf("total[%u][%u] = %.4f\n", out_h, out_w, total);
                        // } // WORKS
                    }

                }

                // output[co_idx * 8 + out_w + out_h] = total + biases[co_idx];
                // FIXME: causes out of mem access
                output[co_idx * output_stride + out_w + out_h] = total + biases[co_idx * bias_stride];
            }
        }
    }


    snrt_cluster_hw_barrier();             

                
}