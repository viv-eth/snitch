// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_cnn.h"

#include "network.h"
#include "cnn.h"
#include "layer.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

void mnist_cnn(const cnn_t *n) {
    
    uint32_t cluster_num = snrt_cluster_num();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_num = snrt_cluster_compute_core_num();
    uint32_t compute_id = snrt_cluster_compute_core_idx();


    // determine the image size including padding
    // we assume symmetric padding
    uint16_t padding_x_left = n->padding;
    uint16_t padding_x_right = n->padding;
    uint16_t padding_y_top = n->padding;
    uint16_t padding_y_bottom = n->padding;

    uint16_t dim_in_x = n->H;
    uint16_t dim_in_y = n->W;
    uint16_t dim_padded_x = dim_in_x + padding_x_left + padding_x_right;
    uint16_t dim_padded_y = dim_in_y + padding_y_top + padding_y_bottom;
    uint16_t padded_image_size =  dim_padded_x * dim_padded_y * n->dtype;

    // determine the dimensions of the output feature map
    uint16_t dim_out_x = floor((n->H - n->K + 2 * n->padding) / n->stride + 1);
    uint16_t dim_out_y = floor((n->W - n->K + 2 * n->padding) / n->stride + 1);
   
    // determine memory size occupation of the CNN
    uint16_t image_size = dim_in_x * dim_in_y * n->dtype;
    uint16_t conv1_weights_size = n->CO * n->CI * n->K * n->K * n->dtype;
    uint16_t conv1_biases_size = n->CO * n->dtype;
    uint16_t conv1_output_size = n->CO * dim_out_x * dim_out_y * n->dtype;



    // INFO FP64 cluster memory setup
    void *ptr = (double *)snrt_cluster_memory().start;
    void *ptr_start = ptr;
    double *image = ptr;
    ptr += image_size;
    double *padded_image = ptr;
    ptr += padded_image_size;
    double *conv1_weights = ptr;
    ptr += conv1_weights_size;
    double *conv1_biases = ptr;
    ptr += conv1_biases_size;
    double *conv1_output = ptr;
    ptr += conv1_output_size;
    // NOTE: following lines for debugging purposes only
    // void *ptr_end = (double *)snrt_cluster_memory().end;
    // if(compute_id == 0){    
    //     printf("Start address of cluster %u memory: 0x%p\n", cluster_id, ptr_start);
    //     printf("End address of cluster %u memory: 0x%p\n", cluster_id, ptr_end);
    //     printf("Available memory on cluster %u: %u KB\n", cluster_id, (ptr_end - ptr_start) / 1000);
    //     printf("Total cluster memory occupation on cluster %u: %u KB\n", cluster_id, (ptr - ptr_start) / 1000);
    // }

    // load the CONV2D parameters from DRAM into the cluster memory
    if (snrt_is_dm_core() && cluster_id == 0) {
        snrt_dma_start_tracking();
                // load image from DRAM into cluster memory 
                snrt_dma_txid_t txid_IMG = 
                    snrt_dma_start_1d(image,                                   // destination
                                    n->image,                                  // source
                                    image_size);                               // size

                // load conv1_weights from DRAM into cluster memory
                snrt_dma_txid_t txid_W = 
                    snrt_dma_start_1d(conv1_weights,                           // destination
                                    n->conv1_weights,                          // source
                                    conv1_weights_size);                       // size

                // load conv1_biases from DRAM into cluster memory
                snrt_dma_txid_t txid_B =
                    snrt_dma_start_1d(conv1_biases,                            // destination
                                    n->conv1_biases,                           // source
                                    conv1_biases_size);             // size

                // wait until each DMA transfer done
                snrt_dma_wait_all();
        snrt_dma_stop_tracking();

    }

    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num && cluster_id == 0) {
        switch(n->dtype){
            case FP64:
                //TODO: parallelize padding over all cores in the cluster
                if(!compute_id) {
                    printf("Start Image Padding\n");
                    pad_image(image, padded_image, n->H, n->W, n->padding);
                    // printf("Test image padding: \n");
                    // for(int i = 0; i < dim_padded_x; i++) {
                    //     printf("|");
                    //     for(int j = 0; j < dim_padded_y; j++) {
                    //         printf("%.2f|", padded_image[i*dim_padded_y + j]);
                    //     }
                    //     printf("\n");
                    // }

                    printf("End Image Padding\n");
                }

                snrt_cluster_hw_barrier();

                printf("Start Convolution fused with ReLu\n");
                // determine the kernel offset for the current core
                uint16_t kernel_offset = n->K * n->K * compute_id;
                // determine the row stride in the weights matrix
                uint16_t weight_row_stride = n->K * n->K * compute_num;
                // determine the bias offset for the current core
                uint16_t bias_offset = compute_id;
                // determine the bias stride
                uint16_t bias_stride = compute_num;
                // determine the output stride in the output matrix
                uint16_t output_stride = dim_out_x * dim_out_y * compute_num;
                // determine the output offset for the current core
                uint16_t output_offset = dim_out_x * dim_out_y * compute_id;

                // printf("weight_row_stride: %d\n", weight_row_stride); --> 200
                // printf("bias_stride: %d\n", bias_stride); --> 8
                // printf("output_stride: %d\n", output_stride); --> 6272

                // printf("bias offset: %d\n", bias_offset);
                // printf("kernel offset: %d\n", kernel_offset);
                // printf("output offset: %d\n", output_offset);
                // printf("conv1_weights[%u]: %f\n", kernel_offset, conv1_weights[kernel_offset]);
                // printf("conv1_biases[%u]: %f\n", bias_offset, conv1_biases[bias_offset]);

                conv2d_relu_fused_fp64(padded_image, &conv1_weights[kernel_offset], &conv1_biases[bias_offset], bias_stride, 
                                    n->CI, n->CO / compute_num, n->H, n->W, n->K, n->padding, n->stride,
                                    dim_out_x, dim_out_y, weight_row_stride, &conv1_output[output_offset], output_stride);
                printf("End Convolution fused with ReLu\n");

                break;

            default:
                if(!compute_id){
                        printf("ERROR: unsupported data type\n");
                }
                break;
        }
    } else if(!snrt_is_compute_core() && cluster_id == 0) {
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
    }

}