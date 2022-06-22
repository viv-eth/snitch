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
    uint16_t padded_image_size =  dim_padded_x * dim_padded_y;
    uint16_t image_size = dim_in_x * dim_in_y;


    // INFO FP64 cluster memory setup
    void *ptr = (double *)snrt_cluster_memory().start;
    double *image = ptr;
    ptr += image_size * n->dtype;
    double *padded_image = ptr;
    ptr += padded_image_size * n->dtype;

    // load the image from DRAM into the cluster memory
    if (snrt_is_dm_core() && cluster_id == 0) {
        snrt_dma_start_tracking();
                snrt_dma_txid_t txid_IMG = 
                    snrt_dma_start_1d(image,                                   // destination
                                    n->image,                                  // source
                                    n->dtype * padded_image_size);                    // size

                // wait until each DMA transfer done
                snrt_dma_wait_all();
        snrt_dma_stop_tracking();

    }

    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num && cluster_id == 0) {
        switch(n->dtype){
            case FP64:
                if(!compute_id) {
                    printf("Start CNN\n");
                    pad_image(image, padded_image, n->H, n->W, n->padding);
                    printf("Test image padding: \n");
                    for(int i = 0; i < dim_padded_x; i++) {
                        printf("|");
                        for(int j = 0; j < dim_padded_y; j++) {
                            printf("%.2f|", padded_image[i*dim_padded_y + j]);
                        }
                        printf("\n");
                    }

                    printf("End CNN\n");
                }
                break;

            default:
                if(!compute_id){
                        printf("ERROR: unsupported data type\n");
                }
                break;
        }
    } 

}