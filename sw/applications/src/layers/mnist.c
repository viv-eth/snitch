// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist.h"

#include "network.h"
#include "mnist_network.h"
#include "printf.h"
#include "snrt.h"
#include "mnist_data.h"
#include "utils.h"


// Padding in between matrices for preventing
// banking conflicts in the beginning
#define MAT_PADDING 8

#define MAT_ROW_PADDING 4

void mnist(const network_t *n){

    uint32_t cluster_num = snrt_cluster_num(); // returns 2
    //printf("cluster_num = %u\n", cluster_num);
    uint32_t cluster_core_num = snrt_cluster_core_num(); // returns 9
    //printf("cluster_core_num = %u\n", cluster_core_num);
    uint32_t cluster_id = snrt_cluster_idx();
    //printf("cluster_id = %u\n", cluster_id);
    uint32_t compute_num = snrt_cluster_compute_core_num(); // Number of compute cores per cluster 
    uint32_t global_compute_num = snrt_global_core_num(); // Total cores incl. DM core per cluster 
    //printf("snrt_global_core_num = %u\n", global_compute_num);
    uint32_t compute_id = snrt_cluster_compute_core_idx();
    uint32_t dm_id = snrt_cluster_dm_core_idx();
    uint32_t global_compute_id = snrt_global_core_idx(); // Core ID of each core on all clusters

    // number of total input channels for a flattened image
    uint32_t IN_CH = n->IN_CH1*n->IN_CH2;

    // number of total images that we load
    // NOTE: partial load for simulation purposes only
    uint32_t number_of_images = 5;

    // size of the weight matrix, same for weight gradients
    // on cluster 0 we store the weights, on cluster 1 the
    // weight gradients at this location
    uint32_t weight_mat_size = (n->OUT_CH * IN_CH + MAT_PADDING) * n->dtype;
    // size of the bias matrix, same for bias gradients
    // on cluster 0 we store the biases, on cluster 1 the
    // bias gradients at this location
    uint32_t bias_mat_size = n->OUT_CH * n->dtype;
    // size of activations (same as biases), which we need to synchronize the clusters
    uint32_t act_mat_size = bias_mat_size;
    // size of a single MNIST image (28x28 = 784 pixels)
    uint32_t image_size = IN_CH * sizeof(double);
    // size for storing the maximum on each core of a cluster (only used on cluster 0)
    uint32_t max_size = n->dtype;//compute_num * n->dtype;
    // size of the target for image classification (0...9)
    uint32_t target_size = sizeof(uint32_t);
    // result of the cross entropy loss calculation
    uint32_t loss_size = n->dtype;
    // synchronization flags for the compute cores on among clusters
    uint32_t core_sync_flag_size = compute_num*sizeof(uint32_t);
    // learning rate of the network
    uint32_t lr_size = sizeof(float);

    // INFO FP64 cluster memory setup
    // @brief Cluster Memory Structure for each cluster to ensure
    // we can access the data of both by using the constant
    // cluster base offset
    // void *ptr = (double *)snrt_cluster_memory().start;
    // // void *ptr_start = ptr;
    // double *max= ptr; // zero initialized
    // ptr += max_size;
    // double *loss = ptr; // zero initialized
    // ptr += loss_size;
    // double *images = ptr;
    // ptr += number_of_images*image_size;
    // double *biases = ptr; // bias GRADIENTS zero initialized
    // ptr += bias_mat_size;
    // double *activations = ptr;
    // ptr += act_mat_size;
    // double *weights = ptr; // weight GRADIENTS zero initialized
    // ptr += weight_mat_size;
    // // NOTE: core sync flag used to indictae whether computation is done or not
    // uint32_t *core_sync = ptr; // zero initialized
    // ptr += core_sync_flag_size;
    // uint32_t *targets = ptr;
    // ptr += number_of_images*target_size;
    // NOTE: following lines for debugging purposes only
    // void *ptr_end = (double *)snrt_cluster_memory().end;
    // if(compute_id == 0){   
    //     printf("Start address of cluster %u memory: 0x%p\n", cluster_id, ptr_start);
    //     printf("End address of cluster %u memory: 0x%p\n", cluster_id, ptr_end);
    //     printf("Available memory on cluster %u: %u KB\n", cluster_id, (ptr_end - ptr_start) / 1000);
    //     printf("Total cluster memory occupation on cluster %u: %u KB\n", cluster_id, (ptr - ptr_start) / 1000);
    // }

    // // INFO FP32 cluster memory setup
    // // images remain in double precision
    void *ptr = (float *)snrt_cluster_memory().start;
    float *max= ptr; // zero initialized
    ptr += max_size;
    float *loss = ptr; // zero initialized
    ptr += loss_size;
    float *images = ptr;
    ptr += number_of_images*image_size;
    float *biases = ptr; // bias GRADIENTS zero initialized
    ptr += bias_mat_size;
    float *activations = ptr;
    ptr += act_mat_size;
    // INFO: setting weights as last element so 
    // when we iterate over data we can zero out
    // excessive rows --> FIXME: this does not work in the RTL probably
    float *weights = ptr; // weight GRADIENTS zero initialized
    ptr += weight_mat_size;
    // NOTE: core sync flag used to indictae whether computation is done or not
    uint32_t *core_sync = ptr; // zero initialized
    ptr += core_sync_flag_size;
    uint32_t *targets = ptr;
    ptr += number_of_images*target_size;
    // NOTE: following lines for debugging purposes only
    // void *ptr_end = (double *)snrt_cluster_memory().end;
    // if(compute_id == 0){   
    //     printf("Start address of cluster %u memory: 0x%p\n", cluster_id, ptr_start);
    //     printf("End address of cluster %u memory: 0x%p\n", cluster_id, ptr_end);
    //     printf("Available memory on cluster %u: %u KB\n", cluster_id, (ptr_end - ptr_start) / 1000);
    //     printf("Total cluster memory occupation on cluster %u: %u KB\n", cluster_id, (ptr - ptr_start) / 1000);
    // }

    // cluster offset in an Occamy quadrant
    uint32_t cluster_offset = 0x00040000;

    // each cluster should set their sync flag to zero initially
    core_sync[compute_id] = 0;

    // define whether SSRs should be used or not
    uint32_t setup_SSR = 1;

    // We load the GM weights and biases into cluster 0 memory 
    // together with the image data.
    // TODO: add the epochs
    if (snrt_is_dm_core() && cluster_id == 0) {
                // load initial biases from Golden Model into Cluster 0 memory
                snrt_dma_txid_t txid_B = 
                    snrt_dma_start_1d(biases,                 // destination
                                    n->b,                     // source
                                    n->dtype * n->OUT_CH);    // size

                // load weight data into Cluster 0 memory
                // TODO: make this 1D DMA transfer
                snrt_dma_txid_t txid_W = 
                    snrt_dma_start_2d(weights,                // destination
                                    n->W,                     // source
                                    n->dtype * IN_CH,         // size
                                    n->dtype * IN_CH ,        // destination stride
                                    n->dtype * IN_CH ,        // source stride
                                    n->OUT_CH);               // repetitions

                snrt_dma_txid_t txid_IMG = 
                    snrt_dma_start_1d(images,                                   // destination
                                    n->images,                                  // source
                                    n->dtype * number_of_images * IN_CH);  // size

                // wait until each DMA transfer done
                snrt_dma_wait_all();

    }

    //snrt_cluster_hw_barrier();

    if (snrt_is_dm_core() && cluster_id == 1) {
        // On cluster 1 we load the labels which are needed for BP
                snrt_dma_txid_t txid_targets = 
                    snrt_dma_start_1d(targets,                                   // destination
                                    n->targets,                                  // source
                                    sizeof(uint32_t) * number_of_images);        // size
                
                snrt_dma_wait_all();

    }

    snrt_cluster_hw_barrier();
    // Global memory access
    // uint32_t *global_mem = (void *)snrt_global_memory().start;
    
    // DRAM dataset memory start address
    // NOTE: this is only needed when preloading the DRAM in banshee
    // uint32_t *dataset_dram = (void *)0x8004000;

    // We now loop through the images
    for(uint32_t image = 0; image < 1; image++){

        // if(!compute_id){
        //     for(uint32_t in = 0; in < 784; in++){
        //         printf("image[%u] = %f\n", in, images[in]);
        //     }
        // } --> this prints the correct image

        // we calculate the pointer postion of the current image
        uint32_t curr_img = image * IN_CH;
        // we perform the forward pass on cluster 0
        if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num && cluster_id == 0) {
            // determine the row offset at which current compute cluster is
            volatile uint32_t W_offset = compute_id * IN_CH;
            volatile uint32_t b_offset = compute_id;
            // Calculate number of rows for each compute
            // core. If multiples of each other we have to 
            // forcefully set it to 1
            volatile uint32_t div = n->OUT_CH % compute_num;
            if(div == 0){
                div = 1;
            }

            //printf("weights[%u] = %f\n", W_offset, weights[W_offset]);
            // printf("biases[%u] = %f\n", b_offset, biases[b_offset]);
            // printf("images[%u] = %f\n", curr_img, images[curr_img]);


            // determine the row stride of each matrix    
            volatile uint32_t ldW = compute_num * IN_CH;
            volatile uint32_t ldB = compute_num;
            volatile uint32_t ldI = IN_CH;

            //printf("ldW: %u, ldB: %u, ldI: %u\n", ldW, ldB, ldI);

            if(!compute_id){
                printf("FF start\n");
            }

            // Start of feedforward
            benchmark_get_cycle();
            // INFO: baseline
            // feedforward_fp64(n->IN_CH1, n->IN_CH2, div, 
            //                 &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
            //                 ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id]);

            // INFO: FP64 with SSRs
            // feedforward_fp64_ssr(n->IN_CH1, n->IN_CH2, div, 
            //                 &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
            //                 ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id],
            //                 setup_SSR);

            // INFO: FP32 with SSRs and SIMD
            feedforward_fp32_ssr_simd(n->IN_CH1, n->IN_CH2, div, 
                            &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
                            ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id],
                            setup_SSR); 

            // INFO: FP32 baseline
            // feedforward_fp32(n->IN_CH1, n->IN_CH2, div, 
            //                 &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
            //                 ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id]);


            // INFO: FP64 baseline
            // softmax_activation_fp64(n->IN_CH1, n->IN_CH2, div, 
            //                 &weights[W_offset], ldW, &activations[b_offset], ldB,
            //                 &images[curr_img], ldI, compute_id, compute_num, max, &core_sync);

            // INFO: FP64 with SSRs
            // softmax_activation_fp64_ssr(n->IN_CH1, n->IN_CH2, div, 
            //                 &weights[W_offset], ldW, &activations[b_offset], ldB,
            //                 &images[curr_img], ldI, compute_id, compute_num, max, &core_sync, setup_SSR);

            softmax_activation_fp32(n->IN_CH1, n->IN_CH2, div, 
                            &weights[W_offset], ldW, &activations[b_offset], ldB,
                            &images[curr_img], ldI, compute_id, compute_num, max, &core_sync, setup_SSR);
            benchmark_get_cycle();

            if(!compute_id){
                printf("FF end\n");
            }

        } else {
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
        }

        // wait until clusters are synchronized to not
        // start gradient update until all activations are 
        // computed
        snrt_global_barrier();

        if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num && cluster_id == 1) {
            // determine the row offset at which current compute cluster is
            volatile uint32_t W_offset = compute_id * IN_CH;
            volatile uint32_t b_offset = compute_id;
            // Calculate number of rows for each compute
            // core. If multiples of each other we have to 
            // forcefully set it to 1
            volatile uint32_t div = n->OUT_CH % compute_num;
            if(div == 0){
                div = 1;
            }


            // determine the row stride of each matrix    
            volatile uint32_t ldW = compute_num * IN_CH;
            volatile uint32_t ldB = compute_num;
            volatile uint32_t ldI = IN_CH;

            uint32_t *act_ptr = ((uint32_t)activations) - cluster_offset;

            uint32_t *img_ptr = ((uint32_t)images) - cluster_offset;

            //double *core_sync_ptr = ((uint32_t)core_sync) - cluster_offset;
            //printf("activations[%u] = %f\n", b_offset, act_ptr[b_offset]);

            if(!compute_id){
                printf("Gradient Update start\n");
            }

            benchmark_get_cycle();
            // INFO: baseline
            // gradient_update_fp64(n->IN_CH1, n->IN_CH2, div, 
            //                 &weights[W_offset], ldW, 
            //                 &biases[b_offset], &act_ptr[b_offset], 
            //                 ldB, &img_ptr[curr_img], &targets[curr_img], ldI, compute_id, 
            //                 loss, compute_num);

            // INFO: FP64 with SSRs
            // gradient_update_fp64_ssr(n->IN_CH1, n->IN_CH2, div, 
            //                 &weights[W_offset], ldW, 
            //                 &biases[b_offset], &act_ptr[b_offset], 
            //                 ldB, &img_ptr[curr_img], &targets[curr_img], ldI, compute_id, 
            //                 loss, compute_num, setup_SSR);

            // INFO: FP32 with SSRs and SIMD
            gradient_update_fp32_ssr_simd(n->IN_CH1, n->IN_CH2, div, 
                            &weights[W_offset], ldW, 
                            &biases[b_offset], &act_ptr[b_offset], 
                            ldB, &img_ptr[curr_img], &targets[curr_img], ldI, compute_id, 
                            loss, compute_num, setup_SSR);

            // INFO: FP32 baseline
            // gradient_update_fp32(n->IN_CH1, n->IN_CH2, div, 
            //                 &weights[W_offset], ldW, 
            //                 &biases[b_offset], &act_ptr[b_offset], 
            //                 ldB, &img_ptr[curr_img], &targets[curr_img], ldI, compute_id, 
            //                 loss, compute_num);
            benchmark_get_cycle();

            if(!compute_id){
                printf("Gradient Update done\n");
            }

            if(!compute_id){
                printf("total loss = %f\n", loss[0]/(image+1));
            }

        } else {
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
        }
    }

    //snrt_global_barrier();

    // after looping through one batch of the dataset
    // we update the biases and weights on Cluster 0
    // if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num && cluster_id == 0) {
    //     // determine the row offset at which current compute cluster is
    //     volatile uint32_t W_offset = compute_id * IN_CH;
    //     volatile uint32_t b_offset = compute_id;
    //     // Calculate number of rows for each compute
    //     // core. If multiples of each other we have to 
    //     // forcefully set it to 1
    //     volatile uint32_t div = n->OUT_CH % compute_num;
    //     if(div == 0){
    //         div = 1;
    //     }


    //     // determine the row stride of each matrix    
    //     volatile uint32_t ldW = compute_num * IN_CH;
    //     volatile uint32_t ldB = compute_num;
    //     volatile uint32_t ldI = IN_CH;

    //     double *weight_grad_ptr = ((uint32_t)weights) + cluster_offset;
    //     double *bias_grad_ptr = ((uint32_t)biases) + cluster_offset;

    //     //TODO: load the LR from the network struct or via DRAM perloading
    //     //*learning_rate = 0.5;

    //     // if(!compute_id){
    //     //         printf("Training step start\n");
    //     // }

    //     benchmark_get_cycle();
    //     // INFO: baseline
    //     training_step_fp64(n->IN_CH1, n->IN_CH2, div, 
    //             &weights[W_offset], &weight_grad_ptr[W_offset], ldW, 
    //             &biases[b_offset], &bias_grad_ptr[b_offset], ldB, 
    //             compute_id, compute_num, number_of_images);
    //     // INFO: FP64 with SSRs
    //     // training_step_fp64_ssr(n->IN_CH1, n->IN_CH2, div, 
    //     //         &weights[W_offset], &weight_grad_ptr[W_offset], ldW, 
    //     //         &biases[b_offset], &bias_grad_ptr[b_offset], ldB, 
    //     //         compute_id, compute_num, number_of_images, setup_SSR);
    //     benchmark_get_cycle();

    //     // if(!compute_id){
    //     //         printf("Training step done\n");
    //     // }

    // }
}
