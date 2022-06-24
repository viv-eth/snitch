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
#define MAT_PADDING 0
#define MAT_ROW_PADDING 0

// define whether to run baseline network or not
#define BASELINE 0

// define which parts of the network to run
#define RUN_FEEDFORWARD 1
#define RUN_GRADIENT_UPDATE 0
#define RUN_TRAINING_STEP 0

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
    uint32_t number_of_images = 1;

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
    void *ptr = (double *)snrt_cluster_memory().start;
    // void *ptr_start = ptr;
    double *max= ptr; // zero initialized
    ptr += max_size;
    double *loss = ptr; // zero initialized
    ptr += loss_size;
    double *images = ptr;
    ptr += number_of_images*image_size;
    double *biases = ptr; // bias GRADIENTS zero initialized
    ptr += bias_mat_size;
    double *activations = ptr;
    ptr += act_mat_size;
    double *weights = ptr; // weight GRADIENTS zero initialized
    ptr += weight_mat_size;
    // NOTE: core sync flag used to indictae whether computation is done or not
    // WARN: fails in the RTL
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

    // // INFO FP32 cluster memory setup
    // // images remain in float precision
    // void *ptr = (float *)snrt_cluster_memory().start;
    // float *max= ptr; // zero initialized
    // ptr += max_size;
    // float *loss = ptr; // zero initialized
    // ptr += loss_size;
    // float *images = ptr;
    // ptr += number_of_images*image_size;
    // float *biases = ptr; // bias GRADIENTS zero initialized
    // ptr += bias_mat_size;
    // float *activations = ptr;
    // ptr += act_mat_size;
    // float *weights = ptr; // weight GRADIENTS zero initialized
    // ptr += weight_mat_size;
    // // NOTE: core sync flag used to indictae whether computation is done or not
    // uint32_t *core_sync = ptr; // zero initialized
    // ptr += core_sync_flag_size;
    // uint32_t *targets = ptr;
    // ptr += number_of_images*target_size;
    // // NOTE: following lines for debugging purposes only
    // // void *ptr_end = (double *)snrt_cluster_memory().end;
    // // if(compute_id == 0){   
    // //     printf("Start address of cluster %u memory: 0x%p\n", cluster_id, ptr_start);
    // //     printf("End address of cluster %u memory: 0x%p\n", cluster_id, ptr_end);
    // //     printf("Available memory on cluster %u: %u KB\n", cluster_id, (ptr_end - ptr_start) / 1000);
    // //     printf("Total cluster memory occupation on cluster %u: %u KB\n", cluster_id, (ptr - ptr_start) / 1000);
    // // }

    // // INFO FP16 cluster memory setup
    // void *ptr = (__fp16*)snrt_cluster_memory().start;
    // __fp16 *max= ptr; // zero initialized
    // ptr += max_size;
    // __fp16 *loss = ptr; // zero initialized
    // ptr += loss_size;
    // __fp16 *biases = ptr; // bias GRADIENTS zero initialized
    // ptr += bias_mat_size;
    // __fp16 *activations = ptr;
    // ptr += act_mat_size;
    // __fp16 *weights = ptr; // weight GRADIENTS zero initialized
    // ptr += weight_mat_size;
    // __fp16 *images = ptr;
    // ptr += number_of_images*image_size;
    // // NOTE: core sync flag used to indictae whether computation is done or not
    // uint32_t *core_sync = ptr; // zero initialized
    // ptr += core_sync_flag_size;
    // uint32_t *targets = ptr;
    // ptr += number_of_images*target_size;
    // // NOTE: following lines for debugging purposes only
    // // void *ptr_end = (double *)snrt_cluster_memory().end;
    // // if(compute_id == 0){   
    // //     printf("Start address of cluster %u memory: 0x%p\n", cluster_id, ptr_start);
    // //     printf("End address of cluster %u memory: 0x%p\n", cluster_id, ptr_end);
    // //     printf("Available memory on cluster %u: %u KB\n", cluster_id, (ptr_end - ptr_start) / 1000);
    // //     printf("Total cluster memory occupation on cluster %u: %u KB\n", cluster_id, (ptr - ptr_start) / 1000);
    // // }

    // cluster offset in an Occamy quadrant
    uint32_t cluster_offset = 0x00040000;
    // define pointers for cluster2cluster communication
    // WARN: make sure dtype is correct
    // double *act_ptr;
    // double *img_ptr;

    // each cluster should set their sync flag to zero initially
    core_sync[compute_id] = 0;

    uint32_t setup_SSR;
    if(BASELINE){
        // baseline does *not* use SSRs
        setup_SSR = 0;
    } else {
        // define whether SSRs should be used or not
        setup_SSR = 1;
    }

    // We load the GM weights and biases into cluster 0 memory 
    // together with the image data.
    // TODO: add the epochs
    if (snrt_is_dm_core() && cluster_id == 0) {
        snrt_dma_start_tracking();
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
                                    n->dtype * number_of_images * IN_CH);       // size

                // wait until each DMA transfer done
                snrt_dma_wait_all();
        snrt_dma_stop_tracking();

    }


    if (snrt_is_dm_core() && cluster_id == 1) {
        snrt_dma_start_tracking();
        // On cluster 1 we load the labels which are needed for BP
                snrt_dma_txid_t txid_targets = 
                    snrt_dma_start_1d(targets,                                   // destination
                                    n->targets,                                  // source
                                    sizeof(uint32_t) * number_of_images);        // size
                
                snrt_dma_wait_all();
        snrt_dma_stop_tracking();
    }

    snrt_cluster_hw_barrier();
    // Global memory access
    // uint32_t *global_mem = (void *)snrt_global_memory().start;
    
    // DRAM dataset memory start address
    // NOTE: this is only needed when preloading the DRAM in banshee
    // uint32_t *dataset_dram = (void *)0x8004000;

    // We now loop through the images
    for(uint32_t image = 0; image < number_of_images; image++){

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
            if(RUN_FEEDFORWARD){
                // perform the forward pass
                if(!compute_id){
                    printf("[MNIST] FF start\n");
                }

                // Start of feedforward
                switch(n->dtype){
                    case FP64:
                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            feedforward_fp64(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
                                            ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id]);
                            softmax_activation_fp64(n->IN_CH1, n->IN_CH2, div, 
                                        &weights[W_offset], ldW, &activations[b_offset], ldB,
                                        &images[curr_img], ldI, compute_id, compute_num, max, &core_sync);
                            benchmark_get_cycle();
                        } else {
                            // INFO: FP64 with SSRs
                            benchmark_get_cycle();
                            feedforward_fp64_ssr(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
                                            ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id],
                                            setup_SSR);
                            softmax_activation_fp64_ssr(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], ldW, &activations[b_offset], ldB,
                                            &images[curr_img], ldI, compute_id, compute_num, max, &core_sync, setup_SSR);
                            benchmark_get_cycle();
                        }
                        break;

                    case FP32:
                        if(BASELINE){
                            // INFO: baseline   
                            benchmark_get_cycle();
                            feedforward_fp32(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
                                            ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id]);
                            softmax_activation_fp32(n->IN_CH1, n->IN_CH2, div,
                                            &weights[W_offset], ldW, &activations[b_offset], ldB, 
                                            &images[curr_img], ldI, compute_id, compute_num, max, &core_sync);
                            benchmark_get_cycle();
                        } else {
                            // INFO: FP32 with SSRs and SIMD
                            benchmark_get_cycle();
                            feedforward_fp32_ssr_simd(n->IN_CH1, n->IN_CH2, div, 
                                        &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
                                        ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id],
                                        setup_SSR); 
                            softmax_activation_fp32(n->IN_CH1, n->IN_CH2, div,
                                            &weights[W_offset], ldW, &activations[b_offset], ldB, 
                                            &images[curr_img], ldI, compute_id, compute_num, max, &core_sync);
                            benchmark_get_cycle();
                        }
                        break;

                    case FP16:
                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            feedforward_fp16(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
                                            ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id]);
                            softmax_activation_fp16(n->IN_CH1, n->IN_CH2, div,
                                            &weights[W_offset], ldW, &activations[b_offset], ldB, 
                                            &images[curr_img], ldI, compute_id, compute_num, max, &core_sync);
                            benchmark_get_cycle();
                        } else {
                            // INFO: FP16 with SSRs and SIMD
                            benchmark_get_cycle();
                            feedforward_fp16_ssr_simd(n->IN_CH1, n->IN_CH2, div, 
                                        &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
                                        ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id],
                                        setup_SSR);
                            softmax_activation_fp16(n->IN_CH1, n->IN_CH2, div, 
                                        &weights[W_offset], ldW, &activations[b_offset], ldB,   
                                        &images[curr_img], ldI, compute_id, compute_num, max, &core_sync);
                            benchmark_get_cycle();
                        }
                        break;
                    
                    case FP8:
                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            feedforward_fp8(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], ldW, &biases[b_offset], &activations[b_offset],
                                            ldB, &images[curr_img], ldI, compute_id, &core_sync[compute_id]);
                            softmax_activation_fp8(n->IN_CH1, n->IN_CH2, div,
                                            &weights[W_offset], ldW, &activations[b_offset], ldB, 
                                            &images[curr_img], ldI, compute_id, compute_num, max, &core_sync);
                            benchmark_get_cycle();
                        } else {
                            // INFO: FP8 with SSRs and SIMD
                            if(!compute_id){
                                printf("ERROR: Not implemented yet.\n");
                            }
                        }
                        break;

                    default:
                        if(!compute_id){
                                printf("ERROR: unsupported data type\n");
                        }
                        break;

                }

                if(!compute_id){
                    printf("[MNIST] FF end\n");
                }
            }

        } else if (!snrt_is_compute_core() && cluster_id == 0){
            switch (n->dtype) {
                case(FP64):
                    if(BASELINE){
                        // INFO: baseline
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                        snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                    } else {
                        // INFO: FP64 with SSRs
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                        snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                    }
                    break;
                case(FP32):
                    if(BASELINE){
                        // INFO: baseline
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                    } else {
                        // INFO: FP32 with SSRs and SIMD
                        snrt_cluster_hw_barrier();
                        // snrt_cluster_hw_barrier();
                        // snrt_cluster_hw_barrier();
                        // snrt_cluster_hw_barrier();
                    }
                    break;
                case (FP16):
                    if(BASELINE){
                        // INFO: baseline
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                    } else {
                        // INFO: FP16 with SSRs and SIMD
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                    }
                    break;
                case(FP8):
                    if(BASELINE){
                        // INFO: baseline
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                    } else {
                        // INFO: FP8 with SSRs and SIMD
                        printf("ERROR: Not implemented yet.\n");
                    }
                    break;
                default:
                    break;
            }
        }

        // wait until clusters are synchronized to not
        // start gradient update until all activations are 
        // computed
        // snrt_global_barrier();
        // INFO: replacing global barrier with custom barrier for RTL sims
        snrt_generic_cluster_barrier(cluster_num*cluster_core_num);

        // Code below for testing cluster2cluster DMA transfer
        // if(snrt_is_dm_core() && cluster_id){
        //     double *act_ptr = ((uint32_t)activations) - cluster_offset;
        //     snrt_dma_txid_t txid_activations = 
        //             snrt_dma_start_1d(activations,                                   // destination
        //                               act_ptr,                                      // source
        //                               n->dtype * n->OUT_CH);                         // size
        //     printf("CLID[%u]: cluster 1 activation ptr: 0x%p\n", cluster_id, activations);
        //     printf("CLID[%u]: cluster 1 image ptr: 0x%p\n", cluster_id, images);
        //     printf("CLID[%u]: cluster 0 activation ptr: 0x%p\n", cluster_id, ((uint32_t)activations) - cluster_offset);
        //     printf("CLID[%u]: cluster 0 image ptr: 0x%p\n", cluster_id, ((uint32_t)images) - cluster_offset);
        // }

        // if(snrt_is_dm_core() && !cluster_id){
        //     printf("CLID[%u]: cluster 0 activation ptr: 0x%p\n", cluster_id, activations);
        //     printf("CLID[%u]: cluster 0 image ptr: 0x%p\n", cluster_id, images);
        // }


        if(setup_SSR && RUN_GRADIENT_UPDATE){
            if(snrt_is_dm_core() && cluster_id==1) {
                snrt_dma_start_tracking();
                // WARN: make sure that pointer types are according to network precision
                double *act_ptr = ((uint32_t)activations) - cluster_offset;
                double *img_ptr = ((uint32_t)images) - cluster_offset;

                // for SSRs we need to DMA transfer the cluster 0 data to cluster 1
                snrt_dma_txid_t txid_activations = 
                    snrt_dma_start_1d(activations,                                 // destination
                                    act_ptr,                                       // source
                                    n->dtype * n->OUT_CH);                         // size

                snrt_dma_txid_t txid_IMG = 
                    snrt_dma_start_1d(images,                                    // destination
                                    img_ptr,                                     // source
                                    n->dtype * number_of_images * IN_CH);        // size
                
                snrt_dma_wait_all();

                snrt_dma_stop_tracking();

            } 
        }

        snrt_cluster_hw_barrier();

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

            // INFO: without SSRs we can use direct cluster2cluster communication
            // however, when SSRs are set we need to DMA transfer the data

            double *act_ptr = ((uint32_t)activations) - cluster_offset;
            double *img_ptr = ((uint32_t)images) - cluster_offset;

            //double *core_sync_ptr = ((uint32_t)core_sync) - cluster_offset;
            //printf("activations[%u] = %f\n", b_offset, act_ptr[b_offset]);

            if(RUN_GRADIENT_UPDATE){
                if(!compute_id){
                    printf("[MNIST] Gradient Update start\n");
                }
                switch (n->dtype) {
                    case(FP64):
                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            gradient_update_fp64(n->IN_CH1, n->IN_CH2, div, 
                                                &weights[W_offset], ldW, 
                                                &biases[b_offset], &act_ptr[b_offset], 
                                                ldB, &img_ptr[curr_img], &targets[curr_img], ldI, compute_id, 
                                                loss, compute_num);
                            benchmark_get_cycle();
                        } else {
                            // INFO: FP64 with SSRs
                            benchmark_get_cycle();
                            gradient_update_fp64_ssr(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], ldW, 
                                            &biases[b_offset], &activations[b_offset], 
                                            ldB, &images[curr_img], &targets[curr_img], ldI, compute_id, 
                                            loss, compute_num, setup_SSR);
                            benchmark_get_cycle();
                        } 
                        break;
                    case(FP32):
                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            gradient_update_fp32(n->IN_CH1, n->IN_CH2, div, 
                                                &weights[W_offset], ldW, 
                                                &biases[b_offset], &act_ptr[b_offset], 
                                                ldB, &img_ptr[curr_img], &targets[curr_img], ldI, compute_id, 
                                                loss, compute_num);
                            benchmark_get_cycle();
                        } else {
                            // INFO: FP32 with SSRs and SIMD
                            benchmark_get_cycle();
                            gradient_update_fp32_ssr_simd(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], ldW, 
                                            &biases[b_offset], &activations[b_offset], 
                                            ldB, &images[curr_img], &targets[curr_img], ldI, compute_id, 
                                            loss, compute_num, setup_SSR);
                            benchmark_get_cycle();
                        }
                        break;
                    case (FP16):
                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            gradient_update_fp16(n->IN_CH1, n->IN_CH2, div, 
                                                &weights[W_offset], ldW, 
                                                &biases[b_offset], &act_ptr[b_offset], 
                                                ldB, &img_ptr[curr_img], &targets[curr_img], ldI, compute_id, 
                                                loss, compute_num);
                            benchmark_get_cycle();
                        } else {
                            // INFO: FP16 with SSRs and SIMD
                            benchmark_get_cycle();
                            gradient_update_fp16_ssr_simd(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], ldW, 
                                            &biases[b_offset], &activations[b_offset], 
                                            ldB, &images[curr_img], &targets[curr_img], ldI, compute_id, 
                                            loss, compute_num, setup_SSR);
                            benchmark_get_cycle();
                        }
                        break;
                    case(FP8):
                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            gradient_update_fp8(n->IN_CH1, n->IN_CH2, div, 
                                                &weights[W_offset], ldW, 
                                                &biases[b_offset], &act_ptr[b_offset], 
                                                ldB, &img_ptr[curr_img], &targets[curr_img], ldI, compute_id, 
                                                loss, compute_num);
                            benchmark_get_cycle();
                        } else {
                            // INFO: FP8 with SSRs and SIMD
                            if(!compute_id){
                                printf("ERROR: Not implemented yet.\n");
                            }
                        }
                        break;
                    default:
                        if(!compute_id){
                            printf("ERROR: Unknown data type.\n");
                        }
                        break;
                }

                if(!compute_id){
                    printf("[MNIST] Gradient Update done\n");
                    printf("total loss = %f\n", loss[0]/(image+1));
                }
            }

        } else if(!snrt_is_compute_core() && cluster_id == 1){
            switch (n->dtype)
            {
            case FP64:
                if(RUN_GRADIENT_UPDATE){
                    snrt_cluster_hw_barrier();
                } else {
                    printf("[MNIST] INFO: Gradient Update not run\n");
                }
                break;
            case FP32:
                if(BASELINE){
                    if(RUN_GRADIENT_UPDATE){
                        snrt_cluster_hw_barrier();
                    } else {
                        printf("[MNIST] INFO: Gradient Update not run\n");
                    }
                } else {
                    if(RUN_GRADIENT_UPDATE){
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                    } else {
                        printf("[MNIST] INFO: Gradient Update not run\n");
                    }
                }
                break;
            case FP16:
                if(BASELINE){
                    if(RUN_GRADIENT_UPDATE){
                        snrt_cluster_hw_barrier();
                    } else {
                        printf("[MNIST] INFO: Gradient Update not run\n");
                    }
                } else {
                    if(RUN_GRADIENT_UPDATE){
                        snrt_cluster_hw_barrier();
                    } else {
                        printf("[MNIST] INFO: Gradient Update not run\n");
                    }
                }
                break;
            case FP8:
                if(BASELINE){
                    if(RUN_GRADIENT_UPDATE){
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                    } else {
                        printf("INFO: Gradient Update not run\n");
                    }
                } else {
                    printf("ERROR: Not implemented yet.\n");
                }
                break;
            default:
                printf("ERROR: Unknown data type.\n");
                break;
            }
        }
    }

    // snrt_global_barrier();
    // INFO: replacing global barrier with custom barrier for RTL sims
    snrt_generic_cluster_barrier(cluster_num*cluster_core_num);

    // TODO: implement correct DMA transfer
    if(setup_SSR && RUN_TRAINING_STEP){
        // for SSRs we need to DMA transfer the cluster 1 data to cluster 0
        if(snrt_is_dm_core() && cluster_id==0) {
            snrt_dma_start_tracking();
            // WARN: make sure that pointer types are according to network precision
            double *weight_grad_ptr = ((uint32_t)weights) + cluster_offset;
            double *bias_grad_ptr = ((uint32_t)biases) + cluster_offset;
            snrt_dma_txid_t txid_WG = 
                snrt_dma_start_2d(weights,                // destination
                                weight_grad_ptr,          // source
                                n->dtype * IN_CH,         // size
                                n->dtype * IN_CH ,        // destination stride
                                n->dtype * IN_CH ,        // source stride
                                n->OUT_CH);               // repetitions

            snrt_dma_txid_t txid_BG = 
                snrt_dma_start_1d(activations,                 // destination
                                bias_grad_ptr,            // source
                                n->dtype * n->OUT_CH);    // size

            
            snrt_dma_wait_all();

            snrt_dma_stop_tracking();
        } 
    }

    snrt_cluster_hw_barrier();

    // after looping through one batch of the dataset
    // we update the biases and weights on Cluster 0
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


        // determine the row stride of each matrix    
        volatile uint32_t ldW = compute_num * IN_CH;
        volatile uint32_t ldB = compute_num;
        volatile uint32_t ldI = IN_CH;

        double *weight_grad_ptr = ((uint32_t)weights) + cluster_offset;
        double *bias_grad_ptr = ((uint32_t)biases) + cluster_offset;

        //TODO: load the LR from the network struct or via DRAM perloading
        //*learning_rate = 0.5;

        if(RUN_TRAINING_STEP){
            if(!compute_id){
                    printf("[MNIST] Training step start\n");
            }

            switch(n->dtype)
            {
                case FP64:
                    if(BASELINE){
                        // INFO: baseline
                        benchmark_get_cycle();
                        training_step_fp64(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], &weight_grad_ptr[W_offset], ldW, 
                                            &biases[b_offset], &bias_grad_ptr[b_offset], ldB, 
                                            compute_id, compute_num, number_of_images);
                        benchmark_get_cycle();
                    } else {
                        // INFO: FP64 with SSRs
                        benchmark_get_cycle();
                        training_step_fp64_ssr(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], &weights[W_offset], ldW, 
                                            &biases[b_offset], &activations[b_offset], ldB, 
                                            compute_id, compute_num, number_of_images, setup_SSR);
                        benchmark_get_cycle();
                    }
                    break;
                case FP32:
                    if(BASELINE){
                        // INFO: baseline
                        benchmark_get_cycle();
                        training_step_fp32(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], &weight_grad_ptr[W_offset], ldW, 
                                            &biases[b_offset], &bias_grad_ptr[b_offset], ldB, 
                                            compute_id, compute_num, number_of_images);
                        benchmark_get_cycle();
                    } else {
                        // INFO: FP32 with SSRs and SIMD
                        benchmark_get_cycle();
                        training_step_fp32_ssr_simd(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], &weights[W_offset], ldW, 
                                            &biases[b_offset], &activations[b_offset], ldB, 
                                            compute_id, compute_num, number_of_images, setup_SSR);
                        benchmark_get_cycle();
                    }
                    break;
                case FP16:
                    if(BASELINE){
                        // INFO: baseline
                        benchmark_get_cycle();
                        training_step_fp16(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], &weight_grad_ptr[W_offset], ldW, 
                                            &biases[b_offset], &bias_grad_ptr[b_offset], ldB, 
                                            compute_id, compute_num, number_of_images);
                        benchmark_get_cycle();
                    } else {
                        // INFO: FP16 with SSRs and SIMD
                        benchmark_get_cycle();
                        training_step_fp16_ssr_simd(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], &weights[W_offset], ldW, 
                                            &biases[b_offset], &activations[b_offset], ldB, 
                                            compute_id, compute_num, number_of_images, setup_SSR);
                        benchmark_get_cycle();
                    }
                    break;
                case FP8:
                    if(BASELINE){
                        // INFO: baseline
                        benchmark_get_cycle();
                        training_step_fp8(n->IN_CH1, n->IN_CH2, div, 
                                            &weights[W_offset], &weight_grad_ptr[W_offset], ldW, 
                                            &biases[b_offset], &bias_grad_ptr[b_offset], ldB, 
                                            compute_id, compute_num, number_of_images);
                        benchmark_get_cycle();
                    } else {
                        // INFO: FP8 with SSRs and SIMD
                        if(!compute_id){
                            printf("ERROR: Not implemented yet.\n");
                        }
                    }
                    break;
                default:
                    if(!compute_id){
                        printf("ERROR: Unknown data type.\n");
                    }
                    break;
            }

            if(!compute_id){
                    printf("[MNIST] Training step done\n");
            }
        }


    } else if(!snrt_is_compute_core() && cluster_id == 0){
        switch(n->dtype){
            case FP64:
                if(BASELINE){
                    if(RUN_TRAINING_STEP){
                        // no HW barriers required for Baseline
                    } else {
                        printf("[MNIST] INFO: Training Step not run\n");
                    }
                } else {
                    if(RUN_TRAINING_STEP){
                    } else {
                        printf("[MNIST] INFO: Training Step not run\n");
                    }
                }
                break;
            case FP32:
                if(BASELINE){
                    if(RUN_TRAINING_STEP){

                    } else {
                        printf("[MNIST] INFO: Training Step not run\n");
                    }
                } else {
                    if(RUN_TRAINING_STEP){
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                    } else {
                        printf("[MNIST] INFO: Training Step not run\n");
                    }
                }
                break;
            case FP16:
                if(BASELINE){
                    if(RUN_TRAINING_STEP){
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                    } else {
                        printf("[MNIST] INFO: Training Step not run\n");
                    }
                } else {

                }
                break;
            case FP8:
                if(BASELINE){
                    if(RUN_TRAINING_STEP){
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                        snrt_cluster_hw_barrier();
                    } else {
                        printf("[MNIST] INFO: Training Step not run\n");
                    }
                } else {
                    printf("ERROR: Not implemented yet.\n");
                }
                break;
        }
    }
}
