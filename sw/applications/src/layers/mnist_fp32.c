// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_fp32.h"

#include "network.h"
#include "mnist_fp32_network.h"
#include "printf.h"
#include "snrt.h"
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

void mnist_fp32(const network_fp32_t *n){

    uint32_t cluster_num = snrt_cluster_num(); 
    uint32_t cluster_core_num = snrt_cluster_core_num();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_num = snrt_cluster_compute_core_num(); // Number of compute cores per cluster 
    uint32_t global_compute_num = snrt_global_core_num(); // Total cores incl. DM core per cluster 
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
    uint32_t image_size = IN_CH * n->dtype;
    // size for storing the maximum on each core of a cluster (only used on cluster 0)
    uint32_t max_size = compute_num * n->dtype; // or n->dtype?
    // size of the target for image classification (0...9)
    uint32_t target_size = sizeof(uint32_t);
    // result of the cross entropy loss calculation
    uint32_t loss_size = n->dtype;
    // synchronization flags for the compute cores on among clusters
    uint32_t core_sync_flag_size = compute_num*sizeof(uint32_t);
    // learning rate of the network
    uint32_t lr_size = sizeof(float);

    // cluster 0 variabels:
    float *weights_cl0;
    float *weight_grads_cl0;
    float *biases_cl0;
    float *images;
    float *activations_cl0;
    float *max;

    // cluster 1 variabels:
    float *weights_cl1; // dummy variable to match offsets with cluster 0
    float *weight_grads_cl1;
    float *bias_grads_cl1;
    float *activations_cl1;
    float *loss;
    float *targets;

    // INFO FP32 cluster memory setup
    // @brief Cluster Memory Structure for each cluster to ensure
    // we can access the data of both by using the constant
    // cluster base offset
    void *ptr = (double *)snrt_cluster_memory().start;
    // void *ptr_start = ptr;
    if(cluster_id == 0){
        weights_cl0 = ptr;
        ptr += weight_mat_size;
        weight_grads_cl0 = ptr;
        ptr += weight_mat_size;
        biases_cl0 = ptr;
        ptr += bias_mat_size;
        images = ptr;
        ptr += image_size * number_of_images;
        // INFO: the activations are also used for the bias gradients
        activations_cl0 = ptr;
        ptr += act_mat_size;
        max = ptr;
        ptr += max_size;
    } else if (cluster_id == 1){
        weights_cl1 = ptr;
        ptr += weight_mat_size;
        weight_grads_cl1 = ptr;
        ptr += weight_mat_size;
        bias_grads_cl1 = ptr;
        ptr += bias_mat_size;
        images = ptr;
        ptr += image_size * number_of_images;
        activations_cl1 = ptr;
        ptr += act_mat_size;
        loss = ptr;
        ptr += loss_size;
        targets = ptr;
        ptr += target_size * number_of_images;
    }
    // void *ptr = (float *)snrt_cluster_memory().start;
    // // void *ptr_start = ptr;
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
    // // WARN: fails in the RTL
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

    // cluster offset in an Occamy quadrant
    uint32_t cluster_offset = 0x00040000;

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
                    snrt_dma_start_1d(biases_cl0,                 // destination
                                    n->b,                     // source
                                    n->dtype * n->OUT_CH);    // size

                // load weight data into Cluster 0 memory
                // TODO: make this 1D DMA transfer
                snrt_dma_txid_t txid_W = 
                    snrt_dma_start_2d(weights_cl0,                // destination
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

            // determine the row stride of each matrix    
            volatile uint32_t ldW = compute_num * IN_CH;
            volatile uint32_t ldB = compute_num;
            volatile uint32_t ldI = IN_CH;

            if(RUN_FEEDFORWARD){
                
                // if(!compute_id){
                //     printf("[MNIST] FF start\n");
                // }

                // TODO: remove core_sync
                if(BASELINE){
                    // INFO: baseline
                    benchmark_get_cycle();
                    feedforward_fp32n(n->IN_CH1, n->IN_CH2, div, 
                                    &weights_cl0[W_offset], ldW, &biases_cl0[b_offset], &activations_cl0[b_offset],
                                    ldB, &images[curr_img], ldI, compute_id);
                    softmax_activation_fp32n(n->IN_CH1, n->IN_CH2, div, 
                                &weights_cl0[W_offset], ldW, &activations_cl0[b_offset], ldB,
                                &images[curr_img], ldI, compute_id, compute_num, max);
                    benchmark_get_cycle();
                } else {
                    // INFO: FP64 with SSRs
                    benchmark_get_cycle();
                    feedforward_fp32_ssr_simd_frep(n->IN_CH1, n->IN_CH2, div, 
                                        &weights_cl0[W_offset], ldW, &biases_cl0[b_offset], &activations_cl0[b_offset],
                                        ldB, &images[curr_img], ldI, compute_id, setup_SSR);
                    softmax_activation_fp32n(n->IN_CH1, n->IN_CH2, div, 
                                &weights_cl0[W_offset], ldW, &activations_cl0[b_offset], ldB,
                                &images[curr_img], ldI, compute_id, compute_num, max);
                    benchmark_get_cycle();
                }

                // if(!compute_id){
                //     printf("[MNIST] FF end\n");
                // }

            }
        } else if (!snrt_is_compute_core() && cluster_id == 0){

            if(RUN_FEEDFORWARD){
                if(BASELINE){
                    // INFO: baseline
                    snrt_cluster_hw_barrier();
                    snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                    snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                } else {
                    // INFO: FP32 with SSRs
                    snrt_cluster_hw_barrier();
                    snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                    snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                }
            } else {
                // if(!cluster_id){
                //     printf("[MNIST] FF not run. \n");
                // }
            }
        }

        // wait until clusters are synchronized to not
        // start gradient update until all activations are 
        // computed
        // snrt_global_barrier();
        // INFO: replacing global barrier with custom barrier for RTL sims
        snrt_generic_cluster_barrier(cluster_num*cluster_core_num);

        if(setup_SSR && RUN_GRADIENT_UPDATE){
            if(snrt_is_dm_core() && cluster_id==1) {
                snrt_dma_start_tracking();
                // WARN: make sure that pointer types are according to network precision
                float *act_ptr = ((uint32_t)activations_cl1) - cluster_offset;
                float *img_ptr = ((uint32_t)images) - cluster_offset;

                // for SSRs we need to DMA transfer the cluster 0 data to cluster 1
                snrt_dma_txid_t txid_activations = 
                    snrt_dma_start_1d(activations_cl1,                                 // destination
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

            float *act_ptr = ((uint32_t)activations_cl1) - cluster_offset;
            float *img_ptr = ((uint32_t)images) - cluster_offset;

            if(RUN_GRADIENT_UPDATE){
                // if(!compute_id){
                //     printf("[MNIST] GU start\n");
                // }
                if(BASELINE){
                    // INFO: baseline
                    benchmark_get_cycle();
                    gradient_update_fp32n(n->IN_CH1, n->IN_CH2, div, 
                                        &weight_grads_cl1[W_offset], ldW, 
                                        &bias_grads_cl1[b_offset], &activations_cl1[b_offset], 
                                        ldB, &img_ptr[curr_img], &targets[curr_img], ldI, compute_id, 
                                        loss, compute_num);
                    benchmark_get_cycle();
                } else {
                    // INFO: FP64 with SSRs
                    benchmark_get_cycle();
                    gradient_update_fp32_ssr_simdn(n->IN_CH1, n->IN_CH2, div, 
                                            &weight_grads_cl1[W_offset], ldW, 
                                            &bias_grads_cl1[b_offset], &activations_cl1[b_offset], 
                                            ldB, &images[curr_img], &targets[curr_img], ldI, compute_id, 
                                            loss, compute_num, setup_SSR);
                    benchmark_get_cycle();
                }
                // if(!compute_id){
                //     printf("[MNIST] GU end\n");
                // }
            } else {
                // if(!compute_id){
                //     printf("[MNIST] GU not run. \n");
                // }
            } // end of gradient update
        } else if (!snrt_is_compute_core() && cluster_id == 1){
            if(RUN_GRADIENT_UPDATE){
                if(BASELINE){
                    // INFO: baseline
                    snrt_cluster_hw_barrier();
                } else {
                    // INFO: FP32 with SSRs
                    snrt_cluster_hw_barrier();
                }
            } else {
                // if(!cluster_id){
                //     printf("[MNIST] GU not run. \n");
                // }
            }
        }
    }

    // snrt_global_barrier();
    // INFO: replacing global barrier with custom barrier for RTL sims
    snrt_generic_cluster_barrier(cluster_num*cluster_core_num);

    if(setup_SSR && RUN_TRAINING_STEP){
        // for SSRs we need to DMA transfer the cluster 1 data to cluster 0
        if(snrt_is_dm_core() && cluster_id==0) {
            // Discuss with GIM how to do DMA benchmarking
            snrt_dma_start_tracking();
            // WARN: make sure that pointer types are according to network precision
            float *weight_grad_ptr = ((uint32_t)weight_grads_cl0) + cluster_offset;
            float *bias_grad_ptr = ((uint32_t)biases_cl0) + cluster_offset;
            snrt_dma_txid_t txid_WG = 
                snrt_dma_start_2d(weight_grads_cl0,       // destination
                                weight_grad_ptr,          // source
                                n->dtype * IN_CH,         // size
                                n->dtype * IN_CH ,        // destination stride
                                n->dtype * IN_CH ,        // source stride
                                n->OUT_CH);               // repetitions

            snrt_dma_txid_t txid_BG = 
                snrt_dma_start_1d(activations_cl0,        // destination
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

        float *weight_grad_ptr = ((uint32_t)weight_grads_cl0) + cluster_offset;
        float *bias_grad_ptr = ((uint32_t)biases_cl0) + cluster_offset;

        //TODO: load the LR from the network struct or via DRAM perloading
        //*learning_rate = 0.5;

        if(RUN_TRAINING_STEP){

            // if(!compute_id){
            //         printf("[MNIST] Training step start\n");
            // }

            if(BASELINE){
                // INFO: baseline
                benchmark_get_cycle();
                training_step_fp32n(n->IN_CH1, n->IN_CH2, div, 
                                            &weights_cl0[W_offset], &weight_grads_cl0[W_offset], ldW, 
                                            &biases_cl0[b_offset], &activations_cl0[b_offset], ldB, 
                                            compute_id, compute_num, number_of_images);
                benchmark_get_cycle();
            } else {
                // INFO: FP64 with SSRs
                benchmark_get_cycle();
                training_step_fp32_ssr_simdn(n->IN_CH1, n->IN_CH2, div, 
                                            &weights_cl0[W_offset], &weight_grads_cl0[W_offset], ldW, 
                                            &biases_cl0[b_offset], &activations_cl0[b_offset], ldB, 
                                            compute_id, compute_num, number_of_images, setup_SSR);
                benchmark_get_cycle();
            }

            // if(!compute_id){
            //         printf("[MNIST] Training step done\n");
            // }
        }

    } else if(!snrt_is_compute_core() && cluster_id == 0){
        if(BASELINE){
            if(RUN_TRAINING_STEP){
                // no HW barriers required for Baseline
            } else {
                // printf("[MNIST] INFO: Training Step not run\n");
            }
        } else {
            if(RUN_TRAINING_STEP){
            } else {
                // printf("[MNIST] INFO: Training Step not run\n");
            }
        }
    }

}