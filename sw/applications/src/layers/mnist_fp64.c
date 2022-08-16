// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_fp64.h"

#include "network.h"
#include "mnist_fp64_network.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

// Padding in between matrices for preventing
// banking conflicts in the beginning
#define MAT_PADDING 0
#define MAT_ROW_PADDING 0

// define whether to run baseline network or not
#define BASELINE 1

// define which parts of the network to run
#define RUN_FEEDFORWARD 1
#define RUN_GRADIENT_UPDATE 1
#define RUN_TRAINING_STEP 1 // WARN: for SSRs we cannot run the training step in the RTL
#define GET_ACCURACY 1
#define GET_LOSS 0
#define RUN_RTL 0
#define RUN_FENGA 1

// define NN properties
#define NUM_EPOCHS 1 // 10: default epochs
#define BATCH_SIZE 3// 256: default batch size
#define NUM_BATCHES 1// (60000 / BATCH_SIZE): default number of batches
#define STEPS(batches) (int)NUM_EPOCHS*(batches)

void mnist_fp64(const network_fp64_t *n){

    // printf("num_batches: %d\n", NUM_BATCHES);


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
    uint32_t number_of_images = 60000;

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
    
    // cluster 0 variabels:
    double *weights_cl0;
    double *weight_grads_cl0;
    double *biases_cl0;
    double *images;
    double *activations_cl0;
    double *max;

    // cluster 1 variabels:
    double *weights_cl1; // dummy variable to match offsets with cluster 0
    double *weight_grads_cl1;
    double *bias_grads_cl1;
    double *activations_cl1;
    double *loss;
    uint32_t *targets;

    // test variable for checking binary dram preloading
    double *test_dram;

    void *ptr = (double *)snrt_cluster_memory().start;
    // void *ptr_start = ptr;
    if(cluster_id == 0){
        weights_cl0 = ptr;
        ptr += weight_mat_size;
        // weight_grads_cl0 = ptr;
        // ptr += weight_mat_size;
        biases_cl0 = ptr;
        ptr += bias_mat_size;
        images = ptr;
        ptr += image_size; // * capacity; // TODO: for every FP type determine how many images we can load
        // INFO: the activations are also used for the bias gradients
        activations_cl0 = ptr;
        ptr += act_mat_size;
        max = ptr;
        ptr += max_size;
        test_dram = ptr;
        ptr += image_size;
        targets = ptr;
        ptr += target_size;
    } else if (cluster_id == 1){
        // weights_cl1 = ptr;
        // ptr += weight_mat_size;
        weight_grads_cl1 = ptr;
        ptr += weight_mat_size;
        bias_grads_cl1 = ptr;
        ptr += bias_mat_size;
        images = ptr;
        ptr += image_size; // * capacity; // TODO: for every FP type determine how many images we can load
        activations_cl1 = ptr;
        ptr += act_mat_size;
        targets = ptr;
        ptr += target_size; // * capacity; // TODO: for every FP type determine how many images we can load
        loss = ptr;
        ptr += loss_size;
    } 
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

    // DRAM dataset memory start address
    // NOTE: this is only needed when preloading the DRAM in banshee
    double *images_dram = (void *)0x80040000;
    uint32_t *targets_dram = (void *)0x80108000;

    // if(snrt_is_dm_core() && cluster_id == 0){
    //     for(int j = 0; j < 10; j++){
    //         // print 10 images for debugging purposes
    //         for(int i = 0; i < IN_CH; i++){
    //             printf("DRAM image[%u] = %f\n", i, images_dram[j*IN_CH + i]);
    //         }
    //     }
    // } // INFO: this works fine

    // if(!compute_id && !cluster_id){

    //     for(uint32_t i = 0; i < 60000; i++){
    //         printf("target[%u] = %u\n", i, targets_dram[i]);
    //     }
    // }

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
                    snrt_dma_start_2d(weights_cl0,                 // destination
                                    n->W,                          // source
                                    n->dtype * IN_CH,         // size
                                    n->dtype * IN_CH ,        // destination stride
                                    n->dtype * IN_CH ,        // source stride
                                    n->OUT_CH);                    // repetitions

                // wait until each DMA transfer done
                snrt_dma_wait_all();

                // for(uint32_t i = 0; i < n->OUT_CH; i++){
                //     printf("%f ", weights_cl0[i]);
                // }
                // printf("\n");

                // snrt_dma_txid_t txid_IMG = 
                //     snrt_dma_start_1d(test_dram,                                   // destination
                //                     &images_dram[0],                                  // source
                //                     n->dtype * IN_CH);                         // size

        snrt_dma_stop_tracking();

    }

    // printf("Before barrier Cluster %u: compute_id %u\n", cluster_id, compute_id);  
    snrt_global_barrier();
    // printf("After barrier Cluster %u: compute_id %u\n", cluster_id, compute_id);
    

    for(uint32_t epoch = 0; epoch < NUM_EPOCHS; epoch++){

    //     // snrt_global_barrier();

        for(uint32_t batch = 0; batch < NUM_BATCHES; batch++){

            if(!compute_id && !cluster_id){
                printf("Batch %u\n", batch);
            }


            if(cluster_id){
                loss[0] = 0;
                if(snrt_is_dm_core()) {
                    // we need to zero initialize the weight and bias gradients
                    // for each new batch
                    dma_memset(weight_grads_cl1, 0, weight_mat_size);
                    dma_memset(bias_grads_cl1, 0, bias_mat_size);

                    
                    // printf("dma_memset test\n");
                    // for(int i=0; i < 10; i++){
                    //     printf("bias_grads_cl1[%u] = %f\n", i, bias_grads_cl1[i]);
                    // }
                    
                }
            }

            // We now loop through the images
            for(uint32_t image = 0; image < BATCH_SIZE; image++){
                
                // we calculate the pointer postion of the current image
                uint32_t volatile curr_img = (batch*BATCH_SIZE + image) * IN_CH; // --> Why * 2? 
                // if(!compute_id){
                //     printf("index of current image = %u\n", curr_img);
                // }
                // we calculate the pointer postion of the current target
                uint32_t volatile curr_target = batch*BATCH_SIZE + image;  
                if(!compute_id){
                    printf("index of current target = %u\n", curr_target);
                }

                // if(!compute_id){
                //     for(int i = 0; i < 40; i++){
                //         printf("target[%u] = %u\n", i, targets_dram[i]);
                //     }
                // }  

                // load a new image into the cluster 0 memory 
                if (snrt_is_dm_core() && cluster_id == 0) {
                    snrt_dma_start_tracking();
                    snrt_dma_txid_t txid_IMG = 
                            snrt_dma_start_1d(images,                                   // destination
                                            &images_dram[curr_img],                     // source
                                            n->dtype * IN_CH);                          // size
                    snrt_dma_wait_all();
                    // for(int i = 0; i < IN_CH; i++){
                    //     printf("image[%u][%u] = %f\n", image, i, images[i]);
                    // }
                    snrt_dma_stop_tracking();
                }

                snrt_cluster_hw_barrier();

                // load the respective target into cluster 1 memory
                if (snrt_is_dm_core() && cluster_id == 1) {
                    snrt_dma_start_tracking();
                    // On cluster 1 we load the labels which are needed for BP
                            snrt_dma_txid_t txid_targets = 
                                snrt_dma_start_1d(targets,                                   // destination
                                                &targets_dram[curr_target],                  // source
                                                sizeof(uint32_t));                           // size
                            
                            snrt_dma_wait_all();
                    snrt_dma_stop_tracking();
                } 

                if (snrt_is_dm_core() && cluster_id == 0) {
                    snrt_dma_start_tracking();
                    // On cluster 1 we load the labels which are needed for BP
                            snrt_dma_txid_t txid_targets = 
                                snrt_dma_start_1d(targets,                                   // destination
                                                &targets_dram[curr_target],                  // source
                                                sizeof(uint32_t));                           // size
                            
                            snrt_dma_wait_all();
                    snrt_dma_stop_tracking();
                } 

                snrt_cluster_hw_barrier();

                // we perform the forward pass on cluster 0
                if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num && cluster_id == 0) {

                    // if(!compute_id){
                    //     // printf("Forward pass on cluster 0\n");
                    //     for(uint32_t i = 0; i < n->OUT_CH; i++){
                    //         printf("%f ", weights_cl0[i]);
                    //         printf("\n");
                    //     }
                    // }
                    
                    // printf("target = %u\n", targets[0]);
                    uint32_t target = targets[0];
                    
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
                        
                        if(!compute_id && !RUN_RTL){
                            printf("[MNIST] FP64 FF start\n");
                        }

                        // TODO: remove core_sync
                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            feedforward_fp64n(n->IN_CH1, n->IN_CH2, div, 
                                            &weights_cl0[W_offset], ldW, &biases_cl0[b_offset], &activations_cl0[b_offset],
                                            ldB, images, ldI, compute_id);
                            benchmark_get_cycle();
                            softmax_activation_fp64n(n->IN_CH1, n->IN_CH2, div, 
                                        &weights_cl0[W_offset], ldW, &activations_cl0[b_offset], ldB,
                                        images, ldI, compute_id, compute_num, max);
                        } else {
                            // INFO: FP64 with SSRs
                            benchmark_get_cycle();
                            feedforward_fp64_ssrn(n->IN_CH1, n->IN_CH2, div, 
                                            &weights_cl0[W_offset], ldW, &biases_cl0[b_offset], &activations_cl0[b_offset],
                                            ldB, images, ldI, compute_id, setup_SSR);
                            benchmark_get_cycle();
                            softmax_activation_fp64_ssrn(n->IN_CH1, n->IN_CH2, div,
                                            &weights_cl0[W_offset], ldW, &activations_cl0[b_offset], ldB,
                                            images, ldI, compute_id, compute_num, max, setup_SSR);
                        }

                        if(!compute_id && !RUN_RTL){
                            printf("[MNIST] FP64 FF end\n");
                        }

                    }
                } else if (!snrt_is_compute_core() && cluster_id == 0){

                    if(RUN_FEEDFORWARD){
                        if(BASELINE){
                            // INFO: baseline
                            snrt_cluster_hw_barrier();
                            snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                            snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                            // snrt_cluster_hw_barrier();
                        } else {
                            // INFO: FP64 with SSRs
                            snrt_cluster_hw_barrier();
                            snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                            snrt_cluster_hw_barrier(); // --> HW barrier for SoftMax, commented out for RTL debug
                        }
                    } else {
                        if(!cluster_id){
                            printf("[MNIST] FP64 FF not run. \n");
                        }
                    }
                }

                // wait until clusters are synchronized to not
                // start gradient update until all activations are 
                // computed
                snrt_global_barrier();
                // INFO: replacing global barrier with custom barrier for RTL sims
                // snrt_generic_cluster_barrier(cluster_num*cluster_core_num);      

                // if(setup_SSR && RUN_GRADIENT_UPDATE){
                if(RUN_GRADIENT_UPDATE){
                    if(snrt_is_dm_core() && cluster_id==1) {
                        snrt_dma_start_tracking();
                        // WARN: make sure that pointer types are according to network precision
                        double *act_ptr = ((uint32_t)activations_cl1) - cluster_offset;
                        double *img_ptr = ((uint32_t)images) - cluster_offset;

                        // for SSRs we need to DMA transfer the cluster 0 data to cluster 1
                        snrt_dma_txid_t txid_activations = 
                            snrt_dma_start_1d(activations_cl1,                                 // destination
                                            act_ptr,                                       // source
                                            n->dtype * n->OUT_CH);                         // size

                        snrt_dma_txid_t txid_IMG = 
                            snrt_dma_start_1d(images,                                    // destination
                                            img_ptr,                                     // source
                                            n->dtype * IN_CH);        // size
                        
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

                    double *act_ptr = ((uint32_t)activations_cl1) - cluster_offset;
                    double *img_ptr = ((uint32_t)images) - cluster_offset;
                    

                    if(GET_ACCURACY && !RUN_RTL) {
                        uint32_t target = targets[0];

                        if(!compute_id){
                            uint32_t correct;
                            if(image == 0){
                                correct = 0;
                            }
                            double max_activation = act_ptr[0];
                            uint32_t max_activation_id = 0;
                            for (int preds = 0; preds < n->OUT_CH; preds++) {
                                if(act_ptr[preds] > max_activation){
                                    max_activation = act_ptr[preds];
                                    max_activation_id = preds;
                                }
                            }
                            printf("DEBUG CL1: target = %u\n", target);
                            printf("DEBUG CL1: pred = %u\n", max_activation_id);
                            // printf("DEBUG: max = %f\n", max_activation);
                            if(max_activation_id == target){
                                correct++;
                            }

                            printf("BATCH accuracy[%u][%u][%u] = %f %%\n", epoch, batch, image, ((double)correct / (double)BATCH_SIZE) * 100);
                        }
                    }

                    if(RUN_GRADIENT_UPDATE){
                        if(!compute_id && !RUN_RTL){
                            printf("[MNIST] FP64 GU start\n");
                        }
                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            gradient_update_fp64n(n->IN_CH1, n->IN_CH2, div, 
                                                &weight_grads_cl1[W_offset], ldW, 
                                                &bias_grads_cl1[b_offset], &activations_cl1[b_offset], 
                                                ldB, img_ptr, targets, ldI, compute_id, 
                                                loss, compute_num);
                            benchmark_get_cycle();
                            if(!compute_id){
                                if(image == 0){
                                    printf("BATCH average loss[%u] = %f\n", batch, loss[compute_id]);
                                } else {
                                    printf("BATCH average loss[%u] = %f\n", batch, loss[compute_id]/image);
                                }
                            }
                        } else {
                            // INFO: FP64 with SSRs
                            benchmark_get_cycle();
                            gradient_update_fp64_ssrn(n->IN_CH1, n->IN_CH2, div, 
                                                &weight_grads_cl1[W_offset], ldW, 
                                                &bias_grads_cl1[b_offset], &activations_cl1[b_offset], 
                                                ldB, images, targets, ldI, compute_id, 
                                                loss, compute_num, setup_SSR);
                            benchmark_get_cycle();
                        }
                        if(!compute_id && !RUN_RTL){
                            printf("[MNIST] FP64 GU end\n");
                        }
                    } else {
                        if(!compute_id && !RUN_RTL){
                            printf("[MNIST] FP64 GU not run. \n");
                        }
                    } // end of gradient update
                } else if (!snrt_is_compute_core() && cluster_id == 1){
                    if(RUN_GRADIENT_UPDATE){
                        if(BASELINE){
                            // INFO: baseline
                            snrt_cluster_hw_barrier();
                        } else {
                            // INFO: FP64 with SSRs
                            snrt_cluster_hw_barrier();
                        }
                    } else {
                        if(!cluster_id && !RUN_RTL){
                            printf("[MNIST] FP64 GU not run. \n");
                        }
                    }
                }

                snrt_global_barrier();

                /// Start of Training Step

                if(setup_SSR && RUN_TRAINING_STEP){ 
                // if(RUN_TRAINING_STEP){
                    // for SSRs we need to DMA transfer the cluster 1 data to cluster 0
                    if(snrt_is_dm_core() && cluster_id==0) {
                        // Discuss with GIM how to do DMA benchmarking
                        snrt_dma_start_tracking();
                        // WARN: make sure that pointer types are according to network precision
                        double *weight_grad_ptr = ((uint32_t)weight_grads_cl0) + cluster_offset;
                        double *bias_grad_ptr = ((uint32_t)biases_cl0) + cluster_offset;
                        snrt_dma_txid_t txid_WG = 
                            snrt_dma_start_2d(weight_grads_cl0,       // destination
                                            weight_grad_ptr,          // source
                                            n->dtype * IN_CH,    // size
                                            n->dtype * IN_CH ,   // destination stride
                                            n->dtype * IN_CH ,   // source stride
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

                    double *weight_grad_ptr = ((uint32_t)weights_cl0) + cluster_offset;
                    double *bias_grad_ptr = ((uint32_t)biases_cl0) + cluster_offset;

                    // if(!compute_id){
                    //     for(int i = 0; i < 10; i++){
                    //         printf("bias_grad_ptr[%u] = %f\n", i, bias_grad_ptr[i]);
                    //     }
                    // }

                    //TODO: load the LR from the network struct or via DRAM perloading
                    //*learning_rate = 0.5;

                    if(RUN_TRAINING_STEP){

                        if(!compute_id && !RUN_RTL){
                            printf("[MNIST] FP64 Training step start\n");
                            // for(uint32_t i = 150; i < 170; i++){
                            //     printf("%f\n", i, weights_cl0[i]);
                            // }
                        }

                        if(BASELINE){
                            // INFO: baseline
                            benchmark_get_cycle();
                            training_step_fp64n(n->IN_CH1, n->IN_CH2, div, 
                                                &weights_cl0[W_offset], &weight_grad_ptr[W_offset], ldW, 
                                                &biases_cl0[b_offset], &activations_cl0[b_offset], ldB, 
                                                compute_id, compute_num, 1);
                            benchmark_get_cycle();
                        } else {
                            // INFO: FP64 with SSRs
                            benchmark_get_cycle();
                            training_step_fp64_ssrn(n->IN_CH1, n->IN_CH2, div, 
                                                &weights_cl0[W_offset], &weight_grads_cl0[W_offset], ldW, 
                                                &biases_cl0[b_offset], &activations_cl0[b_offset], ldB, 
                                                compute_id, compute_num, 1, setup_SSR);
                            benchmark_get_cycle();
                        }

                        if(!compute_id && !RUN_RTL){
                            printf("[MNIST] FP64 Training step done\n");
                            // for(uint32_t i = 150; i < 170; i++){
                            //     printf("%f\n", i, weights_cl0[i]);
                            // }
                        }
                    }

                } else if(!snrt_is_compute_core() && cluster_id == 0){
                    if(BASELINE){
                        if(RUN_TRAINING_STEP){
                            // no HW barriers required for Baseline
                        } else {
                            if(!cluster_id && !RUN_RTL){
                                printf("[MNIST] FP64 Training Step not run. \n");
                            }
                        }
                    } else {
                        if(RUN_TRAINING_STEP){
                        } else {
                            if(!cluster_id && !RUN_RTL){
                                printf("[MNIST] FP64 Training Step not run. \n");
                            }
                        }
                    }
                }

                snrt_global_barrier();
                // End of Training Step
            }

            // /// Start of Training Step
            // snrt_global_barrier();
            // // INFO: replacing global barrier with custom barrier for RTL sims
            // // snrt_generic_cluster_barrier(cluster_num*cluster_core_num);

            // if(setup_SSR && RUN_TRAINING_STEP){ 
            // // if(RUN_TRAINING_STEP){
            //     // for SSRs we need to DMA transfer the cluster 1 data to cluster 0
            //     if(snrt_is_dm_core() && cluster_id==0) {
            //         // Discuss with GIM how to do DMA benchmarking
            //         snrt_dma_start_tracking();
            //         // WARN: make sure that pointer types are according to network precision
            //         double *weight_grad_ptr = ((uint32_t)weight_grads_cl0) + cluster_offset;
            //         double *bias_grad_ptr = ((uint32_t)biases_cl0) + cluster_offset;
            //         snrt_dma_txid_t txid_WG = 
            //             snrt_dma_start_2d(weight_grads_cl0,       // destination
            //                             weight_grad_ptr,          // source
            //                             n->dtype * IN_CH,    // size
            //                             n->dtype * IN_CH ,   // destination stride
            //                             n->dtype * IN_CH ,   // source stride
            //                             n->OUT_CH);               // repetitions

            //         snrt_dma_txid_t txid_BG = 
            //             snrt_dma_start_1d(activations_cl0,        // destination
            //                             bias_grad_ptr,            // source
            //                             n->dtype * n->OUT_CH);    // size

                    
            //         snrt_dma_wait_all();

            //         snrt_dma_stop_tracking();
            //     } 
            // }

            // snrt_cluster_hw_barrier();

            // // after looping through one batch of the dataset
            // // we update the biases and weights on Cluster 0

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

            //     double *weight_grad_ptr = ((uint32_t)weights_cl0) + cluster_offset;
            //     double *bias_grad_ptr = ((uint32_t)biases_cl0) + cluster_offset;

            //     // if(!compute_id){
            //     //     for(int i = 0; i < 10; i++){
            //     //         printf("bias_grad_ptr[%u] = %f\n", i, bias_grad_ptr[i]);
            //     //     }
            //     // }

            //     //TODO: load the LR from the network struct or via DRAM perloading
            //     //*learning_rate = 0.5;

            //     if(RUN_TRAINING_STEP){

            //         if(!compute_id && !RUN_RTL){
            //             printf("[MNIST] FP64 Training step start\n");
            //             // for(uint32_t i = 150; i < 170; i++){
            //             //     printf("%f\n", i, weights_cl0[i]);
            //             // }
            //         }

            //         if(BASELINE){
            //             // INFO: baseline
            //             benchmark_get_cycle();
            //             training_step_fp64n(n->IN_CH1, n->IN_CH2, div, 
            //                                 &weights_cl0[W_offset], &weight_grad_ptr[W_offset], ldW, 
            //                                 &biases_cl0[b_offset], &activations_cl0[b_offset], ldB, 
            //                                 compute_id, compute_num, 1);
            //             benchmark_get_cycle();
            //         } else {
            //             // INFO: FP64 with SSRs
            //             benchmark_get_cycle();
            //             training_step_fp64_ssrn(n->IN_CH1, n->IN_CH2, div, 
            //                                 &weights_cl0[W_offset], &weight_grads_cl0[W_offset], ldW, 
            //                                 &biases_cl0[b_offset], &activations_cl0[b_offset], ldB, 
            //                                 compute_id, compute_num, 1, setup_SSR);
            //             benchmark_get_cycle();
            //         }

            //         if(!compute_id && !RUN_RTL){
            //             printf("[MNIST] FP64 Training step done\n");
            //             // for(uint32_t i = 150; i < 170; i++){
            //             //     printf("%f\n", i, weights_cl0[i]);
            //             // }
            //         }
            //     }

            // } else if(!snrt_is_compute_core() && cluster_id == 0){
            //     if(BASELINE){
            //         if(RUN_TRAINING_STEP){
            //             // no HW barriers required for Baseline
            //         } else {
            //             if(!cluster_id && !RUN_RTL){
            //                 printf("[MNIST] FP64 Training Step not run. \n");
            //             }
            //         }
            //     } else {
            //         if(RUN_TRAINING_STEP){
            //         } else {
            //             if(!cluster_id && !RUN_RTL){
            //                 printf("[MNIST] FP64 Training Step not run. \n");
            //             }
            //         }
            //     }
            // }

            // snrt_global_barrier();
            // // End of Training Step
        } // End of batches

    } // End of epochs

}