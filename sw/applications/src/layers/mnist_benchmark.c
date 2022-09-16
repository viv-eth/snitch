// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_benchmark.h"

#include "network.h"
#include "mnist_benchmark_network.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

#define BASELINE 0

void mnist_benchmark(const network_benchmark_t *n){

    uint32_t cluster_num = snrt_cluster_num(); 
    uint32_t cluster_core_num = snrt_cluster_core_num();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_num = snrt_cluster_compute_core_num(); // Number of compute cores per cluster 
    uint32_t global_compute_num = snrt_global_core_num(); // Total cores incl. DM core per cluster 
    uint32_t compute_id = snrt_cluster_compute_core_idx();
    uint32_t dm_id = snrt_cluster_dm_core_idx();
    uint32_t global_compute_id = snrt_global_core_idx(); // Core ID of each core on all clusters

    uint32_t IN_CH = n->IN_CH;
    uint32_t OUT_CH = n->OUT_CH;
    uint32_t PREC = n->dtype;

    // if(cluster_id == 0 && compute_id == 0){
    //     printf("precision: %d\n", PREC);
    // }

    uint32_t setup_SSR = 1;

    
    uint32_t weight_mat_size = (OUT_CH * IN_CH) * PREC;
    uint32_t bias_mat_size;
    if(PREC != 1) {
        bias_mat_size = OUT_CH * PREC;
    } else {
        bias_mat_size = OUT_CH * sizeof(float);
    }
    uint32_t act_mat_size = bias_mat_size;
    uint32_t act_fp32_mat_size = OUT_CH * sizeof(float);
    uint32_t image_size = IN_CH * PREC;
    uint32_t max_size = PREC;
    uint32_t max_float_size = sizeof(float);
    uint32_t target_size = sizeof(uint32_t);
    uint32_t loss_size = PREC;

    // cluster 0 variabels:
    void *weights_cl0;
    void *weight_grads_cl0;
    void *biases_cl0;
    void *images;
    void *activations_cl0;
    void *max;
    // this variable is only used for FP8
    float *activations_fp32_cl0;

    // cluster 1 variabels:
    void *weights_cl1; 
    void *weight_grads_cl1;
    void *bias_grads_cl1;
    void *activations_cl1;
    // this variable is only used for FP8
    float *activations_fp32_cl1;
    void *loss;
    uint32_t *targets;

    void *ptr = (void *)snrt_cluster_memory().start;
    // void *ptr_start = ptr;
    if(cluster_id == 0){
        weights_cl0 = ptr;
        ptr += weight_mat_size;
        biases_cl0 = ptr;
        ptr += bias_mat_size;
        images = ptr;
        ptr += image_size;
        activations_cl0 = ptr;
        ptr += act_mat_size;
        if(PREC == 1){
            activations_fp32_cl0 = ptr;
            ptr += act_fp32_mat_size;
        }
        max = ptr;
        if(PREC == 1){
            ptr += max_float_size;
        }
        else{
            ptr += max_size;
        }
        ptr += max_size;
    } else if (cluster_id == 1){
        weight_grads_cl1 = ptr;
        ptr += weight_mat_size;
        bias_grads_cl1 = ptr;
        ptr += bias_mat_size;
        images = ptr;
        ptr += image_size; 
        activations_cl1 = ptr;
        ptr += act_mat_size;
        if(PREC == 1){
            activations_fp32_cl1 = ptr;
            ptr += act_fp32_mat_size;
        }
        targets = ptr;
        ptr += target_size; 
        loss = ptr;
        ptr += loss_size;
    } 

    // cluster offset in an Occamy quadrant
    uint32_t cluster_offset = 0x00040000;

    if (snrt_is_dm_core() && cluster_id == 0) {
        snrt_dma_start_tracking();
                // load initial biases from Golden Model into Cluster 0 memory
                snrt_dma_txid_t txid_B = 
                    snrt_dma_start_1d(biases_cl0,                 // destination
                                    n->b,                     // source
                                    PREC * OUT_CH);    // size
                snrt_dma_wait_all();

                // load weight data into Cluster 0 memory
                snrt_dma_txid_t txid_W = 
                    snrt_dma_start_2d(weights_cl0,                 // destination
                                    n->W,                          // source
                                    PREC * IN_CH,         // size
                                    PREC * IN_CH ,        // destination stride
                                    PREC * IN_CH ,        // source stride
                                    OUT_CH);                    // repetitions

                snrt_dma_txid_t txid_IMG = 
                    snrt_dma_start_1d(images,                                   // destination
                                    n->images,                                  // source
                                    PREC * IN_CH);       // size

        // wait until each DMA transfer done
        snrt_dma_stop_tracking();

    }

    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core() && cluster_id == 1) {
        snrt_dma_start_tracking();
        // On cluster 1 we load the labels which are needed for BP
                snrt_dma_txid_t txid_targets = 
                    snrt_dma_start_1d(targets,                                   // destination
                                    n->targets,                  // source
                                    sizeof(uint32_t));                           // size
                
                
                // WARN: dma memset fails for fp16 (discuss with GIM)
                // dma_memset(weight_grads_cl1, 0, weight_mat_size);
                // dma_memset(bias_grads_cl1, 0, bias_mat_size);
                
                snrt_dma_wait_all();
        snrt_dma_stop_tracking();
    }


    snrt_cluster_hw_barrier();

    if(cluster_id){
        ((__fp16 *)loss)[0] = 0;
    }
    if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num && cluster_id == 0) {
        
        // determine the row offset at which current compute cluster is
        volatile uint32_t W_offset = compute_id * IN_CH;
        volatile uint32_t b_offset = compute_id;

        volatile uint32_t div = OUT_CH % compute_num;
        if(div == 0){
            div = OUT_CH / compute_num;
        }

        // determine the row stride of each matrix    
        volatile uint32_t ldW = compute_num * IN_CH;
        volatile uint32_t ldB = compute_num;

        if(PREC == 8) {
            if(BASELINE) {
                benchmark_get_cycle();
                benchmark_feedforward_fp64(IN_CH, div, 
                        &((double *)weights_cl0)[W_offset], ldW, &((double *)biases_cl0)[b_offset], &((double *)activations_cl0)[b_offset],
                        ldB, (double *)images);
                benchmark_get_cycle();
            } else {
                benchmark_get_cycle();
                benchmark_feedforward_fp64_ssrn(IN_CH, div, 
                        &((double *)weights_cl0)[W_offset], ldW, &((double *)biases_cl0)[b_offset], &((double *)activations_cl0)[b_offset],
                        ldB, (double *)images, setup_SSR);
                benchmark_get_cycle();
            }
            benchmark_softmax_activation_fp64(div, 
                                    &((double *)activations_cl0)[b_offset], ldB,
                                    compute_id, compute_num, max);
        } else if (PREC == 4) {
            benchmark_get_cycle();
            benchmark_feedforward_fp32_opt(IN_CH, div, 
                    &((float *)weights_cl0)[W_offset], ldW, &((float *)biases_cl0)[b_offset], &((float *)activations_cl0)[b_offset],
                    ldB, (float *)images, setup_SSR);
            benchmark_get_cycle();
            benchmark_softmax_activation_fp32(div, 
                                    &((float *)activations_cl0)[b_offset], ldB,
                                    compute_id, compute_num, max);
        } else if (PREC == 2) {
            benchmark_get_cycle();
            benchmark_feedforward_fp16_opt(IN_CH, div, 
                    &((__fp16 *)weights_cl0)[W_offset], ldW, &((__fp16 *)biases_cl0)[b_offset], &((__fp16 *)activations_cl0)[b_offset],
                    ldB, (__fp16 *)images, setup_SSR);
            benchmark_get_cycle();
            benchmark_softmax_activation_fp16(div, 
                                    &((__fp16 *)activations_cl0)[b_offset], ldB,
                                    compute_id, compute_num, max);
        } else if (PREC == 1) {
            benchmark_get_cycle();
            benchmark_feedforward_fp8_opt(IN_CH, div, 
                    &((char *)weights_cl0)[W_offset], ldW, &((char *)biases_cl0)[b_offset],
                    ldB, (char *)images, setup_SSR, &activations_fp32_cl0[b_offset]);
            benchmark_get_cycle();
            benchmark_softmax_activation_fp32_ex(div,
                &activations_fp32_cl0[b_offset], ldB, compute_id, compute_num, max);
        }
    } else {
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
    }

    snrt_global_barrier();

    if(snrt_is_dm_core() && cluster_id==1) {

        void *act_ptr = ((uint32_t)activations_cl1) - cluster_offset;
        void *act_fp32_ptr;
        if(PREC == 1) {
            act_fp32_ptr = ((uint32_t)activations_fp32_cl1) - cluster_offset;
        }
        void *img_ptr = ((uint32_t)images) - cluster_offset;
        snrt_dma_start_tracking();
        // for SSRs we need to DMA transfer the cluster 0 data to cluster 1
        if(PREC != 1) {
            snrt_dma_txid_t txid_activations = 
            snrt_dma_start_1d(activations_cl1,                                 // destination
                            act_ptr,                                       // source
                            PREC * OUT_CH);                         // size
        } else {
            snrt_dma_txid_t txid_act = 
                snrt_dma_start_1d(activations_fp32_cl1,                                   // destination
                                act_fp32_ptr,                  // source
                                sizeof(float) * OUT_CH);                           // size
        }

        snrt_dma_txid_t txid_IMG = 
            snrt_dma_start_1d(images,                                    // destination
                            img_ptr,                                     // source
                            PREC * IN_CH);        // size
        
        snrt_dma_wait_all();

        snrt_dma_stop_tracking();

    }

    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num && cluster_id == 1) {

        // determine the row offset at which current compute cluster is
        volatile uint32_t W_offset = compute_id * IN_CH;
        volatile uint32_t b_offset = compute_id;

        volatile uint32_t div = OUT_CH % compute_num;
        if(div == 0){
            div = OUT_CH / compute_num;
        }

        // determine the row stride of each matrix    
        volatile uint32_t ldW = compute_num * IN_CH;
        volatile uint32_t ldB = compute_num;

        void *act_ptr = ((uint32_t)activations_cl1) - cluster_offset;
        void *img_ptr = ((uint32_t)images) - cluster_offset;
            
        if(PREC == 8) {
            if(BASELINE) {
                // INFO: FP64 baseline
                benchmark_get_cycle();
                benchmark_gradient_update_fp64(IN_CH, div, 
                                    &((double *)weight_grads_cl1)[W_offset], ldW, 
                                    &((double *)bias_grads_cl1)[b_offset], &((double *)activations_cl1)[b_offset], 
                                    ldB, (double *)images, targets, compute_id, 
                                    loss);
                benchmark_get_cycle();
            } else {
                // INFO: FP64 with SSRs
                benchmark_get_cycle();
                benchmark_gradient_update_fp64_ssr(IN_CH, div, 
                                    &((double *)weight_grads_cl1)[W_offset], ldW, 
                                    &((double *)bias_grads_cl1)[b_offset], &((double *)activations_cl1)[b_offset], 
                                    ldB, (double *)images, targets, compute_id, 
                                    loss, setup_SSR);
                benchmark_get_cycle();
            }
        } else if (PREC == 4) {
            // INFO: FP32 with SSRs
            benchmark_get_cycle();
            benchmark_gradient_update_fp32_opt(IN_CH, div, 
                                &((float *)weight_grads_cl1)[W_offset], ldW, 
                                &((float *)bias_grads_cl1)[b_offset], &((float *)activations_cl1)[b_offset], 
                                ldB, (float *)images, targets, compute_id, 
                                loss, setup_SSR);
            benchmark_get_cycle();
        } else if (PREC == 2) {
            // INFO: FP16 with SSRs
            benchmark_get_cycle();
            benchmark_gradient_update_fp16_opt(IN_CH, div, 
                                &((__fp16 *)weight_grads_cl1)[W_offset], ldW, 
                                &((__fp16 *)bias_grads_cl1)[b_offset], &((__fp16 *)activations_cl1)[b_offset], 
                                ldB, (__fp16 *)images, targets, compute_id, 
                                loss, setup_SSR);
            benchmark_get_cycle();
        } else if (PREC == 1) {
            // INFO: FP8 with SSRs
            benchmark_get_cycle();
            benchmark_gradient_update_fp8_opt(IN_CH, div, 
                                &((char *)weight_grads_cl1)[W_offset], ldW, 
                                &((float *)bias_grads_cl1)[b_offset], &activations_fp32_cl1[b_offset], 
                                ldB, (char *)images, targets, compute_id, 
                                loss, setup_SSR);
            benchmark_get_cycle();
        }
    } else if (!snrt_is_compute_core() && cluster_id == 1){
        snrt_cluster_hw_barrier();
    } 

    snrt_global_barrier();

    if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num && cluster_id == 0) {
        
        // determine the row offset at which current compute cluster is
        volatile uint32_t W_offset = compute_id * IN_CH;
        volatile uint32_t b_offset = compute_id;

        // Calculate number of rows for each compute core
        volatile uint32_t div = OUT_CH % compute_num;
        if(div == 0){
            div = OUT_CH / compute_num;
        }


        // determine the row stride of each matrix    
        volatile uint32_t ldW = compute_num * IN_CH;
        volatile uint32_t ldB = compute_num;

        void *weight_grad_ptr = ((uint32_t)weights_cl0) + cluster_offset;
        void *bias_grad_ptr = ((uint32_t)biases_cl0) + cluster_offset;

        if(PREC == 8) {
            if(BASELINE) {
                benchmark_get_cycle();
                benchmark_training_step_fp64(IN_CH, div, 
                                    &((double *)weights_cl0)[W_offset], &((double *)weight_grad_ptr)[W_offset], ldW, 
                                    &((double *)biases_cl0)[b_offset], &((double *)bias_grad_ptr)[b_offset], ldB);
                benchmark_get_cycle();
            } else {
                // INFO: FP64 with SSRs
                benchmark_get_cycle();
                benchmark_training_step_fp64_ssr(IN_CH, div, 
                                    &((double *)weights_cl0)[W_offset], &((double *)weight_grad_ptr)[W_offset], ldW, 
                                    &((double *)biases_cl0)[b_offset], &((double *)bias_grad_ptr)[b_offset], ldB, setup_SSR);
                benchmark_get_cycle();
            }
        } else if (PREC == 4) {
            // INFO: FP32 with SSRs
            benchmark_get_cycle();
            benchmark_training_step_fp32_opt(IN_CH, div, 
                                &((float *)weights_cl0)[W_offset], &((float *)weight_grad_ptr)[W_offset], ldW, 
                                &((float *)biases_cl0)[b_offset], &((float *)bias_grad_ptr)[b_offset], ldB, setup_SSR);
            benchmark_get_cycle();
        } else if (PREC == 2) {
            // INFO: FP16 with SSRs
            benchmark_get_cycle();
            benchmark_training_step_fp16_opt(IN_CH, div, 
                                &((__fp16 *)weights_cl0)[W_offset], &((__fp16 *)weight_grad_ptr)[W_offset], ldW, 
                                &((__fp16 *)biases_cl0)[b_offset], &((__fp16 *)bias_grad_ptr)[b_offset], ldB, setup_SSR);
            benchmark_get_cycle();
        } else if (PREC == 1) {
            // INFO: FP8 with SSRs
            benchmark_get_cycle();
            benchmark_training_step_fp8_opt(IN_CH, div, 
                &((char *)weights_cl0)[W_offset], &((char *)weight_grad_ptr)[W_offset], ldW, &((float *)biases_cl0)[b_offset], &((float *)bias_grad_ptr)[b_offset],
                ldB, setup_SSR);
            benchmark_get_cycle();
        }
    } 
}