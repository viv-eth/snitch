// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_benchmark.h"

#include "network.h"
#include "mnist_benchmark_network.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

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

    if(cluster_id == 0 && compute_id == 0){
        printf("precision: %d\n", PREC);
    }

    uint32_t setup_SSR = 1;

    
    uint32_t weight_mat_size = (OUT_CH * IN_CH) * PREC;
    uint32_t bias_mat_size = OUT_CH * PREC;
    uint32_t act_mat_size = bias_mat_size;
    uint32_t image_size = IN_CH * PREC;
    uint32_t max_size = PREC;
    uint32_t target_size = sizeof(uint32_t);
    uint32_t loss_size = PREC;

    // cluster 0 variabels:
    double *weights_cl0;
    double *weight_grads_cl0;
    double *biases_cl0;
    double *images;
    double *activations_cl0;
    double *max;

    // cluster 1 variabels:
    double *weights_cl1; 
    double *weight_grads_cl1;
    double *bias_grads_cl1;
    double *activations_cl1;
    double *loss;
    uint32_t *targets;

    void *ptr = (double *)snrt_cluster_memory().start;
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
        max = ptr;
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

    if (snrt_is_dm_core() && cluster_id == 1) {
        snrt_dma_start_tracking();
        // On cluster 1 we load the labels which are needed for BP
                snrt_dma_txid_t txid_targets = 
                    snrt_dma_start_1d(targets,                                   // destination
                                    n->targets,                  // source
                                    sizeof(uint32_t));                           // size
                
                dma_memset(weight_grads_cl1, 0, weight_mat_size);
                dma_memset(bias_grads_cl1, 0, bias_mat_size);
                
                snrt_dma_wait_all();
        snrt_dma_stop_tracking();
    }

    if(cluster_id){
        loss[0] = 0;
    }

    snrt_cluster_hw_barrier();

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

        benchmark_get_cycle();
        // benchmark_feedforward_fp64(IN_CH, div, 
        //                 &weights_cl0[W_offset], ldW, &biases_cl0[b_offset], &activations_cl0[b_offset],
        //                 ldB, images, compute_id);
        benchmark_feedforward_fp64_ssrn(IN_CH, div, 
                &weights_cl0[W_offset], ldW, &biases_cl0[b_offset], &activations_cl0[b_offset],
                ldB, images, compute_id,
                setup_SSR);
        benchmark_get_cycle();
        benchmark_softmax_activation_fp64(div, 
                                &activations_cl0[b_offset], ldB,
                                compute_id, compute_num, max);
    } else {
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
    }

    snrt_global_barrier();

    if(snrt_is_dm_core() && cluster_id==1) {

        snrt_dma_start_tracking();
        // WARN: make sure that pointer types are according to network precision
        double *act_ptr = ((uint32_t)activations_cl1) - cluster_offset;
        double *img_ptr = ((uint32_t)images) - cluster_offset;

        // for SSRs we need to DMA transfer the cluster 0 data to cluster 1
        snrt_dma_txid_t txid_activations = 
            snrt_dma_start_1d(activations_cl1,                                 // destination
                            act_ptr,                                       // source
                            PREC * OUT_CH);                         // size

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

        double *act_ptr = ((uint32_t)activations_cl1) - cluster_offset;
        double *img_ptr = ((uint32_t)images) - cluster_offset;
            
        // INFO: FP64 with SSRs
        benchmark_get_cycle();
        benchmark_gradient_update_fp64_ssr(IN_CH, div, 
                            &weight_grads_cl1[W_offset], ldW, 
                            &bias_grads_cl1[b_offset], &activations_cl1[b_offset], 
                            ldB, images, targets, compute_id, 
                            loss, setup_SSR);
        benchmark_get_cycle();
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

        double *weight_grad_ptr = ((uint32_t)weights_cl0) + cluster_offset;
        double *bias_grad_ptr = ((uint32_t)biases_cl0) + cluster_offset;

        
                // INFO: FP64 with SSRs
                benchmark_get_cycle();
                // WARN: assigning "wrong" values for RTL benchmarking
                benchmark_training_step_fp64_ssr(IN_CH, div, 
                                    &weights_cl0[W_offset], &weight_grad_ptr[W_offset], ldW, 
                                    &biases_cl0[b_offset], &bias_grad_ptr[b_offset], ldB, 
                                    compute_id, setup_SSR);
                benchmark_get_cycle();
    } 
}