// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "single_cluster.h"

#include "network.h"
#include "single_cluster_network.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

#define BASELINE 1

// define which kernel to run
#define FEEDFORWARD 0
#define GRADIENT_UPDATE 1
#define TRAINING_STEP 0

// define whether RTL simulation is used
#define RTL 1


void single_cluster(const network_single_cluster_t *n){

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

    uint32_t setup_SSR = 1;

    // Set the TCDM base address for the current cluster
    void *ptr = (void *)snrt_cluster_memory().start;
    

    // FEEDFORWARD: load weights, biases and image data from DRAM
    //              into the cluster TCDM

    if(FEEDFORWARD) {

        uint32_t weight_mat_size = (OUT_CH * IN_CH) * PREC;
        uint32_t bias_mat_size;
        if(PREC != 1) {
            bias_mat_size = OUT_CH * PREC;
        } else {
            bias_mat_size = OUT_CH * sizeof(float);
        }
        uint32_t image_size = IN_CH * PREC;
        uint32_t activations_mat_size = bias_mat_size;

        // data needed for feedforward
        void *weights;
        void *biases;
        void *images;
        void *activations;

        // configure the memory map for the cluster
        weights = ptr;
        ptr += weight_mat_size;
        biases = ptr;
        ptr += bias_mat_size;
        images = ptr;
        ptr += image_size;
        activations = ptr;
        ptr += activations_mat_size;

        if (snrt_is_dm_core()) {
            // start DMA tracking
            snrt_dma_start_tracking();

            snrt_dma_txid_t txid_biases = 
                snrt_dma_start_1d(biases,
                                  n->b,
                                  bias_mat_size);

            snrt_dma_txid_t txid_weights =
                snrt_dma_start_2d(weights,
                                  n->W,
                                  IN_CH * PREC,
                                  IN_CH * PREC,
                                  IN_CH * PREC,
                                  OUT_CH);

            snrt_dma_txid_t txid_images =
                snrt_dma_start_1d(images,
                                  n->images,
                                  image_size);

            // wait for DMA to finish
            snrt_dma_wait_all();
            // stop DMA tracking
            snrt_dma_stop_tracking();
        } // End of DMA transfer

        // Synchronize the cores
        snrt_cluster_hw_barrier();

        // Start of the actual computation
        if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {

            // determine the row offset at which current compute cluster is
            volatile uint32_t W_offset = compute_id * IN_CH;
            volatile uint32_t b_offset = compute_id;

            // determine how many columns per compute core
            volatile uint32_t div = OUT_CH % compute_num;
            if(div == 0){
                div = OUT_CH / compute_num;
            }

            // determine the row stride of each matrix    
            volatile uint32_t ldW = compute_num * IN_CH;
            volatile uint32_t ldB = compute_num;

            if(PREC == 8) {
                if(BASELINE) {
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Baseline 64-bit feedforward start\n");
                        }
                    }
                    benchmark_get_cycle();
                    single_cluster_feedforward_fp64(IN_CH, div, 
                            &((double *)weights)[W_offset], ldW, &((double *)biases)[b_offset], &((double *)activations)[b_offset],
                            ldB, (double *)images);
                    benchmark_get_cycle();
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Baseline 64-bit feedforward end\n");
                        }
                    }
                } else {
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Optimized 64-bit feedforward start\n");
                        }
                    }
                    benchmark_get_cycle();
                    single_cluster_feedforward_fp64_opt(IN_CH, div, 
                            &((double *)weights)[W_offset], ldW, &((double *)biases)[b_offset], &((double *)activations)[b_offset],
                            ldB, (double *)images, setup_SSR);
                    benchmark_get_cycle();
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Optimized 64-bit feedforward end\n");
                        }
                    }
                }
            }

        } else {
            // HW barrier of the DM core
            snrt_cluster_hw_barrier();
        } // End of the actual computation
    } // End of FEEDFORWARD

    // Gradient Update: load image, target and activation data from DRAM
    //                  into the cluster TCDM
    if (GRADIENT_UPDATE) {

        uint32_t weight_grads_mat_size = (OUT_CH * IN_CH) * PREC;
        uint32_t bias_grads_mat_size;
        if(PREC != 1) {
            bias_grads_mat_size = OUT_CH * PREC;
        } else {
            bias_grads_mat_size = OUT_CH * sizeof(float);
        }
        uint32_t image_size = IN_CH * PREC;
        uint32_t activations_mat_size = bias_grads_mat_size;
        uint32_t target_size = sizeof(uint32_t);
        uint32_t loss_size = PREC;

        // data needed for feedforward
        void *weight_grads;
        void *bias_grads;
        void *images;
        void *activations;
        void *loss;
        uint32_t *targets;

        // configure the memory map for the cluster
        weight_grads = ptr;
        ptr += weight_grads_mat_size;
        bias_grads = ptr;
        ptr += bias_grads_mat_size;
        images = ptr;
        ptr += image_size;
        activations = ptr;
        ptr += activations_mat_size;
        loss = ptr;
        ptr += loss_size;
        targets = ptr;
        ptr += target_size;

        if (snrt_is_dm_core()) { 
            // start DMA tracking
            snrt_dma_start_tracking();

            snrt_dma_txid_t txid_biases = 
                snrt_dma_start_1d(activations,
                                  n->b,
                                  bias_grads_mat_size);

            snrt_dma_txid_t txid_images =
                snrt_dma_start_1d(images,
                                  n->images,
                                  image_size);

            snrt_dma_txid_t txid_targets =
                snrt_dma_start_1d(targets,
                                  n->targets,
                                  target_size);

            // wait for DMA to finish
            snrt_dma_wait_all();
            // stop DMA tracking
            snrt_dma_stop_tracking();
        } // End of DMA transfer

        // Synchronize the cores
        snrt_cluster_hw_barrier();

        // Start of the actual computation
        if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {

            // determine the row offset at which current compute cluster is
            volatile uint32_t W_offset = compute_id * IN_CH;
            volatile uint32_t b_offset = compute_id;

            // determine how many columns per compute core
            volatile uint32_t div = OUT_CH % compute_num;
            if(div == 0){
                div = OUT_CH / compute_num;
            }

            // determine the row stride of each matrix    
            volatile uint32_t ldW = compute_num * IN_CH;
            volatile uint32_t ldB = compute_num;

            if(PREC == 8) {
                if(BASELINE) {
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Baseline 64-bit gradient update start\n");
                        }
                    }
                    benchmark_get_cycle();
                    single_cluster_gradient_update_fp64(IN_CH, div, 
                            &((double *)weight_grads)[W_offset], ldW, &((double *)bias_grads)[b_offset], &((double *)activations)[b_offset],
                            ldB, (double *)images, targets, compute_id, loss);
                    benchmark_get_cycle();
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Baseline 64-bit gradient update end\n");
                        }
                    }
                } else {
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Optimized 64-bit gradient update start\n");
                        }
                    }
                    benchmark_get_cycle();
                    single_cluster_gradient_update_fp64_opt(IN_CH, div, 
                            &((double *)weight_grads)[W_offset], ldW, &((double *)bias_grads)[b_offset], &((double *)activations)[b_offset],
                            ldB, (double *)images, targets, compute_id, loss, setup_SSR);
                    benchmark_get_cycle();
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Optimized 64-bit gradient update end\n");
                        }
                    }
                }
            }

        } else {
            // HW barrier of the DM core
            snrt_cluster_hw_barrier();
        } // End of the actual computation

    } // End of GRADIENT_UPDATE

    // TRAINING STEP: load weights, weight gradients from DRAM
    // WARN: For FP64, weights and weight gradients are the same
    if(TRAINING_STEP) {

        uint32_t weights_mat_size = (OUT_CH * IN_CH) * PREC;
        uint32_t weight_grads_mat_size = weights_mat_size;
        uint32_t biases_mat_size;
        if(PREC != 1) {
            biases_mat_size = OUT_CH * PREC;
        } else {
            biases_mat_size = OUT_CH * sizeof(float);
        }
        uint32_t bias_grads_mat_size = biases_mat_size;

        // data needed for feedforward
        void *weights;
        void *biases;
        void *weight_grads;
        void *bias_grads;

        // configure the memory map for the cluster
        weights = ptr;
        ptr += weights_mat_size;
        if(PREC != 8) {
            weight_grads = ptr;
            ptr += weight_grads_mat_size;
        } else {
            weight_grads = weights;
        }
        biases = ptr;
        ptr += biases_mat_size;
        bias_grads = ptr;
        ptr += bias_grads_mat_size;

        if (snrt_is_dm_core()) { 
            // start DMA tracking
            snrt_dma_start_tracking();

            snrt_dma_txid_t txid_weights =
                snrt_dma_start_2d(weights,
                                  n->W,
                                  IN_CH * PREC,
                                  IN_CH * PREC,
                                  IN_CH * PREC,
                                  OUT_CH);

            if(PREC != 8) {
                snrt_dma_txid_t txid_weight_grads = 
                    snrt_dma_start_2d(weight_grads,
                                      n->W_grads,
                                      IN_CH * PREC,
                                      IN_CH * PREC,
                                      IN_CH * PREC,
                                      OUT_CH);
            }

            snrt_dma_txid_t txid_biases = 
                snrt_dma_start_1d(biases,
                                  n->b,
                                  biases_mat_size);

            snrt_dma_txid_t txid_bias_grads = 
                snrt_dma_start_1d(bias_grads,
                                  n->b_grads,
                                  bias_grads_mat_size);

            // wait for DMA to finish
            snrt_dma_wait_all();
            // stop DMA tracking
            snrt_dma_stop_tracking();
        } // End of DMA transfer

        // Synchronize the cores
        snrt_cluster_hw_barrier();

        // Start of the actual computation
        if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {

            // determine the row offset at which current compute cluster is
            volatile uint32_t W_offset = compute_id * IN_CH;
            volatile uint32_t b_offset = compute_id;

            // determine how many columns per compute core
            volatile uint32_t div = OUT_CH % compute_num;
            if(div == 0){
                div = OUT_CH / compute_num;
            }

            // determine the row stride of each matrix    
            volatile uint32_t ldW = compute_num * IN_CH;
            volatile uint32_t ldB = compute_num;

            if(PREC == 8) {
                if(BASELINE) {
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Baseline 64-bit training step start\n");
                        }
                    }
                    benchmark_get_cycle();
                    single_cluster_training_step_fp64(IN_CH, div, 
                            &((double *)weights)[W_offset], &((double *)weight_grads)[W_offset], ldW, &((double *)biases)[b_offset], &((double *)bias_grads)[b_offset],
                            ldB);
                    benchmark_get_cycle();
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Baseline 64-bit training step end\n");
                        }
                    }
                } else {
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Optimized 64-bit training step start\n");
                        }
                    }
                    benchmark_get_cycle();
                    single_cluster_training_step_fp64_opt(IN_CH, div, 
                            &((double *)weights)[W_offset], &((double *)weight_grads)[W_offset], ldW, &((double *)biases)[b_offset], &((double *)bias_grads)[b_offset],
                            ldB, setup_SSR);
                    benchmark_get_cycle();
                    if(!RTL) {
                        if(!compute_id) {
                            printf("Optimized 64-bit training step end\n");
                        }
                    }
                }
            }
        } // End of the actual computation
    } // End of TRAINING_STEP
}