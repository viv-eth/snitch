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
#define FEEDFORWARD 1
#define GRADIENT_UPDATE 0
#define TRAINING_STEP 0


void single_cluster(const network_benchmark_t *n){

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
    

    // FEEDFORWARD: load weights, biases and image data from DRAM
    //              into the cluster TCDM

    printf("Cluster ID: %d, Cluster Core ID: %d\n", cluster_id, compute_id);

    snrt_cluster_hw_barrier();
}