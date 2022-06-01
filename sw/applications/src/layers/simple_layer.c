// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "simple_layer.h"

#include "layer.h"
#include "linear.h"
#include "printf.h"
#include "snrt.h"

// Padding of innermost dimension of a Matrix
// Useful for preventing banking conflicts between cores
// that are accessing different rows of the matrix
#define MAT_ROW_PADDING 0

// Padding in between matrices A, B for preventing
// banking conflicts in the beginning
#define MAT_PADDING 0

void simple_layer(const simpl_layer *l, void *l_checksum){

    uint32_t cluster_num = snrt_cluster_num();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_num = snrt_cluster_compute_core_num();
    uint32_t compute_id = snrt_cluster_compute_core_idx();

    uint32_t mat_A_size = (l->M * (l->K + MAT_ROW_PADDING) + MAT_PADDING) * l->dtype;
    uint32_t mat_B_size = (l->K + MAT_ROW_PADDING) * l->N * l->dtype;
    uint32_t mat_C_size = (l->M * l->N) * l->dtype;

    void *ptr = (double *)snrt_cluster_memory().start;
    double *mat_A = ptr;
    ptr += mat_A_size;
    double *mat_C = ptr;
    ptr += mat_C_size;
    double *mat_B = ptr;
    ptr += mat_B_size;

    snrt_global_barrier();

    // start DMA transfer of matrix data
    if (snrt_is_dm_core()) {
        snrt_dma_txid_t txid_A =
            snrt_dma_start_2d(mat_A, l->A, l->dtype * l->K,
                              l->dtype * (l->K + MAT_ROW_PADDING),
                              l->dtype * l->K, l->M);
        snrt_dma_txid_t txid_B =
            snrt_dma_start_2d(mat_B, l->B, l->dtype * l->K,
                              l->dtype * (l->K + MAT_ROW_PADDING),
                              l->dtype * l->K, l->N);

        snrt_dma_txid_t txid_C = snrt_dma_start_1d(
            mat_C, l->C, l->dtype * l->M * l->N);

        // wait until each DMA transfer done
        snrt_dma_wait_all();
    }

    // synchronize the clusters
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {
        const uint32_t setup_SSR = 1;
            // determine the row offset at which current compute cluster is
            volatile uint32_t A_offset =
                compute_id * (l->K + MAT_ROW_PADDING);
            volatile uint32_t C_offset =
                compute_id * l->N;
            // determine row stride of each matrix (i.e. space between subsequent elements)
            volatile uint32_t ldA =
                compute_num * (l->K + MAT_ROW_PADDING);
            volatile uint32_t ldB = (l->N + MAT_ROW_PADDING);
            volatile uint32_t ldC = l->N * compute_num;

            // M dim Should be actually M / # (compute cores), but fails if
            // not multiples of each other due to integer divide
            if(l->dtype == FP64){
                benchmark_get_cycle();
                simple_fp64(l->M / compute_num, l->N, l->K, &mat_A[A_offset], ldA,
                mat_B, ldB, &mat_C[C_offset], ldC, l->ALPHA, compute_id);
                benchmark_get_cycle();
            } else {
                printf("Not implemented yet.\n");
            }

            snrt_cluster_hw_barrier();

    } else {
        snrt_cluster_hw_barrier();
    }

    snrt_cluster_hw_barrier();

    uint32_t errors = 0;

        if (compute_id == 0) {
            if (l->dtype == FP64) {
                for (uint32_t m = 0; m < l->M; m++) {
                    double checksum = ((double*)l_checksum)[m];
                    double sum = 0.0;
                    for (uint32_t n = 0; n < l->N; n++) {
                        sum += ((double *)mat_C)[m * l->N + n];
                    }
                    if (fabs(sum - checksum) > 0.001) {
                        errors++;
                    }
                    printf("Total sum[%u]: %f\n", m, sum);
                    printf("Checksum[%u]: %f\n", m, checksum);
                }
            } else {
                printf("Not implemented yet\n");
            }
        }
}