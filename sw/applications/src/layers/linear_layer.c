// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "linear_layer.h"

#include "layer.h"
#include "linear.h"
#include "printf.h"
#include "snrt.h"

// Padding of innermost dimension of a Matrix
// Useful for preventing banking conflicts between cores
// that are accessing different rows of the matrix
#define MAT_ROW_PADDING 4

// Padding in between matrices A, B for preventing
// banking conflicts in the beginning
#define MAT_PADDING 8

//FIXME: for now type of l_checksum has to be changed manually...

void linear_layer(const lin_layer *l, void *l_checksum){

    uint32_t cluster_num = snrt_cluster_num();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_num = snrt_cluster_compute_core_num();
    uint32_t compute_id = snrt_cluster_compute_core_idx();

    uint32_t mat_W_size = (l->M * (l->K + MAT_ROW_PADDING) + MAT_PADDING) * l->dtype;
    uint32_t mat_X_size = (l->K + MAT_ROW_PADDING) * l->N * l->dtype;
    uint32_t mat_B_size = (l->M * l->N) * l->dtype;
    
    // start of memory mapping
    //FIXME: pointer types are also always changed manually
    //FP 16 - not working yet
    /*void *ptr = (__fp16 *)snrt_cluster_memory().start;
    __fp16 *mat_W = ptr;
    ptr += mat_W_size;
    __fp16 *mat_X = ptr;
    ptr += mat_X_size;
    __fp16 *mat_B = ptr;
    ptr += mat_B_size;*/

    // FP32 - working
    /*void *ptr = (float *)snrt_cluster_memory().start;
    float *mat_W = ptr;
    ptr += mat_W_size;
    float *mat_X = ptr;
    ptr += mat_X_size;
    float *mat_B = ptr;
    ptr += mat_B_size;*/

    void *ptr = (double *)snrt_cluster_memory().start;
    double *mat_W = ptr;
    ptr += mat_W_size;
    double *mat_X = ptr;
    ptr += mat_X_size;
    double *mat_B = ptr;
    ptr += mat_B_size;

    snrt_global_barrier();

    // start DMA transfer of matrix data
    if (snrt_is_dm_core()) {
        snrt_dma_txid_t txid_W =
            snrt_dma_start_2d(mat_W, l->W, l->dtype * l->K,
                              l->dtype * (l->K + MAT_ROW_PADDING),
                              l->dtype * l->K, l->M);
        snrt_dma_txid_t txid_X =
            snrt_dma_start_2d(mat_X, l->X, l->dtype * l->K,
                              l->dtype * (l->K + MAT_ROW_PADDING),
                              l->dtype * l->K, l->N);

        snrt_dma_txid_t txid_B = snrt_dma_start_1d(
            mat_B, l->B, l->dtype * l->M * l->N);

        // wait until each DMA transfer done
        snrt_dma_wait_all();
    }

    // synchronize the clusters
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {
        const uint32_t setup_SSR = 1;
            // determine the row offset at which current compute cluster is
            volatile uint32_t W_offset =
                compute_id * (l->K + MAT_ROW_PADDING);
            volatile uint32_t B_offset =
                compute_id * l->N;
            // determine row stride of each matrix (i.e. space between subsequent elements)
            volatile uint32_t ldW =
                compute_num * (l->K + MAT_ROW_PADDING);
            volatile uint32_t ldX = (l->K + MAT_ROW_PADDING);
            volatile uint32_t ldB = l->N * compute_num;
            printf("W matrix at offset: W[%u] = %f\n", W_offset, mat_W[W_offset]);
            printf("B matrix at offset: B[%u] = %f\n", B_offset, mat_B[B_offset]);

            printf("ldW: %u, ldB: %u, ldI: %u\n", ldW, ldB, ldX);

            // M dim Should be actually M / # (compute cores), but fails if
            // not multiples of each other due to integer divide
            if(l->dtype == FP64){
                benchmark_get_cycle();
                linear_fp64(l->M / 5 , l->N,
                                l->K, &mat_W[W_offset], ldW, l->TW,
                                mat_X, ldX, l->TX, &mat_B[B_offset], l->TB, ldB);
                /*linear_fp64_ssr_frep(l->M / 5 , l->N,
                                l->K, &mat_W[W_offset], ldW, l->TW,
                                mat_X, ldX, l->TX, &mat_B[B_offset], ldB, l->TB, 
                                setup_SSR);*/
                benchmark_get_cycle();
            } else if(l->dtype == FP32){
                benchmark_get_cycle();
                linear_fp32simd_ssr_frep(l->M / 5 , l->N,
                                l->K, &mat_W[W_offset], ldW, l->TW,
                                mat_X, ldX, l->TX, &mat_B[B_offset], ldB, l->TB, 
                                setup_SSR);
                benchmark_get_cycle();
            } else if(l->dtype == FP16){
                benchmark_get_cycle();
                linear_fp16simd_ssr_frep(l->M / 5 , l->N,
                                l->K, &mat_W[W_offset], ldW, l->TW,
                                mat_X, ldX, l->TX, &mat_B[B_offset], ldB, l->TB, 
                                setup_SSR);
                benchmark_get_cycle();
            }
    }
    snrt_cluster_hw_barrier();

    uint32_t errors = 0;

    if (compute_id == 0) {
        if (l->dtype == FP64) {
            for (uint32_t m = 0; m < l->M; m++) {
                double checksum = ((double*)l_checksum)[m];
                double sum = 0.0;
                for (uint32_t n = 0; n < l->N; n++) {
                    sum += ((double *)mat_B)[m * l->N + n];
                }
                if (fabs(sum - checksum) > 0.001) {
                    errors++;
                }
                printf("Total sum[%u]: %f\n", m, sum);
                printf("Checksum[%u]: %f\n", m, checksum);
            }
        } else if (l->dtype == FP32) {
            for (uint32_t m = 0; m < l->M; m++) {
                float checksum = ((float*)l_checksum)[m];
                float sum = 0.0;
                for (uint32_t n = 0; n < l->N; n++) {
                    sum += ((float *)mat_B)[m * l->N + n];
                }
                if (fabs(sum - checksum) > 0.001) {
                    errors++;
                }
                printf("Total sum[%u]: %f\n", m, sum);
                printf("Checksum[%u]: %f\n", m, checksum);
            }
        } else if (l->dtype == FP16) {
            for (uint32_t m = 0; m < l->M; m++) {
                __fp16 checksum = ((__fp16*)l_checksum)[m];
                float sum = 0.0; // Why initialized as float in GEMM
                for (uint32_t n = 0; n < l->N; n++) {
                    sum += ((__fp16 *)mat_B)[m * l->N + n];
                }
                if (fabs(sum - checksum) > 0.05) {
                    errors++;
                }
                printf("Total sum[%u]: %f\n", m, sum);
                printf("Checksum[%u]: %f\n", m, checksum);
            }
        }
        printf("%d/%d Errors\n", errors, l->M * l->N);
    }

}