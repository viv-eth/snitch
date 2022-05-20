// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "celoss_layer.h"

#include "layer.h"
#include "celoss.h"
#include "printf.h"
#include "snrt.h"

// Padding of innermost dimension of a Matrix
// Useful for preventing banking conflicts between cores
// that are accessing different rows of the matrix
#define MAT_ROW_PADDING 4

// Padding in between matrices A, B for preventing
// banking conflicts in the beginning
#define MAT_PADDING 8

void celoss_layer(const cel_layer *l, void *l_checksum){

    uint32_t cluster_num = snrt_cluster_num();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_num = snrt_cluster_compute_core_num();
    uint32_t compute_id = snrt_cluster_compute_core_idx();


    uint32_t mat_IN_size = (l->IN_CH2 * (l->OUT_CH + MAT_ROW_PADDING) + MAT_PADDING) * l->dtype;
    uint32_t sum_data_size = compute_num;
    
    // start of memory mapping
    //FIXME: pointer types are also always changed manually
    // FIXME: FP 16 - not working, gives Seg Fault
    /*void *ptr = (__fp16 *)snrt_cluster_memory().start;
    __fp16 *mat_IN = ptr;
    ptr += mat_IN_size;
    __fp16 *sum_data = ptr;
    ptr += sum_data_size;*/

    // FP32 - working
    void *ptr = (float *)snrt_cluster_memory().start;
    float *mat_IN = ptr;
    ptr += mat_IN_size;
    float *sum_data = ptr;
    ptr += sum_data_size;

    // FP64 - working
    /*void *ptr = (double *)snrt_cluster_memory().start;
    double *mat_IN = ptr;
    ptr += mat_IN_size;
    double *sum_data = ptr;
    ptr += sum_data_size;*/

    snrt_global_barrier();

    // start DMA transfer of matrix data
    if (snrt_is_dm_core()) {
        snrt_dma_txid_t txid_IN =
            snrt_dma_start_2d(mat_IN, l->IN, l->dtype * l->IN_CH2,
                              l->dtype * (l->IN_CH2 + MAT_ROW_PADDING),
                              l->dtype * l->IN_CH2, l->OUT_CH);

        // wait until each DMA transfer done
        snrt_dma_wait_all();
    }

    // synchronize the clusters
    snrt_cluster_hw_barrier();


    if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {
        const uint32_t setup_SSR = 1;
            // determine the row offset at which current compute cluster is
            volatile uint32_t IN_offset = compute_id * l->OUT_CH;
            // determine row stride of each matrix (i.e. space between subsequent elements)
            volatile uint32_t ldIn = l->IN_CH2 + MAT_ROW_PADDING;
            
            printf("IN matrix at offset: IN[%u] = %f\n", IN_offset, mat_IN[IN_offset]);

            // M dim Should be actually M / # (compute cores), but fails if
            // not multiples of each other due to integer divide
            if(l->dtype == FP64){
                benchmark_get_cycle();
                celoss_fp64(l, l->IN_CH1, l->IN_CH2, l->OUT_CH / 5, &mat_IN[IN_offset], ldIn, compute_id, sum_data); //INFO: works
                //celoss_fp64_ssr(l, l->IN_CH1, l->IN_CH2, l->OUT_CH / 5, &mat_IN[IN_offset], ldIn, compute_id, sum_data);
                //TODO: implement FP64 celoss
                benchmark_get_cycle();
            } else if(l->dtype == FP32){
                benchmark_get_cycle();
                celoss_fp32(l, l->IN_CH1, l->IN_CH2, l->OUT_CH / 5, &mat_IN[IN_offset], ldIn, compute_id, sum_data);
                benchmark_get_cycle();
            } else if(l->dtype == FP16){
                benchmark_get_cycle();
                //TODO: celoss_fp16(l, l->IN_CH1, l->IN_CH2, l->OUT_CH / 5, &mat_IN[IN_offset], ldIn, compute_id, sum_data);
                benchmark_get_cycle();
            }
    } else{
        // for ninth core (DMA) core INFO: all cores should have same amount of barriers, HW barrier less computational heavy than others
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
    }
    snrt_cluster_hw_barrier();

}