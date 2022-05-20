// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "softmax_layer.h"

#include "layer.h"
#include "softmax.h"
#include "printf.h"
#include "snrt.h"

// Padding of innermost dimension of a Matrix
// Useful for preventing banking conflicts between cores
// that are accessing different rows of the matrix
#define MAT_ROW_PADDING 0

// Padding in between matrices A, B for preventing
// banking conflicts in the beginning
#define MAT_PADDING 8

//FIXME: for now type of l_checksum has to be changed manually...

void softmax_layer(const sm_layer *l, void *l_checksum){

    uint32_t cluster_num = snrt_cluster_num();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_num = snrt_cluster_compute_core_num();
    uint32_t compute_id = snrt_cluster_compute_core_idx();


    uint32_t mat_IN_size = (l->dim1 * (l->dim2 + MAT_ROW_PADDING) + MAT_PADDING) * l->dtype;
    uint32_t mat_OUT_size = mat_IN_size; // INFO: matrix output dimension same as that of input
    uint32_t max_data_size = compute_num * l->dtype;
    
    // start of memory mapping
    // INFO: pointer types are also always changed manually
    //FP 16 - working
    void *ptr = (__fp16 *)snrt_cluster_memory().start;
    __fp16 *mat_IN = ptr;
    ptr += mat_IN_size;
    __fp16 *mat_OUT = ptr;
    ptr += mat_OUT_size;
    __fp16 *max_data = ptr;
    ptr += max_data_size;

    // INFO: FP32 - working
    /*void *ptr = (float *)snrt_cluster_memory().start;
    float *mat_IN = ptr;
    ptr += mat_IN_size;
    float *mat_OUT = ptr;
    ptr += mat_OUT_size;
    float *max_data = ptr;
    ptr += max_data_size;*/

    // INFO: FP64 - working
    /*double *ptr = (double *)snrt_cluster_memory().start;
    double *mat_IN = ptr;
    ptr += mat_IN_size;
    double *mat_OUT = ptr;
    ptr += mat_OUT_size;
    double *max_data = ptr;
    ptr += max_data_size;*/

    snrt_global_barrier();

    // start DMA transfer of matrix data
    if (snrt_is_dm_core()) {
        snrt_dma_txid_t txid_IN =
            snrt_dma_start_2d(mat_IN, l->IN, l->dtype * l->dim2,
                              l->dtype * (l->dim2 + MAT_ROW_PADDING),
                              l->dtype * l->dim2, l->dim1);

        snrt_dma_txid_t txid_OUT =
            snrt_dma_start_2d(mat_OUT, l->OUT, l->dtype * l->dim2,
                              l->dtype * (l->dim2 + MAT_ROW_PADDING),
                              l->dtype * l->dim2, l->dim1);

        // wait until each DMA transfer done
        snrt_dma_wait_all();
    }

    // synchronize the clusters
    snrt_cluster_hw_barrier();


    if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {
        const uint32_t setup_SSR = 1;
            // determine the row offset at which current compute cluster is
            volatile uint32_t IN_offset =
                compute_id * (l->dim2 + MAT_ROW_PADDING);
            volatile uint32_t ldIn =
                compute_num * (l->dim2 + MAT_ROW_PADDING);

            // M dim Should be actually M / # (compute cores), but fails if
            // not multiples of each other due to integer divide
            if(l->dtype == FP64){
                benchmark_get_cycle();
                //INFO: works
                //softmax_fp64(l, l->dim1 / 5, l->dim2, &mat_IN[IN_offset], ldIn, compute_id, max_data);
                //INFO: works
                softmax_fp64_ssr(l, l->dim1 / 5, l->dim2, &mat_IN[IN_offset], ldIn, compute_id, max_data, setup_SSR); 
                benchmark_get_cycle();
            } else if(l->dtype == FP32){
                benchmark_get_cycle();
                //INFO: works
                softmax_fp32_ssr(l->dim1 / 5, l->dim2, &mat_IN[IN_offset], ldIn, compute_id, max_data, setup_SSR);
                benchmark_get_cycle();
            } else if(l->dtype == FP16){
                benchmark_get_cycle();
                // INFO: WIP
                softmax_fp16_ssr(l->dim1 / 5, l->dim2, &mat_IN[IN_offset], ldIn, compute_id, max_data, setup_SSR);
                benchmark_get_cycle();
            }
    } else{
        // for ninth core (DMA) core 
        // INFO: all cores should have same amount of barriers, HW barrier less computational heavy than others
        snrt_cluster_hw_barrier();
        //snrt_cluster_hw_barrier();
    }
    snrt_cluster_hw_barrier();

    uint32_t errors = 0;

    if (compute_id == 0) {
        uint32_t err_cnt = 0;
        double err;
        double temp_err = 0;

        if (l->dtype == FP64) {
            for (uint32_t d1 = 0; d1 < l->dim1; d1++) {
                for(uint32_t d2 = 0; d2 < l->dim2; d2++){
                    if(((double *)mat_OUT)[d1 * l->dim2 + d2]>1e-20){
                        err_cnt++;
                        temp_err = fabs(((double *)mat_IN)[d1 * l->dim2 + d2]/((double *)mat_OUT)[d1 * l->dim2 + d2] - 1);
                    }
                }
            }

        } else if (l->dtype == FP32) {
            for (uint32_t d1 = 0; d1 < l->dim1; d1++) {
                for(uint32_t d2 = 0; d2 < l->dim2; d2++){
                    if(((float *)mat_OUT)[d1 * l->dim2 + d2]>1e-20){
                        err_cnt++;
                        temp_err = fabs(((float *)mat_IN)[d1 * l->dim2 + d2]/((float *)mat_OUT)[d1 * l->dim2 + d2] - 1);
                    }
                }
            }
        } else if (l->dtype == FP16) {
            // for (uint32_t d1 = 0; d1 < l->dim1; d1++) {
            //     for(uint32_t d2 = 0; d2 < l->dim2; d2++){
            //         if(((__fp16 *)mat_OUT)[d1 * l->dim2 + d2]>1e-20){
            //             err_cnt++;
            //             temp_err = fabs(((__fp16 *)mat_IN)[d1 * l->dim2 + d2]/((__fp16 *)mat_OUT)[d1 * l->dim2 + d2] - 1);
            //         }
            //     }
            // }
        }
        
        //err = 100*(temp_err/err_cnt);

        //printf("Mean relative error: %f %%\n", err);
    }

}