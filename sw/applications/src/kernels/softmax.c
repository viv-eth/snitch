// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include <math.h>
#include <stdlib.h>

#include "softmax_layer.h"

#include "layer.h"
#include "softmax.h"

#include "printf.h"
#include "snrt.h"

typedef float v2f32 __attribute__((vector_size(8)));
typedef __fp16 v4f16 __attribute__((vector_size(8)));
typedef char v8f8 __attribute__((vector_size(8)));


void softmax_fp64(const sm_layer *l, uint32_t dim1, uint32_t dim2, 
                  double* IN, uint32_t ldIn, uint32_t compute_id, double *max){

    uint32_t i = 0;
    double sum = 0.0;
    double max_core = IN[0];

    for (uint32_t d1 = 0; d1 < dim1; d1++) {
        for (uint32_t d2 = 0; d2 < dim2; d2++) {
            if (IN[d2 + d1 * ldIn] > max_core) {
                max_core = IN[d2 + d1 * ldIn];
            }
        }

    }

    max[compute_id] = max_core;
    snrt_cluster_hw_barrier();

    printf("Max value of compute core %u is %f\n", compute_id, max_core);

    double max_global = max[0];

    if(compute_id == 0){
        // determine global maximum
        for(;i < 8; i++){
            if(max_global < max[i]){
                max_global = max[i];
            }
        }

        max[0] = max_global;

        uint32_t cnt = 0;

        uint32_t dim2_padded = l->dim2 + 4;

        // normalize and perform softmax
        //Q: can the loop be parallelized on all cores? I.e. with shared pointer 
        //--> Tim: overhead from parallelizing bigger than gain
        for (uint32_t d1 = 0; d1 < l->dim1; d1++) {
            //FIXME: account properly for padding
            for (uint32_t d2 = 0; d2 < dim2_padded; d2++) {
                if(IN[d2 + d1*dim2_padded] != 0){
                    IN[d2 + d1*dim2_padded] = exp(IN[d2 + d1*dim2_padded] - max_global);
                    sum += IN[d2 + d1*dim2_padded];
                } else {
                    IN[d2 + d1*dim2_padded] = 0.0;
                }
            }
        }

        for (uint32_t d1 = 0; d1 < l->dim1; d1++) {
            for (uint32_t d2 = 0; d2 < dim2_padded; d2++) {
                IN[d2 + d1*dim2_padded] /= sum; //INFO: yields same results as C-model, but noticed weird behaviour
                //Q: Maybe due to normalization? 
            }
        }
    }

    //snrt_cluster_hw_barrier();
}

void softmax_fp64_ssr(const sm_layer *l, uint32_t dim1, uint32_t dim2, double* IN, uint32_t ldIn, uint32_t compute_id, double *max, uint32_t setup_SSR) {
    
    
    uint32_t i = 0;
    double sum = 0.0;
    double max_core = IN[0];
    double comp_val;

    uint32_t dim2_padded = l->dim2 + 4;

    // Setup SSRs
    // IN[d2 + d1*dim2_padded]
    // INFO:
    // Repetition will perform the loop n times, i.e. pop on read register push
    // will be repeated n times
    __builtin_ssr_setup_1d_r(0, 0, dim2_padded -1, sizeof(double), IN);
    __builtin_ssr_setup_1d_w(1, 0, dim2_padded -1, sizeof(double), IN);

    __builtin_ssr_enable();

    for (uint32_t d = 0; d < dim1*dim2_padded; d++){
        comp_val = __builtin_ssr_pop(0);
        if(comp_val > max_core){
            max_core = comp_val;
        }
    }

    __builtin_ssr_disable();

    max[compute_id] = max_core;

    snrt_cluster_hw_barrier();

    printf("Max value of compute core %u is %f\n", compute_id, max_core);

    double max_global = max[0];

    if(compute_id == 0){
        // determine global maximum
        for(;i < 8; i++){
            if(max_global < max[i]){
                max_global = max[i];
            }
        }

        max[0] = max_global;

        uint32_t cnt = 0;

        uint32_t dim2_padded = l->dim2 + 4;

        // normalize and perform softmax
        //Q: can the loop be parallelized on all cores? I.e. with shared pointer 
        //--> Tim: overhead from parallelizing bigger than gain
        for (uint32_t d1 = 0; d1 < l->dim1; d1++) {
            //FIXME: account properly for padding
            for (uint32_t d2 = 0; d2 < dim2_padded; d2++) {
                if(IN[d2 + d1*dim2_padded] != 0){
                    IN[d2 + d1*dim2_padded] = exp(IN[d2 + d1*dim2_padded] - max_global);
                    sum += IN[d2 + d1*dim2_padded];
                } else {
                    IN[d2 + d1*dim2_padded] = 0.0;
                }
            }
        }

        for (uint32_t d1 = 0; d1 < l->dim1; d1++) {
            for (uint32_t d2 = 0; d2 < dim2_padded; d2++) {
                IN[d2 + d1*dim2_padded] /= sum; //INFO: yields same results as C-model, but noticed weird behaviour
                //Q: Maybe due to normalization? 
            }
        }
    }

}

//INFO: FP32 with SSRs implementation gives now same results as Golden Model C reference - yey
void softmax_fp32_ssr(uint32_t dim1, uint32_t dim2, float* IN, uint32_t ldIn, uint32_t compute_id, float *max, uint32_t setup_SSR){
    
    float sum = 0.0;
    float max_core = IN[0];
    uint32_t i = 0;

    // Start of SSR region.
    register volatile double ft0 asm("ft0"); // stores IN
    asm volatile("" : "=f"(ft0));

    snrt_ssr_loop_2d(SNRT_SSR_DM0, dim2, dim1, sizeof(float), sizeof(float)*ldIn);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, IN); // stored in ft0

    snrt_ssr_enable();


    for (uint32_t d1 = 0; d1 < dim1 - 1; d1++) {
        for (uint32_t d2 = 0; d2 < dim2; d2++) {
            // if (IN[d2 + d1*ldIn] > max_core) {
            //     max_core = IN[d2 + d1 * ldIn];
            // } // reference
            asm volatile(
                        "fmv.s      fs2, ft0 \n"
                        "flt.s      t0, %[max_core], fs2\n"
                        "bnez       t0, 1f\n"
                        "beqz       t0, 2f\n"
                        "1: \n"     
                        "fmv.s      %[max_core], fs2 \n"
                        "2: \n"
                        : [ max_core ] "+f"(max_core)::"ft0");
        }

    }

    // End of SSR region.
    snrt_ssr_disable();

    max[compute_id] = max_core;
    snrt_cluster_hw_barrier();

    //printf("Max value of compute core %u is %f\n", compute_id, max_core);

    float max_global = max[0];

    if(compute_id == 0){
        // determine global maximum
        for(;i < 8; i++){
            if(max_global < max[i]){
                max_global = max[i];
            }
        }

        max[0] = max_global;

        //FIXME: account properly for number of compute cores in parallelized dimension
        // at the moment only hard coded
        for (uint32_t d1 = 0; d1 < dim1*5; d1++) {
            for (uint32_t d2 = 0; d2 < dim2; d2++) {
                if(IN[d2 + d1*dim2] != 0){
                    IN[d2 + d1*dim2] = exp(IN[d2 + d1*dim2] - max_global);
                    sum += IN[d2 + d1*dim2];
                } else {
                    IN[d2 + d1*dim2] = 0.0;
                }
            }
        }

        for (uint32_t d1 = 0; d1 < dim1*5; d1++) {
            for (uint32_t d2 = 0; d2 < dim2; d2++) {
                IN[d2 + d1*dim2] /= sum; 
            }
        }
    }
}



// INFO: not working, sth off with instruction ordering of the compiler ?
void softmax_fp16_ssr(uint32_t dim1, uint32_t dim2, __fp16* IN, uint32_t ldIn, uint32_t compute_id, __fp16 *max, uint32_t setup_SSR){
    
    __fp16 sum = 0.0;
    __fp16 max_core = IN[0];
    float fp32_max_core = max_core;
    uint32_t i = 0;

    // Start of SSR region.
    register volatile double ft0 asm("ft0"); // stores IN
    register volatile double ft1 asm("ft1");
    asm volatile("" : "=f"(ft0));
    asm volatile("" : "=f"(ft1));


    snrt_ssr_loop_2d(SNRT_SSR_DM0, dim2, dim1, sizeof(__fp16), sizeof(__fp16)*ldIn);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, IN); // stored in ft0

    snrt_ssr_enable();

    //FIXME: proper data handling of DIM1, since memory wraps around
    for (uint32_t d1 = 0; d1 < dim1 - 1; d1++) {
        for (uint32_t d2 = 0; d2 < dim2; d2++) {
            asm volatile(
                        "fcvt.s.h        fs1, ft0  \n"
                        "flt.s           t0, %[max_core], fs1 \n"
                        : [ max_core ] "+f"(fp32_max_core)
                        : 
                        :"ft0", "ft1");
        }

    }

    // End of SSR region.
    snrt_ssr_disable();

    printf("checkpoint\n");

    // max[compute_id] = max_core;
    snrt_cluster_hw_barrier();

}

