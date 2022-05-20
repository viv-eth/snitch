// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include <math.h>

#include "celoss_layer.h"

#include "layer.h"
#include "celoss.h"

#include "printf.h"
#include "snrt.h"

void celoss_fp64(const cel_layer *l, uint32_t dim1, uint32_t dim2, 
                 uint32_t dim3, double* IN, uint32_t ldIn, 
                 uint32_t compute_id, double* sum){

    uint32_t i = 0;
    double sum_core = 0.0;

    uint32_t target = l->target;
    double pred = IN[target*ldIn];


    //first we determine the summed exponent of the activations from the FW pass
    for (uint32_t d2 = 0; d2 < dim2; d2++) {
        for (uint32_t d3 = 0; d3 < dim3; d3++) {
            if(IN[d2 + d3 * ldIn]){
                sum_core += exp(IN[d2 + d3 * ldIn]); 
            }
        }

    }

    snrt_cluster_hw_barrier();

    // now we have to reduce on a single core
    sum[compute_id] = sum_core;
    //snrt_cluster_hw_barrier();

    if(compute_id == 0){

        double sum_cluster = 0.0;
        // determine sum over all cores
        for(;i < 8; i++){
            sum_cluster += sum[i];
        }

        printf("sum: %f\n", sum_cluster); //INFO: works

        // determine the activation value at the target

        printf("Value at pred: %f\n", pred); //INFO: works

        double log_l = exp(pred)/sum_cluster;

        //printf("Log loss: %f\n", log_l);

        double ce_loss = 0.0 - log(log_l); //INFO: works

        printf("CE Loss: %f\n", ce_loss);
    }

    //snrt_cluster_hw_barrier();
    
}

void celoss_fp32(const cel_layer *l, uint32_t dim1, uint32_t dim2, 
                 uint32_t dim3, float* IN, uint32_t ldIn, 
                 uint32_t compute_id, float* sum){

    uint32_t i = 0;
    float sum_core = 0.0;

    uint32_t target = l->target;
    float pred = IN[target*ldIn];


    //first we determine the summed exponent of the activations from the FW pass
    for (uint32_t d2 = 0; d2 < dim2; d2++) {
        for (uint32_t d3 = 0; d3 < dim3; d3++) {
            if(IN[d2 + d3 * ldIn]){
                sum_core += exp(IN[d2 + d3 * ldIn]); 
            }
        }

    }

    snrt_cluster_hw_barrier();

    // now we have to reduce on a single core
    sum[compute_id] = sum_core;
    snrt_cluster_hw_barrier();

    if(compute_id == 0){

        float sum_cluster = 0.0;
        // determine sum over all cores
        for(;i < 8; i++){
            sum_cluster += sum[i];
        }

        printf("sum: %f\n", sum_cluster); //INFO: works

        // determine the activation value at the target

        printf("Value at pred: %f\n", pred); //INFO: works not

        float log_l = exp(pred)/sum_cluster;

        //printf("Log loss: %f\n", log_l);

        float ce_loss = 0.0 - log(log_l); //INFO: works not

        printf("CE Loss: %f\n", ce_loss);
    }

    //snrt_cluster_hw_barrier();
    
}

// void celoss_fp16(const cel_layer *l, uint32_t dim1, uint32_t dim2, 
//                  uint32_t dim3, __fp16* IN, uint32_t ldIn, 
//                  uint32_t compute_id, __fp16* sum){

//     uint32_t i = 0;
//     float sum_core = 0.0;

//     uint32_t target = l->target;
//     __fp16 pred = IN[target*ldIn];


//     //first we determine the summed exponent of the activations from the FW pass
//     for (uint32_t d2 = 0; d2 < dim2; d2++) {
//         for (uint32_t d3 = 0; d3 < dim3; d3++) {
//             if(IN[d2 + d3 * ldIn]){
//                 sum_core += exp(IN[d2 + d3 * ldIn]); 
//             }
//         }

//     }

//     snrt_cluster_hw_barrier();

//     // now we have to reduce on a single core
//     sum[compute_id] = sum_core;
//     snrt_cluster_hw_barrier();
    
//     if(compute_id == 0){

//         float sum_cluster = 0.0;
//         // determine sum over all cores
//         for(;i < 8; i++){
//             sum_cluster += sum[i];
//         }

//         printf("sum: %f\n", sum_cluster); //INFO: works

//         // determine the activation value at the target

//         printf("Value at pred: %f\n", pred); //INFO: works not

//         float log_l = exp(pred)/sum_cluster;

//         //printf("Log loss: %f\n", log_l);

//         float ce_loss = 0.0 - log(log_l); //INFO: works not

//         printf("CE Loss: %f\n", ce_loss);
//     }

//     //snrt_cluster_hw_barrier();
    
// }



// void celoss_fp64_ssr(const cel_layer *l, uint32_t dim1, uint32_t dim2, 
//                  uint32_t dim3, double* IN, uint32_t ldIn, 
//                  uint32_t compute_id, double* sum){

//     uint32_t i = 0;
//     double sum_core = 0.0;
//     double in_val;

//     uint32_t target = l->target;
//     double pred = IN[target*ldIn];

//     uint32_t dim2_padded = dim2 + 4;

//     __builtin_ssr_setup_1d_r(0, 0, dim3 - 1, sizeof(double), IN);
//     __builtin_ssr_setup_1d_w(1, 0, dim3 - 1, sizeof(double), IN);

//     __builtin_ssr_enable();


//     for (int i = 0; i < dim2 ; i++) {
//         in_val = __builtin_ssr_pop(0);
//         printf("%val: f\n", in_val);
//         //__builtin_ssr_push(1, in_val);
//     }
    

//     //first we determine the summed exponent of the activations from the FW pass
    
//     // for (uint32_t d = 0; d < dim2; d++) {
//     //     in_val = __builtin_ssr_pop(0);
//     //     if(in_val){
//     //         sum_core += exp(in_val); 
//     //     }
//     // }
//     __builtin_ssr_disable();

//     printf("test in val: %f\n", in_val);
//     snrt_cluster_hw_barrier();

//     // now we have to reduce on a single core
//     sum[compute_id] = sum_core;
//     //snrt_cluster_hw_barrier();

//     if(compute_id == 0){

//         double sum_cluster = 0.0;
//         // determine sum over all cores
//         for(;i < 8; i++){
//             sum_cluster += sum[i];
//         }

//         printf("sum: %f\n", sum_cluster); //INFO: works

//         // determine the activation value at the target

//         printf("Value at pred: %f\n", pred); //INFO: works

//         double log_l = exp(pred)/sum_cluster;

//         //printf("Log loss: %f\n", log_l);

//         double ce_loss = 0.0 - log(log_l); //INFO: works

//         printf("CE Loss: %f\n", ce_loss);
//     }
    
// }