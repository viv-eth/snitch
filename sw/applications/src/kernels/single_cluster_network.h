// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

# pragma once

#include "printf.h"
#include "snrt.h"
#include "utils.h"

typedef float v2f32 __attribute__((vector_size(8)));
typedef __fp16 v4f16 __attribute__((vector_size(8)));
typedef char v8f8 __attribute__((vector_size(8)));

typedef union {
    double f64;
    v2f32 vec;
} v2s;
typedef union {
    double f64;
    v4f16 vec;
} v4s;
typedef union {
    double f64;
    v8f8 vec;
} v8s;

#define FLT_MIN 1E-37
#define FLT_MAX 1E+37

static inline double my_fabs(double x) {
    if(x < 0) {
        return -x;
    } else {
        return x;
    }
}

static inline double my_exp(double x) 
{ 
    const double epsilon = 1e-7; 
    double sum = 0.0; 
    int n = 0; 
    double factorial = 1; 
    double power=1.0; 
    double term; 
    do { 
        term = power/factorial; 
        sum += term; 
        n += 1; 
        power *= x; 
        factorial *=n; 
    } while (my_fabs(term)>=epsilon); 
    return sum; 
} 


// INFO: start of FP64 baseline network implementation
static inline void single_cluster_feedforward_fp64(uint32_t IN_CH, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image){
    
    // Linear layer: OUT = X * W^T + B
    for (uint32_t out = 0; out < OUT_CH; out++) {
        register double acc = biases[ldB * out];
        for(uint32_t in = 0; in < IN_CH; in++){
            acc += image[in] * weights[out * ldW + in];
        }
        // OUT is accumulated in activations 
        activations[ldB * out] = acc;
        // printf("Single Cluster: FEEDFORWARD FP64 Baseline: acc = %f\n", activations[ldB * out]);  
    }

    snrt_cluster_hw_barrier();

}

static inline void single_cluster_gradient_update_fp64(uint32_t IN_CH, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, 
                uint32_t compute_id, double *loss){

    
    double b_grad_update = 0.0;
    // double W_grad_update = 0.0;
    volatile uint32_t idx_eff;
    // volatile uint32_t W_idx_eff;
    
    // Commented out for RTL
    // double W_checksum = 0.0;


    // get the value saved at target address
    int32_t target_n = *target;
    
    // the effective index is the iteration index of the biases variable
    // across all entries
    for(uint32_t out = 0; out < OUT_CH; out++){
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        // W_checksum = 0.0;

        for(uint32_t in = 0; in < IN_CH; in++){
            weight_grads[out * ldW + in] = b_grad_update * image[in]; 
            // W_checksum += W_grad_update;
        }
            
        bias_grads[ldB * out] = b_grad_update; 
        // printf("Single Cluster: GU FP64 Baseline W_checksum[%u] = %f\n", idx_eff, W_checksum);
        // printf("Single Cluster: GU FP64 Baseline bias_grads[%u] = %f\n", idx_eff, b_grad_update);
    }

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

}

static inline void single_cluster_training_step_fp64(uint32_t IN_CH, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB){

    float lr = 0.5;
    // double W_checksum = 0.0;

    for(uint32_t out = 0; out < OUT_CH; out++){
        biases[ldB * out] -= lr * bias_grads[ldB * out];
        // W_checksum = 0.0;

        // printf("Single Cluster: TS FP64 Baseline updated bias = %f\n", biases[ldB * out]);

        for(uint32_t in = 0; in < IN_CH; in++){
            weights[out * ldW + in] -= lr * weight_grads[out * ldW + in];
            // W_checksum += weights[out * ldW + in];
        }

        // printf("Single Cluster: TS FP64 Baseline updated weight_checksum = %f\n", W_checksum);
    }

}



