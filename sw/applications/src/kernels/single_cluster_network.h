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

// INFO: start of the optimized FP64 network implementation
//// Feedforward Step
static inline void single_cluster_feedforward_fp64_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t setup_SSR){

    // INFO: due to a compiler bug we need to reserve the registers for the SSR
    //       otherwise it will use them for stack operations breaking the stream(s)
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));

    register double acc = 0.0;

    // const uint32_t unroll = 4;
    // register double acc_tot[unroll] = {0.0, 0.0, 0.0, 0.0};

    for (uint32_t out = 0; out < OUT_CH; out++) {

        
        // SSR strides and bounds only have to be configured
        // once in the beginning
        if (setup_SSR) {

            // setup of input data (MNIST image)
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH, 
                            sizeof(double));
            
            // setup of weights
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH,  
                            sizeof(double));
        }
        // we need to read the image for every new iteration
        // of a core, because otherwise it will evaluate to
        // all zeros due to the stream semantics
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]);
        // benchmark_get_cycle();
        // Start of SSR region
        snrt_ssr_enable();
        
        /// NON-UNROLLED VERSION
        acc = biases[ldB * out];
        asm volatile(
            "frep.o      %[n_frep], 1, 0, 0 \n"
            "fmadd.d     %[acc], ft0, ft1, %[acc] \n"
        : [ acc ] "+f"(acc)
        : [ n_frep ] "r"(IN_CH - 1)
        :"ft0", "ft1", "ft2");
        /// NON-UNROLLED VERSION

        /// UNROLLED VERSION
        // acc_tot[0] = biases[ldB * out];
        // acc_tot[1] = 0;
        // acc_tot[2] = 0;
        // acc_tot[3] = 0;
        // asm volatile(
        //     "frep.o      %[n_frep], 4, 0, 0 \n"
        //     "fmadd.d     %[acc_0], ft0, ft1, %[acc_0] \n"
        //     "fmadd.d     %[acc_1], ft0, ft1, %[acc_1] \n"
        //     "fmadd.d     %[acc_2], ft0, ft1, %[acc_2] \n"
        //     "fmadd.d     %[acc_3], ft0, ft1, %[acc_3] \n"
        // : [ acc_0 ] "+f"(acc_tot[0]), [ acc_1 ] "+f"(acc_tot[1]), [ acc_2 ] "+f"(acc_tot[2]), [ acc_3 ] "+f"(acc_tot[3])
        // : [ n_frep ] "r"(IN_CH / 4  - 1)
        // :"ft0", "ft1", "ft2");
        /// UNROLLED VERSION
        

        /// NON-UNROLLED VERSION
        activations[ldB * out] = acc;
        // acc = 0.0;
        /// NON-UNROLLED VERSION

        /// UNROLLED VERSION
        // activations[ldB * out] = acc_tot[0] + acc_tot[1] + acc_tot[2] + acc_tot[3];
        // acc_tot[0] = 0;
        // acc_tot[1] = 0;
        // acc_tot[2] = 0;
        // acc_tot[3] = 0;
        /// UNROLLED VERSION

        // End of SSR region.
        snrt_ssr_disable();
        // INFO: after disabling the SSRs we can free the registers
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));
        // benchmark_get_cycle();

    }

    // for (uint32_t out = 0; out < OUT_CH; out++) {
    //      printf("Benchmark FEEDFORWARD FP64 with SSRs: acc[%u] = %f\n", activations[ldB * out]);
    // }   
    snrt_cluster_hw_barrier(); 
} 


//// Gradient Update
static inline void single_cluster_gradient_update_fp64_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, 
                uint32_t compute_id, double *loss, uint32_t setup_SSR){

    // INFO: due to a compiler bug we need to reserve the registers for the SSR
    //       otherwise it will use them for stack operations breaking the stream(s)
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));

    
    register double b_grad_update;
    
    // double W_checksum = 0.0;
    volatile uint32_t idx_eff;
    // volatile uint32_t W_idx_eff;

    // double loss_val = 0.0;

    // get the value saved at target address
    uint32_t target_n = *target;
    
    // compute the loss
    // if(!compute_id){
    //     loss_val = 0.0 - log(activations[target_n - compute_id]);
    //     printf("GU current loss = %.5f\n", loss_val);
    //     printf("GU activation[target = %u] = %.15f\n", target_n - compute_id, activations[target_n - compute_id]);
    //     loss[0] += loss_val;
    // } 


    // the effective index is the iteration index of the biases variable
    // across all entries
    for(uint32_t out = 0; out < OUT_CH; out++){

        // W_checksum = 0.0;
        idx_eff = compute_id + ldB * out;

        // SSR strides and bounds only have to be configured
        // once in the beginning
        if (setup_SSR) {

            // SSR READ setup of input data (MNIST image)
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH, 
                            sizeof(double));

            // SSR READ setup of weight gradients
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH, 
                            sizeof(double));

            // SSR WRITE setup of weight gradients
            snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                            IN_CH, 
                            sizeof(double));

        }

        // SSR start address need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image); // ft0
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weight_grads[ldW * out]); // ft1
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weight_grads[out*ldW]); // ft2

        // benchmark_get_cycle();

        // printf("activations[%u] = %.15f\n", idx_eff, activations[ldB * out]);

        b_grad_update = (idx_eff == target_n) ? activations[ldB * out] - 1 : activations[ldB * out];

        // Start of SSR region
        snrt_ssr_enable();  
        /// NON-UNROLLED VERSION
        asm volatile(
                    "frep.o      %[n_frep], 1, 0, 0 \n"
                    "fmadd.d     ft2, %[b_grad_update], ft0, ft1\n"
                    : 
                    : [ b_grad_update ] "f"(b_grad_update), [ n_frep ] "r"(IN_CH - 1)
                    : "ft0", "ft1", "ft2"
        );
        /// NON-UNROLLED VERSION

        /// UNROLLED VERSION
        // asm volatile(
        //             "frep.o      %[n_frep], 4, 0, 0 \n"
        //             "fmadd.d     ft2, %[b_grad_update], ft0, ft1\n"
        //             "fmadd.d     ft2, %[b_grad_update], ft0, ft1\n"
        //             "fmadd.d     ft2, %[b_grad_update], ft0, ft1\n"
        //             "fmadd.d     ft2, %[b_grad_update], ft0, ft1\n"
        //             :
        //             : [ b_grad_update ] "f"(b_grad_update), [ n_frep ] "r"(IN_CH / 4 - 1)
        //             : "ft0", "ft1", "ft2"
        // );
        /// UNROLLED VERSION
        
        // for(uint32_t in = 0; in < IN_CH;){

        // //     // NON-UNROLLED VERSION
        // //     snrt_ssr_disable();
        // //     W_checksum += weight_grads[out * ldW + in + 0];
        // //     snrt_ssr_enable();
        // //     // NON-UNROLLED VERSION

        //     /// UNROLLED VERSION
        //     snrt_ssr_disable();
        //         W_checksum += weight_grads[out * ldW + in + 0] 
        //                 + weight_grads[out * ldW + in + 1] 
        //                 + weight_grads[out * ldW + in + 2] 
        //                 + weight_grads[out * ldW + in + 3];
        //     snrt_ssr_enable();
        //     /// UNROLLED VERSION

        // //     in += 1;
        //     in += unroll;
        // }

        // End of the SSR region. 
        snrt_ssr_disable();
        // INFO: after disabling the SSRs we can free the registers
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));
            
        bias_grads[ldB * out] = b_grad_update; 
        // benchmark_get_cycle();
        // printf("Benchmark GRADIENT UPDATE FP64 with SSRs: bias_grads[%u] = %f\n", idx_eff, b_grad_update);
        // printf("Benchmark GRADIENT UPDATE FP64 with SSRs: W_checksum[%u] = %f\n", idx_eff, W_checksum);
    }

    snrt_cluster_hw_barrier();
}

//// Training Step
static inline void single_cluster_training_step_fp64_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB, uint32_t setup_SSR){
    
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    // FIXME: learning rate should be defined in network struct
    double lr = 0.5;

    // double W_checksum = 0.0;

    // volatile uint32_t idx_eff;
    // volatile uint32_t W_idx_eff;
    // const uint32_t unroll = 4;

    for(uint32_t out = 0; out < OUT_CH; out++){

        // idx_eff = compute_id + ldB * out;
        
        if (setup_SSR) {
    
            // SSR setup of weight gradients
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                    IN_CH, 
                    sizeof(double));

            // SSR READ setup of weights
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                    IN_CH, 
                    sizeof(double));

            // SSR WRITE setup of weights
            snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                    IN_CH, 
                    sizeof(double));
        }
                // SSR start address need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &weight_grads[out*ldW]); // ft0 
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]); // ft1
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weights[out*ldW]); // ft2

        // benchmark_get_cycle();
        
        biases[ldB * out] -= lr * bias_grads[ldB * out];

        // printf("Benchmark TRAINING STEP FP64 with SSRs: biases[%u] = %f\n", idx_eff, biases[ldB * out]);


        // W_checksum = 0.0;

        // Start of SSR region
        snrt_ssr_enable();
        /// NON-UNROLLED VERSION
        asm volatile(
            "frep.o      %[n_frep], 1, 0, 0 \n"
            "fmadd.d     ft2, %[lr], ft0, ft1\n"
            :
            :[ lr ] "f"(-lr), [ n_frep ] "r"(IN_CH - 1)
            :"ft0", "ft1", "ft2"
        );
        /// NON-UNROLLED VERSION

        /// UNROLLED VERSION
        // asm volatile(
        //             "frep.o      %[n_frep], 4, 0, 0 \n"
        //             "fmadd.d     ft2, %[lr], ft0, ft1\n"
        //             "fmadd.d     ft2, %[lr], ft0, ft1\n"
        //             "fmadd.d     ft2, %[lr], ft0, ft1\n"
        //             "fmadd.d     ft2, %[lr], ft0, ft1\n"
        //             :
        //             :[ lr ] "f"(-lr), [ n_frep ] "r"(IN_CH / 4 - 1)
        //             :"ft0", "ft1", "ft2"
        // );
        /// UNROLLED VERSION

        // for(uint32_t in = 0; in < IN_CH;){

        //     snrt_ssr_disable();
        // //     W_checksum += weights[out * ldW + in];
        //     W_checksum += weights[out * ldW + in]
        //                 + weights[out * ldW + in + 1]
        //                 + weights[out * ldW + in + 2]
        //                 + weights[out * ldW + in + 3];
        //     snrt_ssr_enable();
        
        // //     in += 1;
        //     in += unroll;

        // }
        // End of the SSR region. 
        snrt_ssr_disable();
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));
        // benchmark_get_cycle();
        // printf("Benchmark TRAINING STEP FP64 with SSRs: weight_checksum[%u] = %f\n", W_checksum);
    }
}


