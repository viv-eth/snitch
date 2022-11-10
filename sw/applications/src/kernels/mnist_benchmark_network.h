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
static inline void benchmark_feedforward_fp64(uint32_t IN_CH, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image){
    
    // Linear layer: OUT = X * W^T + B
    // benchmark_get_cycle();
    for (uint32_t out = 0; out < OUT_CH; out++) {
        register double acc = biases[ldB * out];
        for(uint32_t in = 0; in < IN_CH; in++){
            acc += image[in] * weights[out * ldW + in];
        }
        // OUT is accumulated in activations 
        activations[ldB * out] = acc;
        // printf("Benchmarking: FEEDFORWARD FP64 Baseline: acc = %f\n", activations[ldB * out]);  
    }
    // benchmark_get_cycle();

    snrt_cluster_hw_barrier();

}

static inline void benchmark_gradient_update_fp64(uint32_t IN_CH, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, 
                uint32_t compute_id, double *loss){

    
    double b_grad_update = 0.0;
    // double W_grad_update = 0.0;
    volatile uint32_t idx_eff;
    // volatile uint32_t W_idx_eff;
    
    // Commented out for RTL
    // double W_checksum = 0.0;

    // double loss_val = 0.0;


    // get the value saved at target address
    int32_t target_n = *target;

    // NOTE: Part below is commented for the RTL,
    // since math library is not supported, and hence it should 
    // not be included in the benchmarking.

    // compute the loss
    // if(!compute_id){
    //     // printf("target = %u\n", target_n);
    //     // printf("activation[%u] = %f\n", target_n, activations[target_n]);
    //     loss_val = 0.0 - log(activations[target_n - compute_id]);
    //     // printf("loss activation[target] = %f\n", activations[target_n - compute_id]);
    //     printf("GU current loss = %f\n", loss_val);
    //     // printf("GU activation[target = %u] = %.15f\n", target_n - compute_id, activations[target_n - compute_id]);
    //     // loss_wo_log = 0.0 - my_log(activations[target_n - compute_id], 50);
    //     // printf("loss with math.h = %f\n", loss_val);
    //     // printf("loss with my_log = %f\n", loss_wo_log);
    //     loss[0] += loss_val;
    // } 
    

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
        // printf("GU FP64 Baseline W_checksum[%u] = %f\n", idx_eff, W_checksum);
        // printf("GU FP64 Baseline bias_grads[%u] = %f\n", idx_eff, b_grad_update);
    }

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

}

static inline void benchmark_training_step_fp64(uint32_t IN_CH, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB){

    float lr = 0.5;
    // double W_checksum = 0.0;

    for(uint32_t out = 0; out < OUT_CH; out++){
        biases[ldB * out] -= lr * bias_grads[ldB * out];
        // W_checksum = 0.0;

        // printf("TS FP64 Baseline updated bias = %f\n", biases[ldB * out]);

        for(uint32_t in = 0; in < IN_CH; in++){
            weights[out * ldW + in] -= lr * weight_grads[out * ldW + in];
            // W_checksum += weights[out * ldW + in];
        }

        // printf("TS FP64 Baseline updated weight_checksum = %f\n", W_checksum);
    }

}

// INFO: start of FP64 network implementation using SSRs
//// Feedforward Step
static inline void benchmark_feedforward_fp64_ssrn(uint32_t IN_CH, uint32_t OUT_CH, 
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


static inline void benchmark_softmax_activation_fp64(uint32_t OUT_CH, 
                double *activations, uint32_t ldB,
                uint32_t compute_id, uint32_t compute_num, double *max){

    double max_core;
    double sum = 0.0;

    max_core = activations[0];

    for(uint32_t out = 0; out < OUT_CH; out++){
        if(activations[ldB * out] > max_core) {
            max_core = activations[ldB * out];
        }
    }

    max[compute_id] = max_core;
    
    snrt_cluster_hw_barrier();

    double max_global = max[0];

    // Reduction on single core
    if(compute_id == 0){
        for(uint32_t core = 0; core < compute_num; core++){
            if(max[core] > max_global){
                max_global = max[core];
            }
        }

        // printf("Benchmark SOFTMAX FP64: max_global = %f\n", max_global);

        // FIXME: actually OUT_CH should be multiplied by number of compute cores
        for(uint32_t out = 0; out < OUT_CH * compute_num; out++){
            if(activations[out]){
                activations[out] = my_exp(activations[out] - max_global); 
                // activations[out] = exp(activations[out] - max_global);
                sum += activations[out];
            } else {
                activations[out] = 0.0;
            }
        }


        for(uint32_t out = 0; out < OUT_CH * compute_num; out++){
            activations[out] /= sum;
            // printf("Benchmark SOFTMAX FP64: activation[%u] = %.10f\n", out, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
}

//// Gradient Update
static inline void benchmark_gradient_update_fp64_ssr(uint32_t IN_CH, uint32_t OUT_CH, 
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
static inline void benchmark_training_step_fp64_ssr(uint32_t IN_CH, uint32_t OUT_CH, 
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


//// Feedforward Step
static inline void benchmark_feedforward_fp32_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *biases, float *activations,
                uint32_t ldB, float *image, uint32_t setup_SSR){

    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));

    const register float zero = 0.0;

    for (uint32_t out = 0; out < OUT_CH; out++) {
        
        register float acc = 0.0;

        /// UNROLLED VERSION
        // const uint32_t unroll = 4;
        // register v2f32 reduce_reg[unroll];
        /// UNROLLED VERSION

        /// NON-UNROLLED VERSION
        register v2f32 reduce_reg;
        /// NON-UNROLLED VERSION

        register float sum = 0;

        
        if (setup_SSR) {

            // setup of DATA MOVER input data (MNIST image)
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH / 2, 
                            sizeof(double));
            
            // setup of DATA MOVER for weights
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH / 2,  
                            sizeof(double));
        }
        // we need to read the image for every new iteration
        // of a core, because otherwise it will evaluate to
        // all zeros due to the stream semantics
        // Start of SSR region
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]);

        snrt_ssr_enable();
        // benchmark_get_cycle();
        acc = biases[ldB * out];

        /// UNROLLED VERSION
        // asm volatile(
        //     "vfcpka.s.s        %[reduce_reg_0], %[acc], %[zero] \n"
        //     "vfcpka.s.s        %[reduce_reg_1], %[zero], %[zero] \n"
        //     "vfcpka.s.s        %[reduce_reg_2], %[zero], %[zero] \n"
        //     "vfcpka.s.s        %[reduce_reg_3], %[zero], %[zero] \n"
        //     "frep.o            %[n_frep], 4, 0, 0 \n"
        //     "vfmac.s           %[reduce_reg_0], ft0, ft1 \n"
        //     "vfmac.s           %[reduce_reg_1], ft0, ft1 \n"
        //     "vfmac.s           %[reduce_reg_2], ft0, ft1 \n"
        //     "vfmac.s           %[reduce_reg_3], ft0, ft1 \n"             
        //     "vfsum.s           %[sum], %[reduce_reg_0] \n"
        //     "vfsum.s           %[sum], %[reduce_reg_1] \n"
        //     "vfsum.s           %[sum], %[reduce_reg_2] \n"
        //     "vfsum.s           %[sum], %[reduce_reg_3] \n"
        //     // "vfcpka.s.s        %[acc], %[sum], %[zero] \n"
        //     : [ acc ] "+&f"(acc), [ sum ] "+&f"(sum), 
        //       [ reduce_reg_0 ] "+&f"(reduce_reg[0]), [ reduce_reg_1 ] "+&f"(reduce_reg[1]), 
        //       [ reduce_reg_2 ] "+&f"(reduce_reg[2]), [ reduce_reg_3 ] "+&f"(reduce_reg[3])
        //     : [ zero ] "f"(zero), [ n_frep ] "r"(IN_CH / (2 * unroll) - 1)
        //     : "ft0", "ft1", "ft2"
        // );
        /// UNROLLED VERSION

        /// NON-UNROLLED VERSION
        asm volatile(
            "vfcpka.s.s        %[reduce_reg], %[acc], %[zero] \n"
            "frep.o            %[n_frep], 1, 0, 0 \n"
            "vfmac.s           %[reduce_reg], ft0, ft1 \n"             
            "vfsum.s           %[sum], %[reduce_reg] \n"
            // "vfcpka.s.s        %[acc], %[sum], %[zero] \n"
            : [ acc ] "+&f"(acc), [ sum ] "+&f"(sum), [ reduce_reg ] "+f"(reduce_reg)
            : [ zero ] "f"(zero), [ n_frep ] "r"(IN_CH / 2 - 1)
            : "ft0", "ft1", "ft2"
        );
        /// NON-UNROLLED VERSION


        // End of SSR region. 
        snrt_ssr_disable();
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));
    

        activations[ldB * out] = sum;
        // benchmark_get_cycle();
        sum = 0.0;


    }

    // for (uint32_t out = 0; out < OUT_CH; out++) {
    //     printf("Benchmark FEEDFORWARD FP32 SIMD with SSRs: acc[%u] = %f\n", compute_id + out * ldB, activations[ldB * out]);
    // }

    snrt_cluster_hw_barrier();

} 

static inline void benchmark_softmax_activation_fp32(uint32_t OUT_CH, 
                float *activations, uint32_t ldB,
                uint32_t compute_id, uint32_t compute_num, float *max){

    float max_core;
    float sum = 0.0;

    max_core = activations[0];

    for(uint32_t out = 0; out < OUT_CH; out++){
        if(activations[ldB * out] > max_core) {
            max_core = activations[ldB * out];
        }
    }

    max[compute_id] = max_core;
    
    snrt_cluster_hw_barrier();

    float max_global = max[0];

    // Reduction on single core
    if(compute_id == 0){
        for(uint32_t core = 0; core < compute_num; core++){
            if(max[core] > max_global){
                max_global = max[core];
            }
        }

        // printf("Benchmark SOFTMAX FP64: max_global = %f\n", max_global);

        // FIXME: actually OUT_CH should be multiplied by number of compute cores
        for(uint32_t out = 0; out < OUT_CH * compute_num; out++){
            if(activations[out]){
                activations[out] = my_exp(activations[out] - max_global); 
                // activations[out] = exp(activations[out] - max_global);
                sum += activations[out];
            } else {
                activations[out] = 0.0;
            }
        }


        for(uint32_t out = 0; out < OUT_CH * compute_num; out++){
            activations[out] /= sum;
            // printf("Benchmark SOFTMAX FP32: activation[%u] = %.10f\n", out, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
}

//// Gradient Update
static inline void benchmark_gradient_update_fp32_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                float *weight_grads, uint32_t ldW, float *bias_grads, float *activations, 
                uint32_t ldB, float *image, uint32_t *target, 
                uint32_t compute_id, float *loss, uint32_t setup_SSR){

    // INFO: due to a compiler bug we need to reserve the registers for the SSR
    //       otherwise it will use them for stack operations breaking the stream(s)
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    float b_grad_update = 0.0;
    // register float W_checksum = 0.0;

    /// NON-UNROLLED VERSION
    register v2f32 reduce_reg;
    /// NON-UNROLLED VERSION

    /// UNROLLED VERSION
    // const uint32_t unroll = 4;
    // register v2s reduce_reg[unroll];
    /// UNROLLED VERSION

    register float sum = 0.0;
    register v2f32 W_grad_update;
    // float loss_val = 0.0;
    volatile uint32_t idx_eff;

    // get the value saved at target address
    uint32_t target_n = *target;
    
    // compute the loss
    // if(!compute_id){
    //     loss_val = 0.0 - log(activations[target_n - compute_id]);
    //     printf("GU current loss = %f\n", loss_val);
    //     loss[0] += loss_val;
    // }

    for(uint32_t out = 0; out < OUT_CH; out++){

        // W_checksum = 0.0;
        idx_eff = compute_id + ldB * out;

        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        bias_grads[ldB * out] = b_grad_update;
        
        // SSR strides and bounds only have to be configured
        // once in the beginning
        if (setup_SSR) {

            // SSR read setup of input data (MNIST image)
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH / 2, // number of iterations
                            sizeof(double));
            
            // SSR read setup of weight gradients 
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH / 2, 
                            sizeof(double));

            // SSR write setup of weight gradients
            snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                            IN_CH / 2, 
                            sizeof(double));


            // SSR start address need to be configured each time
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image); // ft0
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weight_grads[out*ldW]); // ft1
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weight_grads[out*ldW]); // ft2

        
        }

        // printf("Benchmark GRADIENT UPDATE FP32 with SSRs: bias_grads[%u] = %f\n", idx_eff, b_grad_update);
        
        snrt_ssr_enable();

        /// NON-UNROLLED VERSION
        asm volatile(
            "vfcpka.s.s         %[reduce_reg], %[b_grad], %[b_grad] \n"       // load the bias gradient for each vector
            "frep.o             %[n_frep], 2, 0, 0 \n"
            // TODO: implement vfmadd
            // "vfmadd.s           ft2, ft0, %[reduce_reg], ft1 \n"  // multiply the input image with the bias gradient and add it to the weight gradient
            "vfmul.s            %[W_grad_update], ft0, %[reduce_reg] \n"  // multiply the input image with the bias gradient
            "vfadd.s            ft2, %[W_grad_update], ft1 \n"  // add it to the weight gradient
            : [ sum ] "+&f"(sum), 
              [reduce_reg] "+&f"(reduce_reg), 
              [W_grad_update] "+&f"(W_grad_update)
            : [b_grad] "f"(b_grad_update), 
              [n_frep] "r"(IN_CH / 2 - 1)
            : "ft0", "ft1", "ft2"
        );
        /// END OF NON-UNROLLED VERSION

        /// UNROLLED VERSION
        // asm volatile(
        //             // "vfcpka.s.s         %[sum], %[zero_reg], %[zero_reg] \n" 
        //             "vfcpka.s.s         %[reduce_reg_0], %[b_grad], %[b_grad] \n"       // load the bias gradient for each vector
        //             "vfcpka.s.s         %[reduce_reg_1], %[b_grad], %[b_grad] \n" 
        //             "vfcpka.s.s         %[reduce_reg_2], %[b_grad], %[b_grad] \n" 
        //             "vfcpka.s.s         %[reduce_reg_3], %[b_grad], %[b_grad] \n" 
        //             "frep.o             %[n_frep], 8, 0, 0 \n"
        //             "vfmul.s            %[reduce_reg_0], %[reduce_reg_0], ft0 \n"         // compute weight update b_grad * image
        //             "vfmul.s            %[reduce_reg_1], %[reduce_reg_1], ft0 \n"
        //             "vfmul.s            %[reduce_reg_2], %[reduce_reg_2], ft0 \n"
        //             "vfmul.s            %[reduce_reg_3], %[reduce_reg_3], ft0 \n"
        //             "vfadd.s            ft2, %[reduce_reg_0], ft1 \n"         // add weight update to weight gradient
        //             "vfadd.s            ft2, %[reduce_reg_1], ft1 \n"
        //             "vfadd.s            ft2, %[reduce_reg_2], ft1 \n"
        //             "vfadd.s            ft2, %[reduce_reg_3], ft1 \n"
        //             // "vfadd.s            ft2, %[reduce_reg_0], %[zero_reg] \n"           // write the values into the weight gradients
        //             // "vfsum.s            %[sum], %[reduce_reg_0]\n"             // compute the checksum of the weight gradients --> remove it for benchmarking
        //             // "vfadd.s            ft2, %[reduce_reg_1], %[zero_reg] \n"
        //             // "vfsum.s            %[sum], %[reduce_reg_1]\n"
        //             // "vfadd.s            ft2, %[reduce_reg_2], %[zero_reg] \n"
        //             // "vfsum.s            %[sum], %[reduce_reg_2]\n"
        //             // "vfadd.s            ft2, %[reduce_reg_3], %[zero_reg] \n"
        //             // "vfsum.s            %[sum], %[reduce_reg_3]\n"
        //             : //[zero_reg] "+&f"(zero_reg), [ sum ] "+&f"(sum), 
        //               [reduce_reg_0] "+&f"(reduce_reg[0].f64), [reduce_reg_1] "+&f"(reduce_reg[1].f64),
        //               [reduce_reg_2] "+&f"(reduce_reg[2].f64), [reduce_reg_3] "+&f"(reduce_reg[3].f64)
        //             : [b_grad] "f"(b_grad_update), [n_frep] "r"(IN_CH / (2 * unroll) - 1)
        //             : "ft0", "ft1", "ft2"
        // );
        /// END OF UNROLLED VERSION

        snrt_ssr_disable();
        // INFO: after disabling the SSRs we can free the registers
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));
        // for(uint32_t in = 0; in < IN_CH;){

        //     W_checksum += weight_grads[out*ldW + in + 0] + weight_grads[out*ldW + in + 1];
        //     // W_checksum += weight_grads[out*ldW + in + 0] + weight_grads[out*ldW + in + 1]
        //     //             + weight_grads[out*ldW + in + 2] + weight_grads[out*ldW + in + 3]
        //     //             + weight_grads[out*ldW + in + 4] + weight_grads[out*ldW + in + 5]
        //     //             + weight_grads[out*ldW + in + 6] + weight_grads[out*ldW + in + 7];
                
        //     in += 2;
        //     // in += 2*4;
        // }
        // printf("Benchmark GRADIENT UPDATE FP32 with SSRs: W_checksum[%u] = %f\n", idx_eff, W_checksum);

    }

    snrt_cluster_hw_barrier();
}

//// Training Step
static inline void benchmark_training_step_fp32_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                float *weights, float *weight_grads, uint32_t ldW, float *biases, float *bias_grads,
                uint32_t ldB, uint32_t setup_SSR){

    // INFO: due to a compiler bug we need to reserve the registers for the SSR
    //       otherwise it will use them for stack operations breaking the stream(s)
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    float lr = 0.5;

    // float W_checksum = 0.0;

    // convert number of images to float for vectorized computation
    register v2f32 lr_vec;
    register v2f32 W_update;

    /// UNROLLED VERSION
    // const uint32_t unroll = 4;
    // register v2s W_update_reg[unroll];
    /// UNROLLED VERSION

    // pack the learning rate and number of images into a vector for vectorized computation
    asm volatile(
        "vfcpka.s.s          %[lr_vec], %[lr], %[lr] \n"
        : [lr_vec] "+&f"(lr_vec)
        : [lr] "f"(-lr)
        : "ft0", "ft1", "ft2"
    );

    uint32_t volatile idx_eff;


    for(uint32_t out = 0; out < OUT_CH; out++){

        // W_checksum = 0.0;

        // idx_eff = compute_id + out * ldB;

        // SSR strides and bounds only have to be configured
        // once in the beginning
        if (setup_SSR) {
            
            // SSR read setup of weight gradients
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH / 2, // number of iterations
                            sizeof(double));

            // SSR read setup of weights
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH / 2, // number of iterations
                            sizeof(double));

            // SSR write setup of weights
            snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                            IN_CH / 2, // number of iterations
                            sizeof(double));
        }
    
        // SSR start addresses need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &weight_grads[out*ldW]); // weight gradients stored in ft0
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]); // weights stored in ft1 for read
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weights[out*ldW]); // weights stored in ft2 for write    

        biases[ldB * out] -= lr * bias_grads[ldB * out];
       

        // printf("Benchmark TRAINING STEP FP32 with SSRs: biases[%u] = %f\n", idx_eff, biases[ldB * out]);

        snrt_ssr_enable();

        /// NON-UNROLLED VERSION
        asm volatile(
                    "frep.o            %[n_frep], 2, 0, 0 \n"
                    "vfmul.s           ft5, ft0, %[lr_vec] \n"  // multiply the weight gradient with the learning rate
                    "vfadd.s           ft2, ft1, ft5 \n"  // add it to the weight
                    // "vfmadd.s          ft2, ft0, %[lr_vec], ft1 \n"  // multiply the input image with the bias gradient and add it to the weight gradient
                : 
                : [lr_vec] "f"(lr_vec), 
                  [n_frep] "r"(IN_CH / 2 - 1)
                : "ft0", "ft1", "ft2"
        );
        /// END NON-UNROLLED VERSION 

        /// UNROLLED VERSION
        // asm volatile(
        //             "frep.o              %[n_frep], 8, 0, 0 \n"
        //             "vfmul.s             %[W_update_reg_0], %[lr_vec], ft0 \n"
        //             "vfmul.s             %[W_update_reg_1], %[lr_vec], ft0 \n"
        //             "vfmul.s             %[W_update_reg_2], %[lr_vec], ft0 \n"
        //             "vfmul.s             %[W_update_reg_3], %[lr_vec], ft0 \n"
        //             "vfadd.s             ft2, ft1, %[W_update_reg_0] \n"
        //             "vfadd.s             ft2, ft1, %[W_update_reg_1] \n"
        //             "vfadd.s             ft2, ft1, %[W_update_reg_2] \n"
        //             "vfadd.s             ft2, ft1, %[W_update_reg_3] \n"
        //             // "vfmul.s              %[reduce_reg_1], %[lr_vec], ft0 \n"
        //             // "vfmul.s              %[reduce_reg_2], %[lr_vec], ft0 \n"
        //             // "vfmul.s              %[reduce_reg_3], %[lr_vec], ft0 \n"
        //         : [W_update_reg_0] "+&f"(W_update_reg[0].f64), [W_update_reg_1] "+&f"(W_update_reg[1].f64),
        //           [W_update_reg_2] "+&f"(W_update_reg[2].f64), [W_update_reg_3] "+&f"(W_update_reg[3].f64)
        //         : [lr_vec] "f"(lr_vec), 
        //           [n_frep] "r"(IN_CH / (2 * unroll) - 1)
        //         : "ft0", "ft1", "ft2"
        // );
        /// END UNROLLED VERSION

        snrt_ssr_disable();
        // INFO: after disabling the SSRs we can free the registers
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

        // for(uint32_t in = 0; in < IN_CH;){

        //     W_checksum += weights[out*ldW + in + 0] + weights[out*ldW + in + 1];
        //         // W_checksum += reduce_reg[0] + reduce_reg[1];
        //         // W_checksum += weights[out*ldW + in + 0] + weights[out*ldW + in + 1];
        //         // W_checksum += weights[out*ldW + in + 0] + weights[out*ldW + in + 1]
        //         //             + weights[out*ldW + in + 2] + weights[out*ldW + in + 3]
        //         //             + weights[out*ldW + in + 4] + weights[out*ldW + in + 5]
        //         //             + weights[out*ldW + in + 6] + weights[out*ldW + in + 7];
        //         // // GIM: why does vfdiv fail in the RTL?
        //         // // weights[out*ldW + 0] += reduce_reg[0];
        //         // // weights[out*ldW + 1] += reduce_reg[1];
        //         // // W_checksum += reduce_reg_0[0] + reduce_reg_1[0] + reduce_reg_2[0] + reduce_reg_3[0]
        //         // //             + reduce_reg_0[1] + reduce_reg_1[1] + reduce_reg_2[1] + reduce_reg_3[1];
        //         // weights[out*ldW + 0] += reduce_reg_0[0];
        //         // weights[out*ldW + 1] += reduce_reg_0[1];
        //         // weights[out*ldW + 2] += reduce_reg_1[0];
        //         // weights[out*ldW + 3] += reduce_reg_1[1];
        //         // weights[out*ldW + 4] += reduce_reg_2[0];
        //         // weights[out*ldW + 5] += reduce_reg_2[1];
        //         // weights[out*ldW + 6] += reduce_reg_3[0];
        //         // weights[out*ldW + 7] += reduce_reg_3[1];
        //         // snrt_ssr_enable();

        //     in += 2;
        //     // in += 2 * unroll;

        // }

        // printf("Benchmark TRAINING STEP FP32 with SSRs and SIMD: W_checksum[%u] = %f\n", idx_eff, W_checksum);

    }

}

static inline void benchmark_feedforward_fp16_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *biases, __fp16 *activations,
                uint32_t ldB, __fp16 *image, uint32_t setup_SSR){
    
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    
    // INFO: for simplicity image is converted to dtype __fp16 
    const register float zero = 0.0f;

    /// NON-UNROLLED VERSION
    // register v2f32 reduce_reg;
    /// END NON-UNROLLED VERSION

    /// UNROLLED VERSION
    const uint32_t unroll = 4;
    register v2s reduce_reg[unroll];
    /// UNROLLED VERSION

    __fp16 acc = 0.0f;

    for (uint32_t out = 0; out < OUT_CH; out++) {

        register float sum = 0.0f;
        // register float tacc = 0.0f;

        // SSR strides and bounds only have to be configured
        // once in the beginning
        // WARN In the RTL SSR strides MUST BE of size DOUBLE

        if (setup_SSR) {

            // setup of DATA MOVER input data (MNIST image)
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH / 4, 
                            sizeof(double));
            
            // setup of DATA MOVER for weights
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH / 4, 
                            sizeof(double));
        }

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]);
        
        // Start of SSR region
        snrt_ssr_enable();

        acc = biases[ldB * out];
            
        /// NON-UNROLLED VERSION
        // calculate the dot product of the image and the weights (increment by four columns in each iteration)
        // asm volatile(
        //     "vfcpka.s.s       %[reduce_reg], %[zero], %[zero]\n"
        //     "frep.o           %[n_frep], 1, 0, 0\n"
        //     "vfdotpex.s.h     %[reduce_reg], ft1, ft0 \n"
        //     "vfsum.s          %[sum], %[reduce_reg]\n"
        //     // "fadd.s           %[tacc], %[zero], %[sum] \n"
        // : [sum] "+f"(sum), //[tacc] "+f"(tacc), 
        //   [reduce_reg] "+&f"(reduce_reg)
        // : [zero] "f"(zero), [n_frep] "r"(IN_CH / 4 - 1)
        // : "ft0", "ft1", "ft2"
        // );
        /// END NON-UNROLLED VERSION

        /// UNROLLED VERSION
        asm volatile(
            "vfcpka.s.s       %[reduce_reg_0], %[zero], %[zero]\n"
            "vfcpka.s.s       %[reduce_reg_1], %[zero], %[zero]\n"
            "vfcpka.s.s       %[reduce_reg_2], %[zero], %[zero]\n"
            "vfcpka.s.s       %[reduce_reg_3], %[zero], %[zero]\n"
            "frep.o           %[n_frep], 4, 0, 0\n"
            "vfdotpex.s.h     %[reduce_reg_0], ft1, ft0 \n"
            "vfdotpex.s.h     %[reduce_reg_1], ft1, ft0 \n"
            "vfdotpex.s.h     %[reduce_reg_2], ft1, ft0 \n"
            "vfdotpex.s.h     %[reduce_reg_3], ft1, ft0 \n"
            "vfsum.s          %[sum], %[reduce_reg_0]\n"
            "vfsum.s          %[sum], %[reduce_reg_1]\n"
            "vfsum.s          %[sum], %[reduce_reg_2]\n"
            "vfsum.s          %[sum], %[reduce_reg_3]\n"
            // "fadd.s           %[tacc], %[zero], %[sum] \n"
        : [sum] "+f"(sum), //[tacc] "+f"(tacc), 
          [reduce_reg_0] "+&f"(reduce_reg[0].f64), [reduce_reg_1] "+&f"(reduce_reg[1].f64),
          [reduce_reg_2] "+&f"(reduce_reg[2].f64), [reduce_reg_3] "+&f"(reduce_reg[3].f64)
        : [zero] "f"(zero), [n_frep] "r"(IN_CH / (4 * unroll) - 1)
        : "ft0", "ft1", "ft2"
        );
        /// END UNROLLED VERSION

        // End of SSR region.
        snrt_ssr_disable();
        // snrt_fpu_fence();
        acc += sum;
        activations[ldB * out] = acc;
        acc = 0.0;
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

    }


    // for (uint32_t out = 0; out < OUT_CH; out++) {
    //     printf("Benchmark FP16 SIMD with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    // }

    snrt_cluster_hw_barrier();

}

static inline void benchmark_softmax_activation_fp16(uint32_t OUT_CH, 
                __fp16 *activations, uint32_t ldB, uint32_t compute_id, 
                uint32_t compute_num, __fp16 *max){

    __fp16 max_core;
    __fp16 sum = 0.0f;
    __fp16 *act_ptr;

    max_core = activations[0];

    volatile uint32_t idx_eff; 

    for(uint32_t out = 0; out < OUT_CH; out++){
        idx_eff = compute_id + ldB * out;
        if(!(idx_eff > OUT_CH * 5 - 1)){
            if(activations[out * ldB] > max_core){
                max_core = activations[out * ldB];
            }
        }
    }

    max[compute_id] = max_core;
    
    snrt_cluster_hw_barrier();

//     //printf("Max value of compute core %u is %f\n", compute_id, max_core);

    __fp16 max_global = max[0];

    // Reduction on single core
    if(compute_id == 0){
        for(uint32_t core = 0; core < compute_num; core++){
            if(max[core] > max_global){
                max_global = max[core];
            }
        }

        // FIXME: actually OUT_CH should be multiplied by number of compute cores
        for(uint32_t out = 0; out < OUT_CH*compute_num; out++){
            act_ptr = &activations[0];
            if(act_ptr[out] != 0.0f){
                act_ptr[out] = my_exp(act_ptr[out] - max_global);
                sum += act_ptr[out];
            } else {
                act_ptr[out] = 0.0;
            }
        }


        for(uint32_t out = 0; out < OUT_CH*compute_num; out++){
            act_ptr[out] /= sum;
            activations[out] = act_ptr[out];
            // printf("Benchmark SOFTMAX FP16 (no SIMD): activation[%u] = %f\n", out + 1, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
}

static inline void benchmark_gradient_update_fp16_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                __fp16 *weight_grads, uint32_t ldW, __fp16 *bias_grads, __fp16 *activations, 
                uint32_t ldB, __fp16 *image, uint32_t *target, 
                uint32_t compute_id, __fp16 *loss, uint32_t setup_SSR) {

    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

    register float b_grad_update = 0.0;
    
    /// NON-UNROLLED VERSION
    // register v4f16 reduce_reg;
    // register v4f16 W_grad_update_reg;
    /// END NON-UNROLLED VERSION

    /// UNROLLED VERSION
    const uint32_t unroll = 4;
    register v4s reduce_reg[unroll];
    /// END UNROLLED VERSION
    

    // __fp16 W_checksum = 0.0;
    __fp16 W_grad_update = 0.0;
    uint16_t idx_eff;
    

    // get the value saved at target address
    uint16_t target_n = target[0];

    for(uint32_t out = 0; out < OUT_CH; out++){

        // W_checksum = 0.0;
        
        idx_eff = compute_id + ldB * out;

        b_grad_update = (idx_eff == target_n) ? activations[ldB * out] - 1 : activations[ldB * out];
        bias_grads[ldB * out] = b_grad_update;

        // printf("Benchmark GRADIENT UPDATE FP16 SIMD with SSRs: bias_grads[%u] = %f\n", compute_id + out * ldB, bias_grads[ldB * out]);

        if (setup_SSR) {

                // READ setup of DATA MOVER input data (MNIST image)
                snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                                IN_CH / 4, 
                                sizeof(double));
                
                // READ setup of DATA MOVER for weight gradients
                snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                                IN_CH / 4,  
                                sizeof(double));

                // WRITE setup of DATA MOVER for weight gradients
                snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                                IN_CH / 4,  
                                sizeof(double));
            }

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image); // read image ft0
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weight_grads[out*ldW]); // read weight gradients ft1
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weight_grads[out*ldW]); // write weight gradients ft2
            
            // Start of SSR region
            snrt_ssr_enable();

            /// NON-UNROLLED VERSION
            // asm volatile(
            //     "vfcpka.h.s      %[reduce_reg], %[b_grad_update], %[b_grad_update] \n"
            //     "vfcpkb.h.s      %[reduce_reg], %[b_grad_update], %[b_grad_update] \n"
            //     "frep.o          %[n_frep], 2, 0, 0 \n"
            //     "vfmul.h         %[W_grad_update_reg], ft0, %[reduce_reg] \n"
            //     "vfadd.h         ft2, %[W_grad_update_reg], ft1 \n"
            //     : [reduce_reg] "+&f"(reduce_reg), [W_grad_update_reg] "+&f"(W_grad_update_reg)
            //     : [b_grad_update] "f"(b_grad_update), [n_frep] "r"(IN_CH / 4 - 1)
            //     : "ft0", "ft1", "ft2"
            // );
            /// END NON-UNROLLED VERSION

            /// UNROLLED VERSION
            asm volatile(
                "vfcpka.h.s      %[reduce_reg_0], %[b_grad_update], %[b_grad_update] \n"
                "vfcpkb.h.s      %[reduce_reg_0], %[b_grad_update], %[b_grad_update] \n"
                "vfcpka.h.s      %[reduce_reg_1], %[b_grad_update], %[b_grad_update] \n"
                "vfcpkb.h.s      %[reduce_reg_1], %[b_grad_update], %[b_grad_update] \n"
                "vfcpka.h.s      %[reduce_reg_2], %[b_grad_update], %[b_grad_update] \n"
                "vfcpkb.h.s      %[reduce_reg_2], %[b_grad_update], %[b_grad_update] \n"
                "vfcpka.h.s      %[reduce_reg_3], %[b_grad_update], %[b_grad_update] \n"
                "vfcpkb.h.s      %[reduce_reg_3], %[b_grad_update], %[b_grad_update] \n"
                "frep.o          %[n_frep], 8, 0, 0 \n"
                "vfmul.h         %[reduce_reg_0], ft0, %[reduce_reg_0] \n"
                "vfmul.h         %[reduce_reg_1], ft0, %[reduce_reg_1] \n"
                "vfmul.h         %[reduce_reg_2], ft0, %[reduce_reg_2] \n"
                "vfmul.h         %[reduce_reg_3], ft0, %[reduce_reg_3] \n"
                "vfadd.h         ft2, %[reduce_reg_0], ft1 \n"
                "vfadd.h         ft2, %[reduce_reg_1], ft1 \n"
                "vfadd.h         ft2, %[reduce_reg_2], ft1 \n"
                "vfadd.h         ft2, %[reduce_reg_3], ft1 \n"
                : [reduce_reg_0] "+&f"(reduce_reg[0].f64), [reduce_reg_1] "+&f"(reduce_reg[1].f64), 
                  [reduce_reg_2] "+&f"(reduce_reg[2].f64), [reduce_reg_3] "+&f"(reduce_reg[3].f64)
                : [b_grad_update] "f"(b_grad_update), [n_frep] "r"(IN_CH / (4 * unroll) - 1)
                : "ft0", "ft1", "ft2"
            );
            /// END UNROLLED VERSION

            snrt_ssr_disable();
            // INFO: after disabling the SSRs we can free the registers
            asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

            // for(uint32_t in = 0; in < IN_CH;){
            //     W_checksum += weight_grads[out*ldW + in + 0] + weight_grads[out*ldW + in + 1] 
            //                 + weight_grads[out*ldW + in + 2] + weight_grads[out*ldW + in + 3];

            //     in += 4;
            // }

            // printf("Benchmark GRADIENT UPDATE FP16 SIMD with SSRs: W_checksum[%u] = %f\n", idx_eff, W_checksum);

    }

    snrt_cluster_hw_barrier();
}

static inline void benchmark_training_step_fp16_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                __fp16 *weights, __fp16 *weight_grads, uint32_t ldW, __fp16 *biases, __fp16 *bias_grads, 
                uint32_t ldB, uint32_t setup_SSR) {

    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

    float lr = 0.5;
    // __fp16 W_checksum = 0.0;
    uint16_t idx_eff;

    register v4f16 lr_reg;

    /// UNROLLED VERSION
    const uint32_t unroll = 4;
    register v4s W_update_reg[unroll];
    /// END UNROLLED VERSION

    asm volatile (
        "vfcpka.h.s      %[lr_reg], %[lr], %[lr] \n"
        "vfcpkb.h.s      %[lr_reg], %[lr], %[lr] \n"
        : [lr_reg] "+&f"(lr_reg)
        : [lr] "f"(-lr)
        : "ft0", "ft1", "ft2"
    );
    

    

    for(uint32_t out = 0; out < OUT_CH; out++){

        // W_checksum = 0.0;
        
        // idx_eff = compute_id + ldB * out;

        // SSR strides and bounds only have to be configured
        // once in the beginning
        if (setup_SSR) {
            
            // SSR read setup of weight gradients
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH / 4, // number of iterations
                            sizeof(double));

            // SSR read setup of weights
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH / 4, // number of iterations
                            sizeof(double));

            // SSR write setup of weights
            snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                            IN_CH / 4, // number of iterations
                            sizeof(double));
        }

        // SSR start addresses need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &weight_grads[out*ldW]); // weight gradients stored in ft0
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]); // weights stored in ft1 for read
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weights[out*ldW]); // weights stored in ft2 for write   

        biases[ldB * out] -= lr * bias_grads[ldB * out];

        // printf("Benchmark TRAINING STEP FP16 SIMD with SSRs: biases[%u] = %f\n", idx_eff, biases[ldB * out]);
        // printf("Benchmark TRAINING STEP FP16 SIMD with SSRs: bias_grads[%u] = %f\n", idx_eff, bias_grads[ldB * out]);

        // Start of SSR region
        snrt_ssr_enable();

        /// NON-UNROLLED VERSION
        // asm volatile(
        //     "frep.o          %[n_frep], 2, 0, 0 \n"
        //     "vfmul.h         ft5, ft0, %[lr_vec] \n"
        //     "vfadd.h         ft2, ft1, ft5 \n"
        //     : 
        //     : [n_frep] "r"(IN_CH / 4 - 1), [lr_vec] "f"(lr_reg)
        //     : "ft0", "ft1", "ft2"
        // );
        /// END NON-UNROLLED VERSION
        
        /// UNROLLED VERSION
        asm volatile(
            "frep.o          %[n_frep], 8, 0, 0 \n"
            "vfmul.h         %[W_update_reg_0], ft0, %[lr_vec] \n"
            "vfmul.h         %[W_update_reg_1], ft0, %[lr_vec] \n"
            "vfmul.h         %[W_update_reg_2], ft0, %[lr_vec] \n"
            "vfmul.h         %[W_update_reg_3], ft0, %[lr_vec] \n"
            "vfadd.h         ft2, ft1, %[W_update_reg_0] \n"
            "vfadd.h         ft2, ft1, %[W_update_reg_1] \n"
            "vfadd.h         ft2, ft1, %[W_update_reg_2] \n"
            "vfadd.h         ft2, ft1, %[W_update_reg_3] \n"
            : [W_update_reg_0] "+&f"(W_update_reg[0].f64), [W_update_reg_1] "+&f"(W_update_reg[1].f64), 
              [W_update_reg_2] "+&f"(W_update_reg[2].f64), [W_update_reg_3] "+&f"(W_update_reg[3].f64)
            : [n_frep] "r"(IN_CH / (4 * unroll) - 1), [lr_vec] "f"(lr_reg)
            : "ft0", "ft1", "ft2"
        );
        /// END UNROLLED VERSION

        snrt_ssr_disable();
        // INFO: after disabling the SSRs we can free the registers
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

        // for(uint32_t in = 0; in < IN_CH;){
        //     W_checksum += weights[out*ldW + in + 0] + weights[out*ldW + in + 1] 
        //                 + weights[out*ldW + in + 2] + weights[out*ldW + in + 3];

        //     in += 4;
        // }

        // printf("Benchmark TRAINING STEP FP16 SIMD with SSRs: W_checksum[%u] = %f\n", idx_eff, W_checksum);
        
    }
}

static inline void benchmark_feedforward_fp8_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *biases,
                uint32_t ldB, char *image, uint32_t setup_SSR, float *activations_fp32){

    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));

    const register float zero = 0.0;
    register float acc_float = 0.0;

    /// NON-UNROLLED VERSION
    register v8s c_vec;
    /// END NON-UNROLLED VERSION

    /// UNROLLED VERSION
    // const uint32_t unroll = 7;
    // register v8s c_vec[unroll];
    /// END UNROLLED VERSION

    register v2s sum;

    for (uint32_t out = 0; out < OUT_CH; out++) {
        // get the output activation index
        // idx_eff = compute_id + ldB * out;

        if (setup_SSR) {

            // setup of DATA MOVER input data (MNIST image)
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH / 8, 
                            sizeof(double));

            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH / 8, 
                            sizeof(double));
        }

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]);

        // Start of SSR region
        snrt_ssr_enable();

        /// NON-UNROLLED VERSION
        c_vec = (v8s) {biases[ldB * out], 0, 0, 0, 0, 0, 0, 0}; 

        asm volatile(
            "vfcpka.s.s      %[sum], %[zero], %[zero] \n"
            "vfcpka.s.s      %[acc_float], %[zero], %[zero] \n"
            "frep.o          %[n_frep], 1, 0, 0 \n"
            "vfdotpex.h.b    %[c_vec], ft1, ft0 \n"
            "vfsumex.s.h     %[sum], %[c_vec] \n"
            "vfsum.s         %[acc_float], %[sum] \n"
            : [acc_float] "+&f"(acc_float), [sum] "+&f"(sum.f64), [c_vec] "+&f"(c_vec.f64)
            : [zero] "f"(zero), [n_frep] "r"(IN_CH / 8 - 1)
            : "ft0", "ft1", "ft2"
        );
        /// END NON-UNROLLED VERSION

        /// UNROLLED VERSION
        // c_vec[0] = (v8s) {biases[ldB * out], 0, 0, 0, 0, 0, 0, 0}; 
        // c_vec[1] = (v8s) {0, 0, 0, 0, 0, 0, 0, 0};
        // c_vec[2] = (v8s) {0, 0, 0, 0, 0, 0, 0, 0};
        // c_vec[3] = (v8s) {0, 0, 0, 0, 0, 0, 0, 0};
        // c_vec[4] = (v8s) {0, 0, 0, 0, 0, 0, 0, 0};
        // c_vec[5] = (v8s) {0, 0, 0, 0, 0, 0, 0, 0};
        // c_vec[6] = (v8s) {0, 0, 0, 0, 0, 0, 0, 0};


        // asm volatile(
        //     "vfcpka.s.s      %[sum], %[zero], %[zero] \n"
        //     // "vfcpka.s.s      %[acc_float], %[zero], %[zero] \n"
        //     "frep.o          %[n_frep], 7, 0, 0 \n"
        //     "vfdotpex.h.b    %[c_vec_0], ft1, ft0 \n"
        //     "vfdotpex.h.b    %[c_vec_1], ft1, ft0 \n"
        //     "vfdotpex.h.b    %[c_vec_2], ft1, ft0 \n"
        //     "vfdotpex.h.b    %[c_vec_3], ft1, ft0 \n"
        //     "vfdotpex.h.b    %[c_vec_4], ft1, ft0 \n"
        //     "vfdotpex.h.b    %[c_vec_5], ft1, ft0 \n"
        //     "vfdotpex.h.b    %[c_vec_6], ft1, ft0 \n"
        //     "vfsumex.s.h     %[sum], %[c_vec_0] \n"
        //     "vfsumex.s.h     %[sum], %[c_vec_1] \n"
        //     "vfsumex.s.h     %[sum], %[c_vec_2] \n"
        //     "vfsumex.s.h     %[sum], %[c_vec_3] \n"
        //     "vfsumex.s.h     %[sum], %[c_vec_4] \n"
        //     "vfsumex.s.h     %[sum], %[c_vec_5] \n"
        //     "vfsumex.s.h     %[sum], %[c_vec_6] \n"
        //     "vfsum.s         %[acc_float], %[sum] \n"
        //     : [acc_float] "+&f"(acc_float), [sum] "+&f"(sum.f64), [c_vec_0] "+&f"(c_vec[0].f64), 
        //       [c_vec_1] "+&f"(c_vec[1].f64), [c_vec_2] "+&f"(c_vec[2].f64), 
        //       [c_vec_3] "+&f"(c_vec[3].f64), [c_vec_4] "+&f"(c_vec[4].f64), 
        //       [c_vec_5] "+&f"(c_vec[5].f64), [c_vec_6] "+&f"(c_vec[6].f64)
        //     : [zero] "f"(zero), [n_frep] "r"(IN_CH / (8 * unroll) - 1)
        //     : "ft0", "ft1", "ft2"
        // );


        snrt_ssr_disable();
        // INFO: after disabling the SSRs we can free the registers
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));
        // discuss with GIM
        // if(acc_float >= FLT_MAX) {
        //     acc_float = sum.vec[1] * 2;//0.0f;
        // }

        // printf("Benchmark FEEDFORWARD FP8 SIMD with SSRs: acc_float[%u] = %f\n", ldB * out, acc_float);
        activations_fp32[ldB * out] = acc_float;
        // printf("Benchmark FEEDFORWARD FP8 with SSRs again: activations_fp32[%u] = %f\n", ldB * out, activations_fp32[ldB * out]);

    }

    snrt_cluster_hw_barrier();

}

static inline void benchmark_softmax_activation_fp32_ex(uint32_t OUT_CH,
                float *activations_fp32, uint32_t ldB, uint32_t compute_id, 
                uint32_t compute_num, float *max){

    float max_core = 0.0;
    float sum = 0.0;
    float max_global;

    uint32_t idx_eff;
    max_core = activations_fp32[0];

    for(uint32_t out = 0; out < OUT_CH; out++){
        idx_eff = compute_id + ldB * out;
        // printf("Benchmark SOFTMAX FP32 expanding (no SIMD): activations_fp32[%u] = %f\n", idx_eff, activations_fp32[out]);
        if(activations_fp32[ldB * out] > max_core) {
            max_core = activations_fp32[ldB * out];
        }

    }

    max[compute_id] = max_core; 

    // printf("FEEDFORWARD FP32 expanding: max[%u] = %.10f\n", compute_id, max[compute_id]);
    
    snrt_cluster_hw_barrier();

    max_global = max[0];

    // Reduction on single core
    if(compute_id == 0){
        for(uint32_t core = 0; core < compute_num; core++){
            if(max[core] > max_global){
                max_global = max[core];
            }
        }
        
        
        for(uint32_t out = 0; out < OUT_CH*compute_num; out++){
            if(activations_fp32[out]){
                activations_fp32[out] = my_exp(activations_fp32[out] - max_global);
                sum += activations_fp32[out];
            } else {
                activations_fp32[out] = 0.0;
            }
        }


        for(uint32_t out = 0; out < OUT_CH*compute_num; out++){
            activations_fp32[out] /= sum;
            // printf("SOFTMAX FP32 expanding (no SIMD): activation[%u] = %.10f\n", out, activations_fp32[out]);

        }
    }

    snrt_cluster_hw_barrier();
}

static inline void benchmark_gradient_update_fp8_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                        char *weight_grads, uint32_t ldW, float *bias_grads,
                        float *activations_fp32, uint32_t ldB, char *image, 
                        uint32_t *target, uint32_t compute_id, 
                        char *loss, uint32_t setup_SSR) {

    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

    uint32_t idx_eff;
    // uint32_t overflow_cnt = 0;
    float b_grad_update;
    // float W_checksum_fp32;

    // char W_grad_acc;

    register v8f8 b_grad_update_reg;

    /// NON-UNROLLED VERSION
    register v8f8 W_grad_update_reg;
    /// END NON-UNROLLED VERSION

    /// UNROLLED VERSION
    // const uint32_t unroll = 7;
    // register v8s W_grad_update_reg[unroll];
    /// END UNROLLED VERSION


    for(uint32_t out = 0; out < OUT_CH; out++){

        // W_checksum_fp32 = 0.0;
        // W_grad_acc = 0b00000000;

        if (setup_SSR) {

            // SSR read setup of input data (MNIST image)
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH / 8, 
                            sizeof(double));
            
            // SSR read setup of weight gradients 
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH / 8, 
                            sizeof(double));

            // SSR write setup of weight gradients
            snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                            IN_CH / 8, 
                            sizeof(double));


            // SSR start address need to be configured each time
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image); // ft0
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weight_grads[out*ldW]); // ft1
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weight_grads[out*ldW]); // ft2

        
        }


        // W_checksum = 0.0;
        idx_eff = compute_id + ldB * out;
        // printf("Benchmark GRADIENT UPDATE FP8 with SSRs: activations_fp32[%u] = %f\n", idx_eff, activations_fp32[out]);

        b_grad_update = (idx_eff == *target) ? activations_fp32[ldB * out] - 1 : activations_fp32[ldB * out];
        bias_grads[ldB * out] = b_grad_update;

        // printf("Benchmark GRADIENT UPDATE FP8 with SSRs: bias_grads = %f\n", bias_grads[ldB * out]);

        snrt_ssr_enable();

        asm volatile(
            "vfcpka.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
            "vfcpkb.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
            "vfcpkc.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
            "vfcpkd.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
        //     // "vfcpka.s.s       %[W_grad_update_reg], %[zero], %[zero] \n"
            "frep.o           %[n_frep], 2, 0, 0 \n"
        //     // "vfdotpex.h.b     %[c], ft1, ft0\n" // for debugging
            "vfmul.b          %[W_grad_update_reg], %[b_grad_update_reg], ft0 \n"
            "vfadd.b          ft2, %[W_grad_update_reg], ft1 \n"
            : [b_grad_update_reg] "+&f"(b_grad_update_reg), [W_grad_update_reg] "+&f"(W_grad_update_reg)
            : [b_grad_update] "f"(b_grad_update), [zero] "f"(0.0), [n_frep] "r"(IN_CH / 8 - 1)
            : "ft0", "ft1", "ft2"
        );

        // asm volatile(
        //     "vfcpka.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
        //     "vfcpkb.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
        //     "vfcpkc.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
        //     "vfcpkd.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
        // //     // "vfcpka.s.s       %[W_grad_update_reg], %[zero], %[zero] \n"
        //     "frep.o           %[n_frep], 14, 0, 0 \n"
        // //     // "vfdotpex.h.b     %[c], ft1, ft0\n" // for debugging
        //     "vfmul.b          %[W_grad_update_reg_0], %[b_grad_update_reg], ft0 \n" 
        //     "vfmul.b          %[W_grad_update_reg_1], %[b_grad_update_reg], ft0 \n"
        //     "vfmul.b          %[W_grad_update_reg_2], %[b_grad_update_reg], ft0 \n"
        //     "vfmul.b          %[W_grad_update_reg_3], %[b_grad_update_reg], ft0 \n"
        //     "vfmul.b          %[W_grad_update_reg_4], %[b_grad_update_reg], ft0 \n"
        //     "vfmul.b          %[W_grad_update_reg_5], %[b_grad_update_reg], ft0 \n"
        //     "vfmul.b          %[W_grad_update_reg_6], %[b_grad_update_reg], ft0 \n"
        //     "vfadd.b          ft2, %[W_grad_update_reg_0], ft1 \n"
        //     "vfadd.b          ft2, %[W_grad_update_reg_1], ft1 \n"
        //     "vfadd.b          ft2, %[W_grad_update_reg_2], ft1 \n"
        //     "vfadd.b          ft2, %[W_grad_update_reg_3], ft1 \n"
        //     "vfadd.b          ft2, %[W_grad_update_reg_4], ft1 \n"
        //     "vfadd.b          ft2, %[W_grad_update_reg_5], ft1 \n"
        //     "vfadd.b          ft2, %[W_grad_update_reg_6], ft1 \n"
        //     : [b_grad_update_reg] "+&f"(b_grad_update_reg), [W_grad_update_reg_0] "+&f"(W_grad_update_reg[0].f64), 
        //       [W_grad_update_reg_1] "+&f"(W_grad_update_reg[1].f64), [W_grad_update_reg_2] "+&f"(W_grad_update_reg[2].f64), 
        //       [W_grad_update_reg_3] "+&f"(W_grad_update_reg[3].f64), [W_grad_update_reg_4] "+&f"(W_grad_update_reg[4].f64), 
        //       [W_grad_update_reg_5] "+&f"(W_grad_update_reg[5].f64), [W_grad_update_reg_6] "+&f"(W_grad_update_reg[6].f64)
        //     : [b_grad_update] "f"(b_grad_update), [zero] "f"(0.0), [n_frep] "r"(IN_CH / (8 * unroll) - 1)
        //     : "ft0", "ft1", "ft2"
        // );

        snrt_ssr_disable();
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

        // for (uint32_t in = 0; in < IN_CH; in++) {
            
        //     if(weight_grads[out*ldW + in] == 125 || weight_grads[out*ldW + in] == 255 || weight_grads[out*ldW + in] == 254 || weight_grads[out*ldW + in] == 126){	
        //         printf("WARNING: weight gradient is NaN\n");
        //         weight_grads[out*ldW + in] = 0;
        //     } else if (weight_grads[out*ldW + in] == 124){
        //         printf("WARNING: weight gradient is +Inf at index %u\n", compute_id + out*ldW + in);
        //         weight_grads[out*ldW + in] = 0;
        //     } else if (weight_grads[out*ldW + in] == 252) {
        //         printf("WARNING: weight gradient is -Inf at index %u\n", compute_id + out*ldW + in);
        //         weight_grads[out*ldW + in] = 0;
        //     }


        //     W_grad_acc += weight_grads[out*ldW + in];
        //     // printf("image[%u] = %d\n", in, image[in]);
        //     // printf("Benchmark GRADIENT UPDATE FP8 with SSRs: weight_grads[%u] = %d\n", out*ldW + in, weight_grads[out*ldW + in]);
        //     W_checksum_fp8_reg.vec[0] = weight_grads[out*ldW + in];
        //     W_checksum_fp8_reg.vec[1] = 0.0;
        //     W_checksum_fp8_reg.vec[2] = 0.0;
        //     W_checksum_fp8_reg.vec[3] = 0.0;
        //     W_checksum_fp8_reg.vec[4] = 0.0;
        //     W_checksum_fp8_reg.vec[5] = 0.0;
        //     W_checksum_fp8_reg.vec[6] = 0.0;
        //     W_checksum_fp8_reg.vec[7] = 0.0;

        //     asm volatile (
        //         "vfcpka.s.s     %[W_checksum_fp16_reg], %[zero], %[zero] \n"
        //         "vfcpka.s.s     %[W_checksum_fp32_reg], %[zero], %[zero] \n"
        //         "vfsumex.h.b    %[W_checksum_fp16_reg], %[W_checksum_fp8_reg] \n"   // 8x8 -> 4x16
        //         "vfsumex.s.h    %[W_checksum_fp32_reg], %[W_checksum_fp16_reg] \n"  // 4x16 -> 2x32

        //     : [W_checksum_fp16_reg] "+&f"(W_checksum_fp16_reg), [W_checksum_fp32_reg] "+&f"(W_checksum_fp32_reg)
        //     : [W_checksum_fp8_reg] "f"(W_checksum_fp8_reg.f64), [zero] "f"(0.0f)
        //     : "ft0", "ft1", "ft2"
        //     );

        //     // check if the checksum is nan
        //     if(W_checksum_fp32_reg[0] >= FLT_MAX) {
        //         printf("An overflow occured in the weight checksum calculation at index %u! weight_grad[%u] = %d\n", compute_id + out*ldW + in, compute_id + out*ldW + in, weight_grads[out*ldW + in]);
        //         W_checksum_fp32_reg[0] = 0.0;
        //     } else if (W_checksum_fp32_reg[0] <= -FLT_MAX) {
        //         printf("An underflow occured in the weight checksum calculation at index %u! weight_grad[%u] = %d\n", compute_id + out*ldW + in, compute_id + out*ldW + in, weight_grads[out*ldW + in]);
        //         W_checksum_fp32_reg[0] = 0.0;
        //     }

        //     W_checksum_fp32 += W_checksum_fp32_reg[0];
        // }

        // W_acc_fp8_reg.vec[0] = W_grad_acc;
        // W_acc_fp8_reg.vec[1] = 0.0;
        // W_acc_fp8_reg.vec[2] = 0.0;
        // W_acc_fp8_reg.vec[3] = 0.0;
        // W_acc_fp8_reg.vec[4] = 0.0;
        // W_acc_fp8_reg.vec[5] = 0.0;
        // W_acc_fp8_reg.vec[6] = 0.0;
        // W_acc_fp8_reg.vec[7] = 0.0;

        // asm volatile (
        //         "vfcpka.s.s     %[W_acc_fp16_reg], %[zero], %[zero] \n"
        //         "vfcpka.s.s     %[W_acc_fp32_reg], %[zero], %[zero] \n"
        //         "vfsumex.h.b    %[W_acc_fp16_reg], %[W_acc_fp8_reg] \n"   // 8x8 -> 4x16
        //         "vfsumex.s.h    %[W_acc_fp32_reg], %[W_acc_fp16_reg] \n"  // 4x16 -> 2x32

        //         : [W_acc_fp16_reg] "+&f"(W_acc_fp16_reg), [W_acc_fp32_reg] "+&f"(W_acc_fp32_reg)
        //         : [W_acc_fp8_reg] "f"(W_acc_fp8_reg.f64), [zero] "f"(0.0f)
        //         : "ft0", "ft1", "ft2"
        // );

        // printf("Benchmark GRADIENT UPDATE FP8 with SSRs: W_checksum_fp32[%u] = %f\n", idx_eff, W_checksum_fp32);
        // printf("Benchmark GRADIENT UPDATE FP8 with SSRs: W_acc_fp32[%u] = %f\n", idx_eff, W_acc_fp32_reg[0]);
        // // printf("Benchmark GRADIENT UPDATE FP8 with SSRs: W_acc_fp32[1][%u] = %f\n", idx_eff, W_acc_fp32_reg[1]);

    }

    snrt_cluster_hw_barrier();

}

void benchmark_training_step_fp8_opt(uint32_t IN_CH, uint32_t OUT_CH, 
                char *weights, char *weight_grads, uint32_t ldW, float *biases, float *bias_grads,
                uint32_t ldB, uint32_t setup_SSR) {


    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

    float lr = 0.5;
    // __fp16 W_checksum = 0.0;
    // uint16_t idx_eff;

    register v8f8 lr_reg;

    /// UNROLLED VERSION
    // const uint32_t unroll = 7;
    // register v8s W_update_reg[unroll];
    /// END UNROLLED VERSION

    asm volatile (
        "vfcpka.b.s     %[lr_reg], %[lr], %[lr] \n"
        "vfcpkb.b.s     %[lr_reg], %[lr], %[lr] \n"
        "vfcpkc.b.s     %[lr_reg], %[lr], %[lr] \n"
        "vfcpkd.b.s     %[lr_reg], %[lr], %[lr] \n"
        : [lr_reg] "+&f"(lr_reg)
        : [lr] "f"(-lr)
        : "ft0", "ft1", "ft2"
    );

    for (uint32_t out = 0; out < OUT_CH; out++) {
        
        if (setup_SSR) {
            
            // SSR read setup of weight gradients
            snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                            IN_CH / 8, 
                            sizeof(double));

            // SSR read setup of weights
            snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                            IN_CH / 8, 
                            sizeof(double));

            // SSR write setup of weights
            snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                            IN_CH / 8, 
                            sizeof(double));
        }

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &weight_grads[out*ldW]); // weight gradients stored in ft0
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]); // weights stored in ft1 for read
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weights[out*ldW]); // weights stored in ft2 for write

        biases[ldB * out] -= lr * bias_grads[ldB * out]; 

        // printf("Benchmark TRAINING STEP FP8 with SSRs: bias_grads = %f\n", bias_grads[ldB * out]);


        // Start of SSR region
        snrt_ssr_enable();

        asm volatile(
            "frep.o          %[n_frep], 2, 0, 0 \n"
            "vfmul.h         ft5, ft0, %[lr_vec] \n"
            "vfadd.h         ft2, ft1, ft5 \n"
            : 
            : [n_frep] "r"(IN_CH / 8 - 1), [lr_vec] "f"(lr_reg)
            : "ft0", "ft1", "ft2"
        );

        // asm volatile(
        //     "frep.o          %[n_frep], 14, 0, 0 \n"
        //     "vfmul.h         %[W_update_reg_0], ft0, %[lr_vec] \n"
        //     "vfmul.h         %[W_update_reg_1], ft0, %[lr_vec] \n"
        //     "vfmul.h         %[W_update_reg_2], ft0, %[lr_vec] \n"
        //     "vfmul.h         %[W_update_reg_3], ft0, %[lr_vec] \n"
        //     "vfmul.h         %[W_update_reg_4], ft0, %[lr_vec] \n"
        //     "vfmul.h         %[W_update_reg_5], ft0, %[lr_vec] \n"
        //     "vfmul.h         %[W_update_reg_6], ft0, %[lr_vec] \n"
        //     "vfadd.h         ft2, ft1, %[W_update_reg_0] \n"
        //     "vfadd.h         ft2, ft1, %[W_update_reg_1] \n"
        //     "vfadd.h         ft2, ft1, %[W_update_reg_2] \n"
        //     "vfadd.h         ft2, ft1, %[W_update_reg_3] \n"
        //     "vfadd.h         ft2, ft1, %[W_update_reg_4] \n"
        //     "vfadd.h         ft2, ft1, %[W_update_reg_5] \n"
        //     "vfadd.h         ft2, ft1, %[W_update_reg_6] \n"
        //     : [W_update_reg_0] "+&f" (W_update_reg[0].f64), [W_update_reg_1] "+&f" (W_update_reg[1].f64),
        //       [W_update_reg_2] "+&f" (W_update_reg[2].f64), [W_update_reg_3] "+&f" (W_update_reg[3].f64), 
        //       [W_update_reg_4] "+&f" (W_update_reg[4].f64), [W_update_reg_5] "+&f" (W_update_reg[5].f64), 
        //       [W_update_reg_6] "+&f" (W_update_reg[6].f64)
        //     : [n_frep] "r"(IN_CH / (8 * unroll) - 1), [lr_vec] "f"(lr_reg)
        //     : "ft0", "ft1", "ft2"
        // );

        snrt_ssr_disable();
        // INFO: after disabling the SSRs we can free the registers
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));       


    }

}