// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_fp16_network.h"

#include "printf.h"
#include "snrt.h"

typedef float v2f32 __attribute__((vector_size(8)));
union fp32_v2f32_u { 
        v2f32 v2; 
        float v[2]; 
};
typedef __fp16 v4f16 __attribute__((vector_size(8)));
union fp16_v4f16_u { 
        v4f16 v4; 
        __fp16 v[4]; 
};
typedef char v8f8 __attribute__((vector_size(8)));

// INFO: start of FP16 baseline network implementation
//// Feedforward Step
void feedforward_fp16n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *biases, __fp16 *activations,
                uint32_t ldB, __fp16 *image, uint32_t ldI, uint32_t compute_id){

    // Linear layer: OUT = X * W^T + B
    for (uint32_t out = 0; out < OUT_CH; out++) {
        __fp16 acc = biases[ldB * out];
        //printf("FP16 baseline init: acc[%u] = %f\n", 1 + compute_id + out * ldB, acc);
        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            acc += image[in] * weights[out * ldW + in];
            // INFO: If this is not set harts start reading outside the mem map
            // FIXME: Next harts should start computation of the subsequent image
            if(compute_id + out * ldB > OUT_CH * 5 - 1){
                acc = 0;
            }
        }
        // OUT is accumulated in activations 
        activations[ldB * out] = acc;
        printf("FEEDFORWARD FP16 Baseline: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
        //printf("Core %u done with the computation: core_sync[%u] = %u.\n", compute_id + 1, compute_id + 1, core_sync);   
    }

    snrt_cluster_hw_barrier();

}

//// Activation Step
void softmax_activation_fp16n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *activations, uint32_t ldB,
                __fp16 *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, __fp16 *max){

    __fp16 max_core;
    __fp16 sum = 0.0f;
    __fp16 *act_ptr;
    //printf("DEBUG: activation[%u] = %f\n", compute_id + 1, activations[0]); OK

    max_core = activations[0];

    for(uint32_t out = 0; out < OUT_CH; out++){
        //printf("DEBUG: activation[%u] = %f\n", compute_id + 1 + out * ldB, activations[out * ldB]);
        if(activations[ldB * out] > max_core) {
            max_core = activations[ldB * out];
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
        for(uint32_t out = 0; out < OUT_CH*5; out++){
            act_ptr = &activations[0];
            //printf("DEBUG: act_ptr[%u] = %f\n", out + 1, act_ptr[out]);
            if(act_ptr[out] != 0.0f){
                //printf("DEBUG NON ZERO: act_ptr[%u] = %f\n", out, act_ptr[out]);
                act_ptr[out] = exp(act_ptr[out] - max_global);
                //printf("DEBUG: act_ptr[%u] = %f\n", out, act_ptr[out]);
                sum += act_ptr[out];
                //printf("DEBUG: sum = %f\n", sum);
            } else {
                //printf("DEBUG ZERO: act_ptr[%u] = %f\n", out, act_ptr[out]);
                act_ptr[out] = 0.0;
            }
        }


        for(uint32_t out = 0; out < OUT_CH*5; out++){
            act_ptr[out] /= sum;
            activations[out] = act_ptr[out];
            printf("SOFTMAX FP16 (no SIMD): activation[%u] = %f\n", out + 1, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
}

void gradient_update_fp16n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weight_grads, uint32_t ldW, __fp16 *bias_grads, __fp16 *activations, 
                uint32_t ldB, __fp16 *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, __fp16 *loss, uint32_t compute_num){

    
    __fp16 b_grad_update = 0.0;
    __fp16 W_grad_update = 0.0;
    __fp16 b_checksum = 0.0;
    __fp16 W_checksum = 0.0;
    volatile uint32_t idx_eff;


    // get the value saved at target address
    uint32_t target_n = *target;
    // compute the loss
    // TODO: check if indexig of target var correct
    //printf("DEBUG: activations[%u] = %f\n", target_n - compute_id, activations[target_n - compute_id]);
    __fp16 loss_val = 0.0 - log(activations[target_n - compute_id]);

    // save the value into the loss pointer
    if(!compute_id){
        loss[0] += loss_val;
    } else {
        loss[0] += 0;
    }

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    

    // the effective index is the iteration index of the biases variable
    // across all entries
    for(uint32_t out = 0; out < OUT_CH; out++){
        // printf("bias grads[%u] = %f\n", compute_id + ldB * out, bias_grads[ldB * out]);
        // printf("biases[%u] = %f\n", compute_id + ldB * out, biases[ldB * out]);
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        b_checksum += b_grad_update;

        //printf("b_grad_update = %f\n", b_grad_update);

        for(uint32_t in = 0; in < IN_CH; in++){
            
            W_grad_update = b_grad_update * image[in];
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH* (OUT_CH - 1) * 5)){
                weight_grads[out * ldW + in] += W_grad_update;
                W_checksum += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update;
        // printf("GRADIENT UPDATE FP16 Baseline: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    }

    printf("GRADIENT UPDATE FP16 Baseline: W_checksum = %f\n", W_checksum);
    printf("GRADIENT UPDATE FP16 Baseline: b_checksum = %f\n", b_checksum);

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

}

void training_step_fp16n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, __fp16 *weight_grads, uint32_t ldW, __fp16 *biases, __fp16 *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images){

    __fp16 lr = 0.5;
    __fp16 b_checksum = 0.0;
    __fp16 W_checksum = 0.0;

    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    for(uint32_t out = 0; out < OUT_CH; out++){

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((__fp16) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        b_checksum += biases[ldB * out];

        for(uint32_t in = 0; in < IN_CH; in++){
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH * (OUT_CH - 1) * 5)){
                // if(!compute_id){
                //     printf("weight grad[%u] = %f\n", compute_id + out * ldW + in, weight_grads[out * ldW + in]);
                // }
                // if(!compute_id){
                //     printf("weight[%u] = %f\n", compute_id + out * ldW + in, weights[out * ldW + in]);
                // }
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((__fp16) number_of_images);
                // if(!compute_id){
                //     printf("weight[%u] = %f\n", compute_id + out * ldW + in, weights[out * ldW + in]);
                // }
                W_checksum += weights[out * ldW + in];
            } 
        }
    }

    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("FP16 baseline: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    // }
    printf("TRAINING STEP FP16 baseline: W_checksum = %f\n", W_checksum);
    printf("TRAINING STEP FP16 baseline: b_checksum = %f\n", b_checksum);
}

void feedforward_fp16_ssr_simd_frep(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *biases, __fp16 *activations,
                uint32_t ldB, __fp16 *image, uint32_t ldI, uint32_t compute_id,
                uint32_t setup_SSR){
    
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    
    // INFO: for simplicity image is converted to dtype __fp16 --> discuss with GIM 
    const register float zero = 0.0f;

    __fp16 acc = 0.0f;

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    volatile uint32_t idx_eff;

    // SSR strides and bounds only have to be configured
    // once in the beginning
    // WARN In the RTL SSR strides MUST BE of size DOUBLE

    if (setup_SSR) {

        // setup of DATA MOVER input data (MNIST image)
        snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                        IN_CH / 4, 
                        sizeof(double));
        
        // setup of DATA MOVER for weights
        snrt_ssr_loop_2d(SNRT_SSR_DM1, 
                        IN_CH / 4, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW / 4);
    }


    for (uint32_t out = 0; out < OUT_CH; out++) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        register v2f32 reduce_reg;
        register float sum;
        register v2f32 test;
        register v4f16 dotp;
        register v4f16 zero_reg;
        register float tacc;
        // add dummy register for SSRs
        register float dummy;

        idx_eff = compute_id + ldB * out;

        // Start of SSR region
        snrt_ssr_enable();
        if(!(idx_eff > OUT_CH * 5 - 1)){
            
            acc = biases[ldB * out];
            
            asm volatile(
                "vfcpka.s.s    %[tacc], %[zero], %[zero]\n" // zero initialize accumulator
                "vfcpka.h.s    %[zero_reg], %[zero], %[zero] \n"
                : [tacc] "+&f"(tacc), [zero_reg] "+&f"(zero_reg)
                : [zero] "f"(zero)
            );
            
                
    //         // calculate the dot product of the image and the weights (increment by four columns in each iteration)
            asm volatile(
                "frep.o           %[n_frep], 4, 0, 0\n"
                "vfcpka.s.s       %[dotp], %[zero], %[zero]\n" // initialize the dot product with zeros
                "vfcpka.s.s       %[sum], %[zero], %[zero] \n" // initialize the sum with zeros
                "vfdotpex.s.h     %[dotp], ft1, ft0 \n"
                "vfsum.s          %[sum], %[dotp] \n"
    //             "fadd.s           %[tacc], %[tacc], %[sum] \n"
            : [sum] "+f"(sum), [dotp] "+f"(dotp), [tacc] "+f"(tacc)
            : [zero] "f"(zero), [n_frep] "r"(IN_CH / 4 - 1)
            : "ft0", "ft1", "ft2"
            );
            
    //         // snrt_ssr_disable();
    //         // printf("DEBUG: tacc[%u] = %f\n", compute_id + out + 1, tacc);
    //         // printf("DEBUG: acc[%u] = %f\n", compute_id + out + 1, acc);
    //         // printf("DEBUG: tacc[%u] + acc[%u] = %f\n", compute_id + out + 1, compute_id + out + 1, tacc + acc);
    //         snrt_ssr_enable();
    //         acc += tacc;
    //         printf("DEBUG: acc_up[%u] = %f\n", compute_id + out + 1, acc);
    //         snrt_ssr_disable();
    //         // snrt_ssr_enable();

        } else {
            asm volatile(
                "frep.o           %[n_frep], 1, 0, 0\n"
                "vfadd.s          %[dummy], ft1, ft0 \n"
                : [dummy] "+f"(dummy)
                : [n_frep] "r"(IN_CH / 4 - 1)
                : "ft0", "ft1", "ft2"
            );
        }

        activations[ldB * out] = acc;
        acc = 0.0;

        // End of SSR region.
        snrt_fpu_fence();
        snrt_ssr_disable();
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));


    }


    for (uint32_t out = 0; out < OUT_CH; out++) {
        printf("FP16 SIMD with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    }

    snrt_cluster_hw_barrier();

}

//// Gradient Update
void gradient_update_fp16_ssr_simdn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weight_grads, uint32_t ldW, __fp16 *bias_grads, __fp16 *activations, 
                uint32_t ldB, __fp16 *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, __fp16 *loss, uint32_t compute_num, uint32_t setup_SSR){

    float b_grad_update = 0.0f;
    __fp16 b_checksum = 0.0f;
    __fp16 W_grad_update = 0.0f;
    __fp16 W_checksum = 0.0f;
    volatile uint32_t idx_eff;

    // get the value saved at target address
    uint32_t target_n = *target;
    
    // compute the loss
    __fp16 loss_val = 0.0 - log(activations[target_n -compute_id]);

    // save the value into the loss pointer
    if(!compute_id){
        loss[0] += loss_val;
    } else {
        loss[0] += 0;
    }

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;


    // SSR strides and bounds only have to be configured
    // once in the beginning
    if (setup_SSR) {

        // setup of input data (MNIST image)
        snrt_ssr_loop_2d(SNRT_SSR_DM0, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldI);
        
        // SSR setup of weights
        snrt_ssr_loop_2d(SNRT_SSR_DM1, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW);

    }

    // TODO: check if this is actually correct ...


    for(uint32_t out = 0; out < OUT_CH; out++){
        // SSR start address need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weight_grads[out*ldW]);
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        // snrt_cluster_hw_barrier();
        b_checksum += b_grad_update;
        // Discuss with GIM why the following printf are necessary for correct output
        printf("DEBUG: EFFECTIVE b_grad_update[%u] = %f\n", compute_id + out * ldB + 1, b_grad_update);
        printf("DEBUG: EFFECTIVE b_checksum[%u] = %f\n", compute_id + out * ldB + 1, b_checksum);
        register float sum;
        register float acc;
        register v4f16 reduce_reg;
        register v4f16 dotp;
        asm volatile (
            "vfcpka.h.s       %[reduce_reg], %[b_grad_update], %[b_grad_update]\n"
            "vfcpkb.h.s       %[reduce_reg], %[b_grad_update], %[b_grad_update]\n"
            "fadd.s           %[acc], %[zero], %[zero]\n"
        : [reduce_reg] "+&f"(reduce_reg), [dotp] "+&f"(dotp), [acc] "+&f"(acc)
        : [b_grad_update] "f"(b_grad_update), [zero] "f"(0.0f)
        : "ft0", "ft1", "ft2"
        );
        // TODO: check the SSR enable/disable signals in each loop!!
        snrt_ssr_enable();
        for(uint32_t in = 0; in < IN_CH;){
            if(!(compute_id * IN_CH + out * ldW + in + 3 > IN_CH * (OUT_CH-1) * 5)){
                asm volatile(
                    "vfcpka.s.s       %[dotp], %[zero], %[zero]\n"
                    "vfcpka.s.s       %[sum], %[zero], %[zero]\n"
                    "vfmul.h          %[dotp], %[reduce_reg], ft0\n"

                : [reduce_reg] "+&f"(reduce_reg), [dotp] "+&f"(dotp), [sum] "+&f"(sum)
                : [zero] "f"(0.0f)
                : "ft0", "ft1", "ft2"
                );

                    //printf("DEBUG: index = %u\n", compute_id * IN_CH + out * ldW + in + 4);
                    snrt_ssr_disable(); // Discuss with GIM why we have to disable SSR here
                    // printf(("DEBUG: W_checksum update = %f\n"), dotp[0] + dotp[1] + dotp[2] + dotp[3]);
                    weight_grads[out*ldW + in + 0] += dotp[0];
                    weight_grads[out*ldW + in + 1] += dotp[1];
                    weight_grads[out*ldW + in + 2] += dotp[2];
                    weight_grads[out*ldW + in + 3] += dotp[3];
                    W_checksum += dotp[0] + dotp[1] + dotp[2] + dotp[3];
                    snrt_ssr_enable();
            }

            in += 4;
        }

        // snrt_cluster_hw_barrier();

        snrt_ssr_disable();

        bias_grads[ldB * out] = b_grad_update;

    }

    printf("GRADIENT UPDATE FP16 SIMD with SSRs: W_checksum = %f\n", W_checksum);
    printf("GRADIENT UPDATE FP16 SIMD with SSRs: b_checksum = %f\n", b_checksum);

    snrt_cluster_hw_barrier();

}

//// Training Step
void training_step_fp16_ssr_simdn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, __fp16 *weight_grads, uint32_t ldW, __fp16 *biases, __fp16 *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR){

    float lr = 0.5;

    // convert number of images to float for vectorized computation
    float nimg = ((float)number_of_images);
    
    register v4f16 lr_vec;
    register v4f16 nimg_vec;

    __fp16 b_checksum = 0.0f;
    __fp16 W_checksum = 0.0f;

    const register v4f16 zero = {b_checksum, b_checksum, b_checksum, b_checksum};

    // pack the learning rate and number of images into a vector for vectorized computation
    // snrt_ssr_enable();
    // asm volatile(
    //     "vfcpka.h.s          %[lr_vec], %[lr], %[lr] \n"
    //     "vfcpkb.h.s          %[lr_vec], %[lr], %[lr] \n"
    //     "vfcpka.h.s          %[nimg_vec], %[nimg], %[nimg] \n"
    //     "vfcpkb.h.s          %[nimg_vec], %[nimg], %[nimg] \n"
    //     : [lr_vec] "+&f"(lr_vec), [nimg_vec] "+&f"(nimg_vec)
    //     : [lr] "f"(-0.5f), [nimg] "f"(1.0f)
    //     : "ft0", "ft1", "ft2"
    // );
    // snrt_ssr_disable();

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    // WARN In the RTL SSR strides MUST BE of size DOUBLE

    // SSR strides and bounds only have to be configured
    // once in the beginning
    if (setup_SSR) {
        
        // SSR setup of weight gradients
        snrt_ssr_loop_2d(SNRT_SSR_DM0, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW);

        // SSR setup of weights
        snrt_ssr_loop_2d(SNRT_SSR_DM1, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW);
    }


    for(uint32_t out = 0; out < OUT_CH; out++){

        
        // NOTE: we don't use SSRs for biases, as overhead too big
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((float) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        b_checksum += biases[ldB * out];
        // Discuss with GIM why we need the printfs here
        printf("DEBUG: bias checksum update = %f\n", biases[ldB * out]);
        printf("DEBUG: bias checksum = %f\n", b_checksum);
        
        // SSR start addresses need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &weight_grads[out*ldW]);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        register v4f16 reduce_reg;
        register v4f16 weight_reg;
        // snrt_cluster_hw_barrier();
        snrt_ssr_enable();
        for(uint32_t in = 0; in < IN_CH;){
            if(!(compute_id*IN_CH + out * ldW + in + 3 > IN_CH * (OUT_CH - 1) * 5)){
                //weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images); FP32 reference
                asm volatile(
                    "vfadd.s                %[reduce_reg], %[zero], ft0 \n"                 // load the weight gradients into a vector
                    "vfadd.s                %[weight_reg], %[zero], ft1 \n"                 // load the weights into a vector
                    // "vfdiv.s              %[reduce_reg], %[reduce_reg], %[nimg_vec] \n"     // divde by the size of the dataset --> banshee: add floating point exception for divide by zero
                : [reduce_reg] "+&f"(reduce_reg), [weight_reg] "+&f"(weight_reg)
                : [lr_vec] "f"(lr_vec), [nimg_vec] "f"(nimg_vec), [zero] "f"(zero)
                : "ft0", "ft1", "ft2"
                );  

                snrt_ssr_disable(); // Discuss with GIM why we have to disable SSR here
                // printf(("DEBUG: W_checksum update = %f\n"), reduce_reg[0] + reduce_reg[1] + reduce_reg[2] + reduce_reg[3]);
                // if(!compute_id){
                //     printf("weight grad[%u] = %f\n", compute_id + out * ldW + in, reduce_reg[0]);
                //     printf("weight grad[%u] = %f\n", compute_id + out * ldW + in + 1, reduce_reg[1]);
                //     printf("weight grad[%u] = %f\n", compute_id + out * ldW + in + 2, reduce_reg[2]);
                //     printf("weight grad[%u] = %f\n", compute_id + out * ldW + in + 3, reduce_reg[3]);
                // }
                // if(!compute_id){
                //     printf("weights[%u] = %f\n", compute_id + out * ldW + in, weights[out*ldW + in + 0]);
                //     printf("weights[%u] = %f\n", compute_id + out * ldW + in + 1, weights[out*ldW + in + 1]);
                //     printf("weights[%u] = %f\n", compute_id + out * ldW + in + 2, weights[out*ldW + in + 2]);
                //     printf("weights[%u] = %f\n", compute_id + out * ldW + in + 3, weights[out*ldW + in + 3]);
                // }
                weights[out*ldW + in + 0] -= lr * reduce_reg[0] / ((float) number_of_images);
                weights[out*ldW + in + 1] -= lr * reduce_reg[1] / ((float) number_of_images);
                weights[out*ldW + in + 2] -= lr * reduce_reg[2] / ((float) number_of_images);
                weights[out*ldW + in + 3] -= lr * reduce_reg[3] / ((float) number_of_images);
                // if(!compute_id){
                //     printf("weights[%u] = %f\n", compute_id + out * ldW + in, weights[out*ldW + in + 0]);
                //     printf("weights[%u] = %f\n", compute_id + out * ldW + in + 1, weights[out*ldW + in + 1]);
                //     printf("weights[%u] = %f\n", compute_id + out * ldW + in + 2, weights[out*ldW + in + 2]);
                //     printf("weights[%u] = %f\n", compute_id + out * ldW + in + 3, weights[out*ldW + in + 3]);
                // }
                W_checksum += weights[out*ldW + in + 0] + weights[out*ldW + in + 1] + weights[out*ldW + in + 2] + weights[out*ldW + in + 3];
                snrt_ssr_enable();
            } 

            in += 4;
        }
        snrt_ssr_disable();


    }


    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("FP16 with SSRs and SIMD: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    // }
    printf("TRAINING STEP FP16 SIMD with SSRs: W_checksum = %f\n", W_checksum);
    printf("TRAINING STEP FP16 SIMD with SSRs: b_checksum = %f\n", b_checksum);

}