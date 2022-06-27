// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_fp32_network.h"

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

// INFO: start of FP32 baseline network implementation
void feedforward_fp32n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *biases, float *activations,
                uint32_t ldB, float *image, uint32_t ldI, uint32_t compute_id){
    
    // Linear layer: OUT = X * W^T + B
    for (uint32_t out = 0; out < OUT_CH; out++) {
        register float acc = biases[ldB * out];
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
        printf("FEEDFORWARD FP32 Baseline: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);  
    }

    snrt_cluster_hw_barrier();

}

//// Activation Step
void softmax_activation_fp32n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *activations, uint32_t ldB,
                float *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, float *max){

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

        // FIXME: actually OUT_CH should be multiplied by number of compute cores
        for(uint32_t out = 0; out < OUT_CH*5; out++){
            if(activations[out]){
                activations[out] = exp(activations[out] - max_global);
                sum += activations[out];
            } else {
                activations[out] = 0.0;
            }
        }


        for(uint32_t out = 0; out < OUT_CH*5; out++){
            activations[out] /= sum;
            printf("SOFTMAX FP32 (no SIMD): activation[%u] = %f\n", out + 1, activations[out]);
        }
    }
    snrt_cluster_hw_barrier();
}

void gradient_update_fp32n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weight_grads, uint32_t ldW, float *bias_grads, float *activations, 
                uint32_t ldB, float *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, float *loss, uint32_t compute_num){

    
    float b_grad_update = 0.0;
    float W_grad_update = 0.0;
    float b_checksum = 0.0;
    float W_checksum = 0.0;
    volatile uint32_t idx_eff;

    const uint32_t IN_CH = IN_CH1 * IN_CH2;


    // get the value saved at target address
    uint32_t target_n = *target;
    // compute the loss
    float loss_val = 0.0 - log(activations[target_n -compute_id]);

    // save the value into the loss pointer
    if(!compute_id){
        loss[0] += loss_val;
    } else {
        loss[0] += 0;
    }

    
    //printf("loss = %f\n", loss[0]);
    

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
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH * (OUT_CH * 5 - 1))){
                weight_grads[out * ldW + in] += W_grad_update;
                W_checksum += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update;
        // printf("GRADIENT UPDATE FP32 Baseline: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    }

    printf("GRADIENT UPDATE FP32 Baseline: b_checksum = %f\n", b_checksum);
    printf("GRADIENT UPDATE FP32 Baseline: W_checksum = %f\n", W_checksum);

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

}

void training_step_fp32n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, float *weight_grads, uint32_t ldW, float *biases, float *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images){

    float lr = 0.5;
    float b_checksum = 0.0;
    float W_checksum = 0.0;

    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    for(uint32_t out = 0; out < OUT_CH; out++){

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((float) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        b_checksum += biases[ldB * out];

        for(uint32_t in = 0; in < IN_CH; in++){
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH * (OUT_CH * 5 - 1))){
                // printf("DEBUG: weight grads[%u] = %f\n", compute_id*IN_CH + out * ldW + in, weight_grads[out*ldW + in]);
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images);
                W_checksum += weights[out * ldW + in];
            } 
        }
    }

    printf("new TRAINING STEP FP32 Baseline: b_checksum = %f\n", b_checksum);
    printf("new TRAINING STEP FP32 Baseline: W_checksum = %f\n", W_checksum);
}

// INFO: start of FP32 network implementation using SSRs and SIMD instructions
//// Feedforward Step
void feedforward_fp32_ssr_simd_frep(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *biases, float *activations,
                uint32_t ldB, float *image, uint32_t ldI, uint32_t compute_id,
                uint32_t setup_SSR){
    
    register float acc = 0.0;

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    // SSR strides and bounds only have to be configured
    // once in the beginning
    // WARN In the RTL SSR strides MUST BE of size DOUBLE

    if (setup_SSR) {

        // setup of DATA MOVER input data (MNIST image)
        snrt_ssr_loop_2d(SNRT_SSR_DM0, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldI);
        
        // setup of DATA MOVER for weights
        snrt_ssr_loop_2d(SNRT_SSR_DM1, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW);
    }

    for (uint32_t out = 0; out < OUT_CH; out++) {
        // we need to read the image for every new iteration
        // of a core, because otherwise it will evaluate to
        // all zeros due to the stream semantics
        // Start of SSR region
        snrt_ssr_enable();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        register v2f32 reduce_reg;
        register v2f32 sum;
        const register float zero = 0;
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            acc = biases[ldB * out];
            // INFO: The zero reg causes Out of Memory accesses - WTF? Discuss with GIM --> Issue was missing ft2 clobber (reserved for SSR)
            asm volatile(
                "vfcpka.s.s     %[reduce_reg], %[acc], %[zero] \n"
                "frep.o         %[n_frep], 1, 0, 0 \n"
                "vfmac.s        %[reduce_reg], ft0, ft1 \n"                     // load two values from image and weights into SIMD vector
                // "vfcpka.s.s        %[sum], %[zero], %[zero] \n"
                // "vfsum.s           %[sum], %[reduce_reg] \n"
                // "vfcpka.s.s        %[acc], %[sum], %[zero] \n"
                : [ reduce_reg ] "+f"(reduce_reg), [ acc ] "+f"(acc), [ sum ] "=&f"(sum)
                : [ zero ] "f"(zero), [ n_frep ] "r"(IN_CH / 2 - 1)
                : "ft0", "ft1", "ft2");
        } else {
            acc = 0.0;
        }

        asm volatile(
                "vfcpka.s.s        %[sum], %[zero], %[zero] \n"
                "vfsum.s           %[sum], %[reduce_reg] \n"
                "vfcpka.s.s        %[acc], %[sum], %[zero] \n"
                : [ acc ] "+f"(acc), [ sum ] "=&f"(sum)
                : [ zero ] "f"(zero),  [ reduce_reg ] "f"(reduce_reg)
                :"ft0", "ft1", "ft2");

        activations[ldB * out] = acc;
        acc = 0.0;

        // End of SSR region.
        snrt_ssr_disable();

    }

    for (uint32_t out = 0; out < OUT_CH; out++) {
        // cleanup of leftover columns
        // if(compute_id + out * ldB > OUT_CH * 5 - 1){
        //     activations[ldB * out] = 0;
        // }

        printf("FEEDFORWARD FP32 SIMD with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    }

    snrt_cluster_hw_barrier();

}

//// Gradient Update
void gradient_update_fp32_ssr_simdn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weight_grads, uint32_t ldW, float *bias_grads, float *activations, 
                uint32_t ldB, float *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, float *loss, uint32_t compute_num, uint32_t setup_SSR){


    float b_grad_update = 0.0;
    float W_grad_update = 0.0;
    float b_checksum = 0.0;
    register float W_checksum = 0.0;
    volatile uint32_t idx_eff;
    union fp32_v2f32_u reduce_reg_u;

    // get the value saved at target address
    uint32_t target_n = *target;
    
    // compute the loss
    float loss_val = 0.0 - log(activations[target_n -compute_id]);

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


    for(uint32_t out = 0; out < OUT_CH; out++){
        // SSR start address need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weight_grads[out*ldW]);
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        // INFO: we need to add this, since after DMA transfer out of bound activations
        //       evaluate to -INF
        if(idx_eff + 1 > OUT_CH * 5 - 1){
            b_grad_update = 0.0;
        } else {
            b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        }
        b_checksum += b_grad_update;
        snrt_cluster_hw_barrier(); // Discuss with GIM
        snrt_ssr_enable();
        register v2f32 reduce_reg;
        register const float zero = 0;
        // zero initialze the reduce register for each loop
        asm volatile(
                "vfcpka.s.s      %[reduce_reg], %[zero], %[zero] \n"
                : [ reduce_reg ] "+&f"(reduce_reg)
                : [ zero ] "f"(zero)
                : "ft0", "ft1", "ft2"
        );

        for(uint32_t in = 0; in < IN_CH;){
            asm volatile(
                "vfcpka.s.s         %[reduce_reg], %[b_grad], %[b_grad] \n"       // load the bias gradient for each vector
                "vfmul.s            %[reduce_reg], %[reduce_reg], ft0 \n"         // compute weight update b_grad * image
                "vfadd.s            %[reduce_reg], %[reduce_reg], ft1 \n"         // add weight update to weight gradient
                : [reduce_reg] "+&f"(reduce_reg)
                : [b_grad] "f"(b_grad_update)
                : "ft0", "ft1", "ft2"
            );


            snrt_ssr_disable(); // Discuss with GIM
            weight_grads[out*ldW + in] += reduce_reg[0];
            weight_grads[out*ldW + in + 1] += reduce_reg[1];
            W_checksum += reduce_reg[0] + reduce_reg[1];
            snrt_ssr_enable();
            in += 2;
        }

        bias_grads[ldB * out] = b_grad_update;

        snrt_ssr_disable();

    }


    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("GRADIENT UPDATE FP32 SIMD with SSRs: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    // }

    printf("GRADIENT UPDATE FP32 SIMD with SSRs: b_checksum = %f\n", b_checksum);
    printf("GRADIENT UPDATE FP32 SIMD with SSRs: W_checksum = %f\n", W_checksum);

    snrt_cluster_hw_barrier();
}

//// Training Step
// FIXME: not giving correct weight checksum compared to baseline 
void training_step_fp32_ssr_simdn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, float *weight_grads, uint32_t ldW, float *biases, float *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR){

    float lr = 0.5;

    float b_checksum = 0.0;
    float W_checksum = 0.0;

    // convert number of images to float for vectorized computation
    float nimg = ((float)number_of_images);
    register v2f32 lr_vec;
    register v2f32 nimg_vec;
    // pack the learning rate and number of images into a vector for vectorized computation
    asm volatile(
        "vfcpka.s.s          %[lr_vec], %[lr], %[lr] \n"
        "vfcpka.s.s          %[nimg_vec], %[nimg], %[nimg] \n"
        : [lr_vec] "+&f"(lr_vec), [nimg_vec] "+&f"(nimg_vec)
        : [lr] "f"(-lr), [nimg] "f"(nimg)
        : "ft0", "ft1", "ft2"
    );

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    // SSR strides and bounds only have to be configured
    // once in the beginning
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
        
        snrt_ssr_enable();
        // SSR start addresses need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &weight_grads[out*ldW]); // weight gradients stored in ft0
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]); // weights stored in ft1
        register v2f32 reduce_reg;
        const register float zero = 0.0;
        register v2f32 zero_reg;
        snrt_cluster_hw_barrier();
        for(uint32_t in = 0; in < IN_CH;){
            
            if(!(compute_id*IN_CH + out * ldW + in + 1 > IN_CH * OUT_CH * 5)){
                snrt_ssr_disable();
                // printf("DEBUG: weight grads[%u] = %f\n", compute_id*IN_CH + out * ldW + in, weight_grads[out*ldW + in]);
                snrt_ssr_enable();
                //weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images); FP32 reference
                asm volatile(
                    "vfmul.s              %[reduce_reg], %[lr_vec], ft0 \n"         // compute the weight update
                    "vfdiv.s              %[reduce_reg], %[reduce_reg], %[nimg_vec] \n"     // divde by the size of the dataset --> TODO: banshee: add floating point exception for divide by zero
                    // "vfcpka.s.s              %[zero_reg], %[zero], %[zero] \n"         // vectorized zero register
                    // "vfadd.s                 %[reduce_reg], %[zero_reg], ft0"
                : [reduce_reg] "+&f"(reduce_reg), [zero_reg] "+&f"(zero_reg)
                : [lr_vec] "f"(lr_vec), [nimg_vec] "f"(nimg_vec), [zero] "f"(zero)
                : "ft0", "ft1", "ft2"
                ); 

                // discuss with GIM: can I FREP this somehow?
                snrt_ssr_disable(); // Discuss with GIM: why do we need to disable SSRs?
                weights[out*ldW + in] -= reduce_reg[0];
                weights[out*ldW + in + 1] -= reduce_reg[1];
                W_checksum += reduce_reg[0] + reduce_reg[1];
                snrt_ssr_enable();
            } 

            in += 2;

        }

        snrt_ssr_disable();

    }


    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("FP32 with SSRs and SIMD: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    // }

    printf("FP32 with SSRs and SIMD: b_checksum = %f\n", b_checksum);
    printf("FP32 with SSRs and SIMD: W_checksum = %f\n", W_checksum);

}