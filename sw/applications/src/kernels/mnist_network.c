// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_network.h"

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

// INFO: start of FP64 baseline network implementation

// The output of the feedforward is accumulated in the activations variable
void feedforward_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync){

    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    
    // Linear layer: OUT = X * W^T + B
    for (uint32_t out = 0; out < OUT_CH; out++) {
        //printf("Step: %u\n", out + compute_id);
        register double acc = biases[ldB * out];
        for(uint32_t in = 0; in < IN_CH; in++){
            // acc += image[in] * weights[out * ldW + in];
            // INFO: If this is not set harts start reading outside the mem map
            // FIXME: Next harts should start computation of the subsequent image
            if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
                acc += image[in] * weights[out * ldW + in];
            } else {
                acc = 0;
            }
        }
        // OUT is accumulated in activations 
        activations[ldB * out] = acc;
        printf("FEEDFORWARD FP64 Baseline: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);  
    }

    snrt_cluster_hw_barrier();

} // WORKS on Cluster 0

void softmax_activation_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max, uint32_t* core_sync){

    double max_core;
    double sum = 0.0;

    max_core = activations[0];

    for(uint32_t out = 0; out < OUT_CH; out++){
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            if(activations[ldB * out] > max_core) {
                max_core = activations[ldB * out];
            }
        }
    }

    max[compute_id] = max_core;
    
    snrt_cluster_hw_barrier();

    //printf("Max value of compute core %u is %f\n", compute_id, max_core);

    double max_global = max[0];

    // Reduction on single core
    if(compute_id == 0){
        for(uint32_t core = 0; core < compute_num; core++){
            if(max[core] > max_global){
                max_global = max[core];
            }
        }

        // FIXME: actually OUT_CH should be multiplied by number of compute cores
        for(uint32_t out = 0; out < OUT_CH * 5; out++){
            if(activations[out]){
                activations[out] = exp(activations[out] - max_global);
                sum += activations[out];
            } else {
                activations[out] = 0.0;
            }
        }

        // printf("sum = %f\n", sum);


        for(uint32_t out = 0; out < OUT_CH * 5; out++){
            activations[out] /= sum;
            printf("SOFTMAX FP64 Baseline: activation[%u] = %f\n", out + 1, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
} // WORKS on Cluster 0

void gradient_update_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, double *loss, uint32_t compute_num){

    
    double b_grad_update = 0.0;
    double W_grad_update = 0.0;
    volatile uint32_t idx_eff;

    double b_checksum = 0.0;
    double W_checksum = 0.0;


    // get the value saved at target address
    uint32_t target_n = *target;
    // compute the loss
    double loss_val = 0.0 - log(activations[target_n -compute_id]);

    // save the value into the loss pointer
    if(!compute_id){
        loss[0] += loss_val;
    } else {
        loss[0] += 0;
    }

    
    //printf("loss = %f\n", loss[0]);

    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    

    // the effective index is the iteration index of the biases variable
    // across all entries
    for(uint32_t out = 0; out < OUT_CH; out++){
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];

        // add the update to the bias gradient checksum
        b_checksum += b_grad_update;

        for(uint32_t in = 0; in < IN_CH; in++){
            
            W_grad_update = b_grad_update * image[in];
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH * (OUT_CH * 5 - 1))){
                weight_grads[out * ldW + in] += W_grad_update;
                W_checksum += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update; // INFO: "+" only for debugging to check if bias_grads zero initialized!!
        // printf("GRADIENT UPDATE FP64 Baseline: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    }

    printf("GRADIENT UPDATE FP64 Baseline: b_checksum = %f\n", b_checksum);
    printf("GRADIENT UPDATE FP64 Baseline: W_checksum = %f\n", W_checksum);

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

} // WORKS on Cluster 1

void training_step_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images){

    double lr = 0.5;
    double b_checksum = 0.0;
    double W_checksum = 0.0;

    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    for(uint32_t out = 0; out < OUT_CH; out++){

        // printf("TRAINING STEP FP64 baseline: old biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
        // printf("TRAINING STEP FP64 baseline: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((double) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        b_checksum += biases[ldB * out];

        for(uint32_t in = 0; in < IN_CH; in++){
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH * (OUT_CH * 5 - 1))){
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((double) number_of_images);
                W_checksum += weights[out * ldW + in];
            } 
        }
    }

    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("TRAINING STEP FP64 Baseline: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    // }

    printf("TRAINING STEP FP64 Baseline: b_checksum = %f\n", b_checksum);
    printf("TRAINING STEP FP64 Baseline: W_checksum = %f\n", W_checksum);
}

// INFO: start of FP64 network implementation using SSRs
//// Feedforward Step
void feedforward_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync,
                uint32_t setup_SSR){

    register double acc = 0.0;

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
        
        // setup of weights
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
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        // Start of SSR region
        snrt_ssr_enable();
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            acc = biases[ldB * out];
            asm volatile(
                "frep.o      %[n_frep], 1, 0, 0 \n"
                "fmadd.d     %[acc], ft0, ft1, %[acc] \n"
            : [ acc ] "+f"(acc)
            : [ n_frep ] "r"(IN_CH - 1)
            :"ft0", "ft1", "ft2");
        }

        activations[ldB * out] = acc;
        acc = 0.0;
        // End of SSR region.
        snrt_ssr_disable();

    }

    for (uint32_t out = 0; out < OUT_CH; out++) {
        printf("FEEDFORWARD FP64 with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    }
    
    snrt_cluster_hw_barrier(); 

}

//// Activation Step
void softmax_activation_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max, uint32_t* core_sync, uint32_t setup_SSR){

    double max_core;
    double sum = 0.0;

    max_core = activations[0];

    if (setup_SSR) {

        const uint32_t ssr0_b = OUT_CH;
        const uint32_t ssr0_i = sizeof(double);

        snrt_ssr_loop_1d(SNRT_SSR_DM0, ssr0_b, ssr0_i);

    }

    
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, activations);

    // Start of SSR region
    snrt_ssr_enable();
    for(uint32_t out = 0; out < OUT_CH; out++){
        asm volatile(
                    "fmv.d      fs2, ft0 \n"                // move the first value of the activations into fs2
                    "flt.d      t0, %[max_core], fs2\n"     // compare which value greater
                    "bnez       t0, 1f\n"                   // if the value was greater overwrite the old
                    "beqz       t0, 2f\n"                   // else go to loop start
                    "1: \n"     
                    "fmv.d      %[max_core], fs2 \n"
                    "2: \n"
                    : [ max_core ] "+f"(max_core)
                    :
                    :"ft0", "ft1", "ft2");
    }

    max[compute_id] = max_core;

    // End of the SSR region. 
    snrt_ssr_disable();

    snrt_cluster_hw_barrier();

    double max_global = max[0];

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
            printf("SOFTMAX FP64 with SSRs: activation[%u] = %f\n", out + 1, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
}

//// Gradient Update
void gradient_update_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, double *loss, uint32_t compute_num, uint32_t setup_SSR){

    // set up SSR registers (there is a total of three data movers)
    register volatile double ft0 asm("ft0"); // stores image
    register volatile double ft1 asm("ft1"); // stores weight gradients
    register volatile double ft2 asm("ft2"); // stores activations
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    double b_grad_update = 0.0;
    double W_grad_update = 0.0;
    double b_checksum = 0.0;
    double W_checksum = 0.0;
    volatile uint32_t idx_eff;

    // get the value saved at target address
    uint32_t target_n = *target;
    
    // compute the loss
    double loss_val = 0.0 - log(activations[target_n -compute_id]);

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
        
        // SSR setup of weight gradients
        snrt_ssr_loop_2d(SNRT_SSR_DM1, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW);


        // SSR setup of activations
        snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                        OUT_CH, 
                        sizeof(double));
    }

    // SSR start address need to be configured each time
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, weight_grads);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, activations);

    // Start of SSR region
    snrt_ssr_enable();

    // the effective index is the iteration index of the biases variable
    // across all entries
    for(uint32_t out = 0; out < OUT_CH; out++){
        // printf("bias grads[%u] = %f\n", compute_id + ldB * out, bias_grads[ldB * out]);
        // printf("biases[%u] = %f\n", compute_id + ldB * out, biases[ldB * out]);
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        b_checksum += b_grad_update;

        for(uint32_t in = 0; in < IN_CH; in++){
            
            W_grad_update = b_grad_update * image[in];
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH * (OUT_CH * 5 - 1))){
                weight_grads[out * ldW + in] += W_grad_update;
                W_checksum += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update; // INFO: "+" only for debugging to check if bias_grads zero initialized!!
    }

    // End of the SSR region. 
    snrt_ssr_disable();

    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("GRADIENT UPDATE FP64 with SSRs: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    // }
    printf("GRADIENT UPDATE FP64 with SSRs: b_checksum = %f\n", b_checksum);
    printf("GRADIENT UPDATE FP64 with SSRs: W_checksum = %f\n", W_checksum);

    snrt_cluster_hw_barrier();

}

//// Training Step
void training_step_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR){
    
    // set up SSR registers (there is a total of three data movers)
    register volatile double ft0 asm("ft0"); // stores weight gradients
    register volatile double ft1 asm("ft1"); // stores bias gradients
    register volatile double ft2 asm("ft2"); // stores weights
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    // FIXME: learning rate should be defined in network struct
    double lr = 0.5;

    double b_checksum = 0.0;
    double W_checksum = 0.0;

    double nimg = ((double)number_of_images);

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    // SSR strides and bounds only have to be configured
    // once in the beginning
    if (setup_SSR) {
        
        // SSR setup of weight gradients
        snrt_ssr_loop_2d(SNRT_SSR_DM0, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW);


        // SSR setup of bias gradients
        snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                        OUT_CH, 
                        sizeof(double));

        // SSR setup of weights
        snrt_ssr_loop_2d(SNRT_SSR_DM2, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW);
    }

    // Start of SSR region
    snrt_ssr_enable();

    for(uint32_t out = 0; out < OUT_CH; out++){
        // SSR start address need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &weight_grads[out*ldW]);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, &weights[out*ldW]);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &bias_grads[out*ldB]);
        // collect the bias gradients in a reg
        register double acc_b = bias_grads[ldB * out];
        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            asm volatile(
                "fmul.d        %[acc_b], %[lr], ft1 \n"             // acc = lr * bias_grads[ldB * out]
                "fdiv.d        %[acc_b], %[acc_b], %[nimg] \n"      // acc = acc / nimg
                //"fsub.d        %[acc_b], ft2, %[acc_b] \n"        // acc = biases[ldB * out] - acc
            :[ acc_b ] "+f"(acc_b), [ nimg ] "+f"(nimg), [ lr ] "+f"(lr)
            :
            :"ft1"
            );
            biases[ldB * out] -= acc_b;
            // biases[ldB * out] -= lr * bias_grads[ldB * out] / ((double) number_of_images); reference
        } else {
            biases[ldB * out] = 0;
        }

        b_checksum += biases[ldB * out];

        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            if(!(compute_id*IN_CH1*IN_CH2 + out * ldW + in > IN_CH1*IN_CH2 * OUT_CH * 5)){
                register double acc_w = weight_grads[out * ldW + in];
                asm volatile(
                    "fmul.d        %[acc_w], %[lr], ft0 \n"             // acc = lr * weight_grads[out * ldW + in]
                    "fdiv.d        %[acc_w], %[acc_w], %[nimg] \n"      // acc = acc / nimg
                    "fsub.d        %[acc_w], %[acc_w], ft2 \n"          // acc = acc - weights[out * ldW + in]
                    :[ acc_w ] "+f"(acc_w), [ nimg ] "+f"(nimg), [ lr ] "+f"(lr)
                    :
                    :"ft0", "ft2"
                );

                weights[out * ldW + in] = acc_w;
                W_checksum += weights[out * ldW + in];
                //weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images); reference
            } 
        }
    }

    // End of the SSR region. 
    snrt_ssr_disable();

    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("TRAINING STEP FP64 with SSRs: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    // }

    printf("TRAINING STEP FP64 with SSRs: b_checksum = %f\n", b_checksum);
    printf("TRAINING STEP FP64 with SSRs: W_checksum = %f\n", W_checksum);
}

// INFO: start of FP32 network implementation using SSRs and SIMD instructions
//// Feedforward Step
void feedforward_fp32_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *biases, float *activations,
                uint32_t ldB, float *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync,
                uint32_t setup_SSR){
    
    register float acc = 0.0;
    register float acc2 = 0.0;
    register float z_test = 0;

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
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        register v2f32 reduce_reg;
        register v2f32 sum;
        // Start of SSR region
        snrt_ssr_enable();
        const register float zero = 0;
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            acc = biases[ldB * out];
            // INFO: The zero reg causes Out of Memory accesses - WTF? Discuss with GIM --> Issue was missing ft2 clobber (reserved for SSR)
            asm volatile(
                "vfcpka.s.s     %[reduce_reg], %[acc], %[zero] \n"
                "frep.o         %[n_frep], 1, 0, 0 \n"
                "vfmac.s        %[reduce_reg], ft0, ft1 \n"                     // load two values from image and weights into SIMD vector
                : [ reduce_reg ] "+&f"(reduce_reg)
                : [ zero ] "f"(zero), [ acc ] "f"(acc), [ n_frep ] "r"(IN_CH / 2 - 1)
                : "ft0", "ft1", "ft2");
        }

        asm volatile(
                "vfcpka.s.s        %[sum], %[zero], %[zero] \n"
                "vfsum.s           %[sum], %[reduce_reg] \n"
                "vfcpka.s.s        %[acc], %[sum], %[zero] \n"
                : [ acc ] "+f"(acc), [ sum ] "=&f"(sum)
                : [ zero ] "f"(zero),  [ reduce_reg ] "f"(reduce_reg)
                :"ft0", "ft1", "ft2");

        // Q: Why do we need to do this?
        // snrt_cluster_hw_barrier();

        activations[ldB * out] = acc;
        acc = 0.0;

        // End of SSR region.
        snrt_ssr_disable();

    }

    for (uint32_t out = 0; out < OUT_CH; out++) {
        // cleanup of leftover columns
        if(compute_id + out * ldB > OUT_CH * 5 - 1){
            activations[ldB * out] = 0;
        }

        printf("FEEDFORWARD FP32 SIMD with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    }

    snrt_cluster_hw_barrier();

}

//// Activation Step
void softmax_activation_fp32(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *activations, uint32_t ldB,
                float *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, float *max, uint32_t* core_sync){

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

    //printf("Max value of compute core %u is %f\n", compute_id, max_core);

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

    //core_sync[compute_id] = 0;
    snrt_cluster_hw_barrier();
}

//// Gradient Update
void gradient_update_fp32_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weight_grads, uint32_t ldW, float *bias_grads, float *activations, 
                uint32_t ldB, float *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, float *loss, uint32_t compute_num, uint32_t setup_SSR){

    // set up SSR registers (there is a total of three data movers)
    register volatile double ft0 asm("ft0"); // stores image
    register volatile double ft1 asm("ft1"); // stores weight gradients
   
    asm volatile("" : "=f"(ft0), "=f"(ft1));

    float b_grad_update = 0.0;
    float W_grad_update = 0.0;
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

    // Start of SSR region
    snrt_ssr_enable();

    for(uint32_t out = 0; out < OUT_CH; out++){
        // SSR start address need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weight_grads[out*ldW]);
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        register v2f32 reduce_reg;
        snrt_cluster_hw_barrier();
        for(uint32_t in = 0; in < IN_CH;){
            asm volatile(
                "vfcpka.s.s         %[reduce_reg], %[b_grad], %[b_grad] \n"       // load the bias gradient for each vector
                "vfmul.s            %[reduce_reg], %[reduce_reg], ft0 \n"         // compute weight update b_grad * image
                "vfadd.s            %[reduce_reg], %[reduce_reg], ft1 \n"         // add weight update to weight gradient
                : [reduce_reg] "+&f"(reduce_reg)
                : [b_grad] "f"(b_grad_update)
                : "ft0", "ft1"
            );

            reduce_reg_u.v2 = reduce_reg;

            weight_grads[out*ldW + in] = reduce_reg_u.v[0];
            weight_grads[out*ldW + in + 1] = reduce_reg_u.v[1];
            in += 2;
        }

        bias_grads[ldB * out] = b_grad_update;


    }

    snrt_ssr_disable();

    for(uint32_t out = 0; out < OUT_CH; out++){
        printf("GRADIENT UPDATE FP32 SIMD with SSRs: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    }

    snrt_cluster_hw_barrier();
}

//// Training Step
void training_step_fp32_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, float *weight_grads, uint32_t ldW, float *biases, float *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR){

    float lr = 0.5;

    float test; // test variable for debugging

    // convert number of images to float for vectorized computation
    float nimg = ((float)number_of_images);
    register v2f32 lr_vec;
    register v2f32 nimg_vec;
    union fp32_v2f32_u reduce_reg_u;
    // pack the learning rate and number of images into a vector for vectorized computation
    asm volatile(
        "vfcpka.s.s          %[lr_vec], %[lr], %[lr] \n"
        "vfcpka.s.s          %[nimg_vec], %[nimg], %[nimg] \n"
        : [lr_vec] "+&f"(lr_vec), [nimg_vec] "+&f"(nimg_vec)
        : [lr] "f"(-lr), [nimg] "f"(nimg)
        : "ft0", "ft1"
    );

    register volatile float ft0 asm("ft0"); // stores weight gradients
    register volatile float ft1 asm("ft1"); // stores weights

    asm volatile("" : "=f"(ft0), "=f"(ft1));

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
        
        snrt_ssr_enable();
        // SSR start addresses need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &weight_grads[out*ldW]);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        register v2f32 reduce_reg;
        snrt_cluster_hw_barrier();
        for(uint32_t in = 0; in < IN_CH;){
            
            if(!(compute_id*IN_CH + out * ldW + in + 1 > IN_CH * OUT_CH * 5)){
                //weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images); FP32 reference
                asm volatile(
                    "vfmul.s              %[reduce_reg], %[lr_vec], ft0 \n"         // compute the weight update
                    "vfdiv.s              %[reduce_reg], %[reduce_reg], %[nimg_vec] \n"     // divde by the size of the dataset --> banshee: add floating point exception for divide by zero
                : [reduce_reg] "+&f"(reduce_reg)
                : [lr_vec] "f"(lr_vec), [nimg_vec] "f"(nimg_vec)
                : "ft0", "ft1"  
                );

                reduce_reg_u.v2 = reduce_reg;

                weights[out*ldW + in] = reduce_reg_u.v[0];
                weights[out*ldW + in + 1] = reduce_reg_u.v[1];
            } 

            in += 2;

        }

        snrt_ssr_disable();

    }


    for(uint32_t out = 0; out < OUT_CH; out++){
        printf("FP32 with SSRs and SIMD: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    }

}

// INFO: start of FP32 baseline network implementation
void feedforward_fp32(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *biases, float *activations,
                uint32_t ldB, float *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync){
    
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

void gradient_update_fp32(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
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

void training_step_fp32(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
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
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images);
                W_checksum += weights[out * ldW + in];
            } 
        }
    }

    printf("TRAINING STEP FP32 Baseline: b_checksum = %f\n", b_checksum);
    printf("TRAINING STEP FP32 Baseline: W_checksum = %f\n", W_checksum);
}

// INFO: start of FP16 network implementation using SSRs and SIMD instructions
//// Feedforward Step
void feedforward_fp16_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *biases, __fp16 *activations,
                uint32_t ldB, __fp16 *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync,
                uint32_t setup_SSR){
    
    // INFO: for simplicity image is converted to dtype __fp16 --> discuss with GIM 
    register volatile float ft0 asm("ft0"); // stores image
    register volatile float ft1 asm("ft1"); // stores weights
    register volatile float ft2 asm("ft2"); // stores biases
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    const register float zero = 0.0f;

    __fp16 acc = 0.0f;

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

        // // setup of DATA MOVER for biases
        // snrt_ssr_loop_1d(SNRT_SSR_DM2,
        //                 OUT_CH,
        //                 sizeof(double));
    }

    // Start of SSR region
    snrt_ssr_enable();

    for (uint32_t out = 0; out < OUT_CH; out++) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        // snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, &biases[out*ldB]);
        register v2f32 reduce_reg;
        register v2f32 sum;
        register v2f32 test;
        register v4f16 dotp;
        register float tacc;

        acc = biases[ldB*out];
        asm volatile(
            "fadd.s    %[tacc], %[zero], %[zero]\n" // dummy instruction for debugging
            : [test] "+&f"(test), [tacc] "+&f"(tacc)
            : [zero] "f"(zero)
        );
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            for (uint32_t in = 0; in < IN_CH;) {
                
                // calculate the dot product of the image and the weights (increment by four columns in each iteration)
                asm volatile(
                    "vfcpka.s.s       %[dotp], %[zero], %[zero]\n"
                    "vfcpka.s.s       %[sum], %[zero], %[zero] \n"
                    "vfdotpex.s.h     %[dotp], ft1, ft0 \n"
                    "vfsum.s          %[sum], %[dotp] \n"
                    "fadd.s           %[tacc], %[tacc], %[sum] \n"
                : [sum] "+&f"(sum), [dotp] "+&f"(dotp), [tacc] "+&f"(tacc)
                : [zero] "f"(zero)
                : "ft0", "ft1", "ft2"
                );

                in += 4;
            }
            snrt_ssr_disable();
            //printf("DEBUG: tacc[%u] = %f\n", compute_id + out + 1, tacc);
            //printf("DEBUG: acc[%u] = %f\n", compute_id + out + 1, acc);
            printf("DEBUG: tacc[%u] + acc[%u] = %f\n", compute_id + out + 1, compute_id + out + 1, tacc + acc);
            snrt_ssr_enable();
            acc += tacc;
            snrt_ssr_disable();
            printf("DEBUG: acc_up[%u] = %f\n", compute_id + out + 1, acc);
            snrt_ssr_enable();

        }

        // Q: Why do we need to do this?
        //snrt_cluster_hw_barrier(); --> actually not needed, but execution has to be prolonged for some
        // strange reason --> Discuss with GIM

        activations[ldB * out] = acc;

    }

    // End of SSR region.
    snrt_ssr_disable();

    for (uint32_t out = 0; out < OUT_CH; out++) {
        // cleanup of leftover columns
        if(compute_id + out * ldB > OUT_CH * 5 - 1){
            activations[ldB * out] = 0;
        }

        printf("FP16 SIMD with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    }

    snrt_cluster_hw_barrier();

}

//// Gradient Update
void gradient_update_fp16_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weight_grads, uint32_t ldW, __fp16 *bias_grads, __fp16 *activations, 
                uint32_t ldB, __fp16 *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, __fp16 *loss, uint32_t compute_num, uint32_t setup_SSR){

    // set up SSR registers (there is a total of three data movers)
    register volatile double ft0 asm("ft0"); // stores image
    register volatile double ft1 asm("ft1"); // stores weight gradients
   
    asm volatile("" : "=f"(ft0), "=f"(ft1));

    float b_grad_update = 0.0f;
    __fp16 W_grad_update = 0.0f;
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
        printf("DEBUG: EFFECTIVE b_grad_update[%u] = %f\n", compute_id + out * ldB + 1, b_grad_update);
        // TODO: check the SSR enable/disable signals in each loop!!
        snrt_ssr_enable();
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
        : "ft0", "ft1"
        );
        for(uint32_t in = 0; in < IN_CH;){
            if(!(compute_id * IN_CH + out * ldW + in + 3 > IN_CH * (OUT_CH-1) * 5)){
                asm volatile(
                    "vfcpka.s.s       %[dotp], %[zero], %[zero]\n"
                    "vfcpka.s.s       %[sum], %[zero], %[zero]\n"
                    "vfmul.h          %[dotp], %[reduce_reg], ft0\n"
                    //"vfdotpex.s.h     %[dotp], %[reduce_reg], ft0\n"
                    //"vfsum.s          %[sum], %[dotp]\n"
                    //"fadd.s           %[acc], %[acc], %[sum]\n"

                : [reduce_reg] "+&f"(reduce_reg), [dotp] "+&f"(dotp), [acc] "+&f"(acc), [sum] "+&f"(sum)
                : [b_grad_update] "f"(b_grad_update), [zero] "f"(0.0f)
                : "ft0", "ft1"
                );

                    //printf("DEBUG: index = %u\n", compute_id * IN_CH + out * ldW + in + 4);
                    weight_grads[out*ldW + in + 0] += dotp[0];
                    weight_grads[out*ldW + in + 1] += dotp[1];
                    weight_grads[out*ldW + in + 2] += dotp[2];
                    weight_grads[out*ldW + in + 3] += dotp[3];
            }

            in += 4;
        }

        snrt_ssr_disable();
        if(compute_id + out * ldB + 1 < OUT_CH * 5){
            bias_grads[ldB * out] = b_grad_update;
            printf("GRADIENT UPDATE FP16 SIMD with SSRs: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, b_grad_update);
        }
        // TODO: check this in other implementations as well
        // bias_grads[ldB * out] = b_grad_update;
        // printf("GRADIENT UPDATE FP16 SIMD with SSRs: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, b_grad_update);

    }

    snrt_cluster_hw_barrier();

}


//// Training Step
void training_step_fp16_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, __fp16 *weight_grads, uint32_t ldW, __fp16 *biases, __fp16 *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR){

    float lr = 0.5;

    // convert number of images to float for vectorized computation
    float nimg = ((float)number_of_images);
    
    register v4f16 lr_vec;
    register v4f16 nimg_vec;

    // pack the learning rate and number of images into a vector for vectorized computation
    asm volatile(
        "vfcpka.h.s          %[lr_vec], %[lr], %[lr] \n"
        "vfcpkb.h.s          %[lr_vec], %[lr], %[lr] \n"
        "vfcpka.h.s          %[nimg_vec], %[nimg], %[nimg] \n"
        "vfcpka.b.s          %[nimg_vec], %[nimg], %[nimg] \n"
        : [lr_vec] "+&f"(lr_vec), [nimg_vec] "+&f"(nimg_vec)
        : [lr] "f"(-lr), [nimg] "f"(nimg)
        : "ft0", "ft1"
    );

    register volatile float ft0 asm("ft0"); // stores weight gradients
    register volatile float ft1 asm("ft1"); // stores weights

    asm volatile("" : "=f"(ft0), "=f"(ft1));

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
        
        snrt_ssr_enable();
        // SSR start addresses need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &weight_grads[out*ldW]);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        snrt_ssr_disable();
        register v4f16 reduce_reg;
        // snrt_cluster_hw_barrier();
        for(uint32_t in = 0; in < IN_CH;){
            snrt_ssr_enable();
            if(!(compute_id*IN_CH + out * ldW + in + 3 > IN_CH * (OUT_CH - 1) * 5)){
                //weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images); FP32 reference
                asm volatile(
                    "vfmul.s              %[reduce_reg], %[lr_vec], ft0 \n"                 // compute the weight update
                    "vfdiv.s              %[reduce_reg], %[reduce_reg], %[nimg_vec] \n"     // divde by the size of the dataset --> banshee: add floating point exception for divide by zero
                : [reduce_reg] "+&f"(reduce_reg)
                : [lr_vec] "f"(lr_vec), [nimg_vec] "f"(nimg_vec)
                : "ft0", "ft1"  
                );

                weights[out*ldW + in + 0] = reduce_reg[0];
                weights[out*ldW + in + 1] = reduce_reg[1];
                weights[out*ldW + in + 2] = reduce_reg[2];
                weights[out*ldW + in + 3] = reduce_reg[3];
            } 

            in += 4;
            snrt_ssr_disable();
        }


    }


    for(uint32_t out = 0; out < OUT_CH; out++){
        printf("FP16 with SSRs and SIMD: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    }

}

// INFO: start of FP16 baseline network implementation
//// Feedforward Step
void feedforward_fp16(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *biases, __fp16 *activations,
                uint32_t ldB, __fp16 *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync){

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
    //core_sync = 1;
    snrt_cluster_hw_barrier();

}

//// Activation Step
void softmax_activation_fp16(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *activations, uint32_t ldB,
                __fp16 *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, __fp16 *max, uint32_t* core_sync){

    __fp16 max_core;
    __fp16 sum = 0.0f;
    __fp16 *act_ptr;
        
    while(!(core_sync[0])){
        max_core = 0.0;
    }

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

void gradient_update_fp16(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weight_grads, uint32_t ldW, __fp16 *bias_grads, __fp16 *activations, 
                uint32_t ldB, __fp16 *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, __fp16 *loss, uint32_t compute_num){

    
    __fp16 b_grad_update = 0.0;
    __fp16 W_grad_update = 0.0;
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

        //printf("b_grad_update = %f\n", b_grad_update);

        for(uint32_t in = 0; in < IN_CH; in++){
            
            W_grad_update = b_grad_update * image[in];
            
            if(!(compute_id*IN_CH1*IN_CH2 + out * ldW + in > IN_CH1*IN_CH2 * OUT_CH * 5)){
                weight_grads[out * ldW + in] += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update;
        printf("GRADIENT UPDATE FP16 Baseline: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    }

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

}

void training_step_fp16(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, __fp16 *weight_grads, uint32_t ldW, __fp16 *biases, __fp16 *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images){

    __fp16 lr = 0.5;

    for(uint32_t out = 0; out < OUT_CH; out++){

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((__fp16) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            
            if(!(compute_id*IN_CH1*IN_CH2 + out * ldW + in > IN_CH1*IN_CH2 * OUT_CH * 5)){
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((__fp16) number_of_images);
            } 
        }
    }

    for(uint32_t out = 0; out < OUT_CH; out++){
        printf("FP16 baseline: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    }
}

// INFO: start of FP8 baseline network implementation
//// Feedforward Step
void feedforward_fp8(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *biases, char *activations,
                uint32_t ldB, char *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync){

    // Linear layer: OUT = X * W^T + B
    char test = 0.0;
    for (uint32_t out = 0; out < OUT_CH; out++) {
        char acc = biases[ldB * out];
        //printf("FP16 baseline init: acc[%u] = %f\n", 1 + compute_id + out * ldB, acc);
        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            acc += image[in] * weights[out * ldW + in];
            test = image[in] * weights[out * ldW + in];
            // INFO: If this is not set harts start reading outside the mem map
            // FIXME: Next harts should start computation of the subsequent image
            if(compute_id + out * ldB > OUT_CH * 5 - 1){
                acc = 0;
            }
        }
        // OUT is accumulated in activations 
        activations[ldB * out] = acc;
        //printf("FP16 Baseline: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
        //printf("Core %u done with the computation: core_sync[%u] = %u.\n", compute_id + 1, compute_id + 1, core_sync);   
    }
    core_sync = 1;

}

//// Activation Step
void softmax_activation_fp8(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *activations, uint32_t ldB,
                char *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, char *max, uint32_t* core_sync){}

//     char max_core;
//     char sum = 0.0;
        
//     while(!(core_sync[0])){
//         max_core = 0.0;
//     }

//     max_core = activations[0];

//     if(core_sync[compute_id]){

//         //core_sync[compute_id] = 0;

//         for(uint32_t out = 0; out < OUT_CH; out++){
//             if(activations[ldB * out] > max_core) {
//                 max_core = activations[ldB * out];
//             }
//         }

//         max[compute_id] = max_core;

//     }
    
//     snrt_cluster_hw_barrier();

//     //printf("Max value of compute core %u is %f\n", compute_id, max_core);

//     char max_global = max[0];

//     // Reduction on single core
//     if(compute_id == 0){
//         for(uint32_t core = 0; core < compute_num; core++){
//             if(max[core] > max_global){
//                 max_global = max[core];
//             }
//         }

//         // FIXME: actually OUT_CH should be multiplied by number of compute cores
//         for(uint32_t out = 0; out < OUT_CH*5; out++){
//             if(activations[out]){
//                 activations[out] = exp(activations[out] - max_global);
//                 sum += activations[out];
//             } else {
//                 activations[out] = 0.0;
//             }
//         }


//         for(uint32_t out = 0; out < OUT_CH*5; out++){
//             activations[out] /= sum;
//             printf("FP8 (no SIMD): activation[%u] = %f\n", out + 1, activations[out]);
//         }
//     }

//     //core_sync[compute_id] = 0;
//     snrt_cluster_hw_barrier();
// }

void gradient_update_fp8(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weight_grads, uint32_t ldW, char *bias_grads, char *activations, 
                uint32_t ldB, char *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, char *loss, uint32_t compute_num){

    
    char b_grad_update = 0.0;
    char W_grad_update = 0.0;
    volatile uint32_t idx_eff;


    // get the value saved at target address
    uint32_t target_n = *target;
    // compute the loss
    char loss_val = 0.0 - log(activations[target_n -compute_id]);

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

        //printf("b_grad_update = %f\n", b_grad_update);

        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            
            W_grad_update = b_grad_update * image[in];
            
            if(!(compute_id*IN_CH1*IN_CH2 + out * ldW + in > IN_CH1*IN_CH2 * OUT_CH * 5)){
                weight_grads[out * ldW + in] += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update;
        //printf("bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    }

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

}

void training_step_fp8(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, char *weight_grads, uint32_t ldW, char *biases, char *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images){

    char lr = 0.5;

    for(uint32_t out = 0; out < OUT_CH; out++){

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((char) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            
            if(!(compute_id*IN_CH1*IN_CH2 + out * ldW + in > IN_CH1*IN_CH2 * OUT_CH * 5)){
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((char) number_of_images);
            } 
        }
    }

    for(uint32_t out = 0; out < OUT_CH; out++){
        printf("FP16 baseline: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    }
}