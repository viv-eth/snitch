// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_network.h"

#include "printf.h"
#include "snrt.h"

// INFO: start of FP64 baseline network implementation

// The output of the feedforward is accumulated in the activations variable
void feedforward_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync){

    
    // Linear layer: OUT = X * W^T + B
    for (uint32_t out = 0; out < OUT_CH; out++) {
        //printf("Step: %u\n", out + compute_id);
        register double acc = biases[ldB * out];
        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            acc += image[in] * weights[out * ldW + in];
            // INFO: If this is not set harts start reading outside the mem map
            // FIXME: Next harts should start computation of the subsequent image
            if(compute_id + out * ldB > OUT_CH * 5){
                acc = 0;
            }
        }
        // OUT is accumulated in activations 
        activations[ldB * out] = acc;
        printf("Baseline: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
        core_sync = 1;
        //printf("Core %u done with the computation: core_sync[%u] = %u.\n", compute_id + 1, compute_id + 1, core_sync);   
    }
    // when a cluster is done with its part of the FF it asserts the 
    // sync flag to indicate that the computation is done
    snrt_cluster_hw_barrier();

} // WORKS on Cluster 0

void softmax_activation_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max, uint32_t* core_sync){

    double max_core;
    double sum = 0.0;
        
    while(!(core_sync[0])){
        max_core = 0.0;
    }

    max_core = activations[0];
    
    if(core_sync[compute_id]){

        //core_sync[compute_id] = 0;

        for(uint32_t out = 0; out < OUT_CH; out++){
            if(activations[ldB * out] > max_core) {
                max_core = activations[ldB * out];
            }
        }

        max[compute_id] = max_core;
    
    }
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
            //printf("Cluster 0: activation[%u] = %f\n", out + 1, activations[out]);
        }
    }

    //core_sync[compute_id] = 0;
    snrt_cluster_hw_barrier();
} // WORKS on Cluster 0

void gradient_update_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, double *loss, uint32_t compute_num){

    
    double b_grad_update = 0.0;
    double W_grad_update = 0.0;
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

    
    //printf("loss = %f\n", loss[0]);
    

    // the effective index is the iteration index of the biases variable
    // across all entries
    for(uint32_t out = 0; out < OUT_CH; out++){
        // printf("bias grads[%u] = %f\n", compute_id + ldB * out, bias_grads[ldB * out]);
        // printf("biases[%u] = %f\n", compute_id + ldB * out, biases[ldB * out]);
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];

        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            
            W_grad_update = b_grad_update * image[in];
            
            if(!(compute_id*IN_CH1*IN_CH2 + out * ldW + in > IN_CH1*IN_CH2 * OUT_CH * 5)){
                weight_grads[out * ldW + in] += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update; // INFO: "+" only for debugging to check if bias_grads zero initialized!!
        //printf("bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    }

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

} // WORKS on Cluster 1

// TODO: implement training step for multiple images
void training_step_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images){

    float lr = 0.5;

    for(uint32_t out = 0; out < OUT_CH; out++){

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((float) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            
            if(!(compute_id*IN_CH1*IN_CH2 + out * ldW + in > IN_CH1*IN_CH2 * OUT_CH * 5)){
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images);
            } 
        }
    }
}

// INFO: start of FP64 network implementation using SSRs
//// Feedforward Step
void feedforward_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync,
                uint32_t setup_SSR){

    register volatile double ft0 asm("ft0"); // stores image
    register volatile double ft1 asm("ft1"); // stores weights
    asm volatile("" : "=f"(ft0), "=f"(ft1));

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
        

        snrt_ssr_loop_2d(SNRT_SSR_DM1, 
                        IN_CH, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW);
    }
    
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, weights);
    
    // Start of SSR region
    snrt_ssr_enable();

    for (uint32_t out = 0; out < OUT_CH; out++) {
        // we need to read the image for every new iteration
        // of a core, because otherwise it will evaluate to
        // all zeros due to the stream semantics
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        register double acc = biases[ldB * out];
        if(compute_id + out * ldB > OUT_CH * 5){
            acc = 0;
        } else {
            for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            asm volatile(
                "fmadd.d %[acc], ft0, ft1, %[acc] \n"
            : [ acc ] "+f"(acc)
            ::"ft0", "ft1");
            }
        }

        activations[ldB * out] = acc;

    }

    // End of SSR region.
    snrt_ssr_disable();

    // for (uint32_t out = 0; out < OUT_CH; out++) {
    //     printf("FP64 with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    // }
    
    core_sync = 1;
    //snrt_cluster_hw_barrier();

}

//// Activation Step
void softmax_activation_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max, uint32_t* core_sync, uint32_t setup_SSR){

    double max_core;
    double sum = 0.0;
        
    while(!(core_sync[0])){
        max_core = 0.0;
    }

    max_core = activations[0];
    
    register volatile double ft0 asm("ft0"); // stores activations
    asm volatile("" : "=f"(ft0));

    if (setup_SSR) {

        const uint32_t ssr0_b = OUT_CH;
        const uint32_t ssr0_i = sizeof(double);

        snrt_ssr_loop_1d(SNRT_SSR_DM0, ssr0_b, ssr0_i);

    }

    if(core_sync[compute_id]){
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
                        : [ max_core ] "+f"(max_core)::"ft0");
        }

        max[compute_id] = max_core;
    
    }
        

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
            //printf("FP64 with SSRs: activation[%u] = %f\n", out + 1, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
}

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

    // Unrolling factor of most inner loop.
    // Should be at least as high as the FMA delay
    // for maximum utilization
    const uint32_t unroll = 8;

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

        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            
            W_grad_update = b_grad_update * image[in];
            
            if(!(compute_id*IN_CH1*IN_CH2 + out * ldW + in > IN_CH1*IN_CH2 * OUT_CH * 5)){
                weight_grads[out * ldW + in] += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update; // INFO: "+" only for debugging to check if bias_grads zero initialized!!
    }

    // End of the SSR region. 
    snrt_ssr_disable();

    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("FP64 with SSRs: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    // }

    snrt_cluster_hw_barrier();

}