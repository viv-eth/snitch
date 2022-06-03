// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_network.h"

#include "printf.h"
#include "snrt.h"

// INFO: start of FP64 baseline network implementation

// The output of the feedforward is accumulated in the biases variable
void feedforward_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id){

    
    // Linear layer: OUT = X * W^T + B
    for (uint32_t out = 0; out < OUT_CH; out++) {
        //printf("Step: %u\n", out + compute_id);
        register double acc = biases[ldB * out];
        for(uint32_t in = 0; in < IN_CH1*IN_CH2; in++){
            acc += image[in] * weights[out * ldW + in];
            // TODO: for some reason the weights evaluate to -inf
            // when they should be zero --> fix the bug
            // INFO: If this is not set harts start reading outside the mem map
            if(compute_id + out * ldB > OUT_CH * 5){
                acc = 0;
            }
        }
        // OUT is accumulated in activations 
        activations[ldB * out] = acc;
        //printf("acc[%u] = %f\n", compute_id + out * ldB, activations[ldB * out]);   
    }
    snrt_cluster_hw_barrier();

} // WORKS on Cluster 0

void softmax_activation_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max){

    
    double max_core = activations[0];

    double sum = 0.0;
    
    for(uint32_t out = 0; out < OUT_CH; out++){
        if(activations[ldB * out] > max_core) {
            max_core = activations[ldB * out];
        }
    }

    max[compute_id] = max_core;
    snrt_cluster_hw_barrier();

    //printf("Max value of compute core %u is %f\n", compute_id, max_core);

    double max_global = max[0];

    // Reduction on single core
    if(compute_id == 0){
        for(uint32_t core; core < compute_num; core++){
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
            //printf("Cluster 0: Bias[%u] = %f\n", out, activations[out]);
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


    // get the value saved at target address
    uint32_t target_n = *target;
    // compute the loss
    double loss_val = 0.0 - log(activations[target_n -compute_id]);

    // save the value into the loss pointer
    if(!compute_id){
        loss[0] += loss_val;
    } else{
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