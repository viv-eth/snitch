// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_fp64_network.h"

#include "printf.h"
#include "snrt.h"
#include "math.h"

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

double my_fabs(double x) {
    if(x < 0) {
        return -x;
    } else {
        return x;
    }
}

double my_exp(double x) 
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

// The output of the feedforward is accumulated in the activations variable
void feedforward_fp64n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id){

    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    volatile uint32_t idx_eff;

    
    // Linear layer: OUT = X * W^T + B
    for (uint32_t out = 0; out < OUT_CH; out++) {
        //printf("Step: %u\n", out + compute_id);
        register double acc = biases[ldB * out];
        idx_eff = compute_id + ldB * out;
        for(uint32_t in = 0; in < IN_CH; in++){
            // acc += image[in] * weights[out * ldW + in];
            // INFO: If this is not set harts start reading outside the mem map
            // FIXME: Next harts should start computation of the subsequent image
            if(!(idx_eff > OUT_CH * 5 - 1)){
                acc += image[in] * weights[out * ldW + in];
            } else {
                acc = 0;
            }
        }
        // OUT is accumulated in activations 
        activations[ldB * out] = acc;
        // printf("new FEEDFORWARD FP64 Baseline: acc[%u] = %f\n", 1 + idx_eff, activations[ldB * out]);  
    }

    snrt_cluster_hw_barrier();

} // WORKS on Cluster 0

void softmax_activation_fp64n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max){

    double max_core;
    double sum = 0.0;

    double euler_constant = 2.7182818284590452353602874713527;


    volatile uint32_t idx_eff;

    max_core = activations[0];

    for(uint32_t out = 0; out < OUT_CH; out++){
        idx_eff = compute_id + ldB * out;
        if(!(idx_eff > OUT_CH * 5 - 1)){
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
                // activations[out] = exp(activations[out] - max_global); //INFO: changing this to not use EXP, since exponential seem to fail in the RTL
                // activations[out] = activations[out] - max_global; // this works in the RTL
                // INFO: pow() is slower than exp() due to heavier error checking mechanism and less optimization for exponentiation
                // activations[out] = pow(euler_constant, activations[out] - max_global); // TODO: test this in the RTL
                activations[out] = my_exp(activations[out] - max_global); // TODO: test this in the RTL
                sum += activations[out];
            } else {
                activations[out] = 0.0;
            }
        }

        // printf("sum = %f\n", sum);


        for(uint32_t out = 0; out < OUT_CH * 5; out++){
            activations[out] /= sum;
            // printf("new SOFTMAX FP64 Baseline: activation[%u] = %f\n", out + 1, activations[out]);
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
    volatile uint32_t W_idx_eff;

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
            
            W_idx_eff = compute_id*IN_CH + out * ldW + in;

            W_grad_update = b_grad_update * image[in];
            
            if(!(W_idx_eff > IN_CH * OUT_CH * 5 - 1)){
                weight_grads[out * ldW + in] += W_grad_update;
                W_checksum += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update; // INFO: "+" only for debugging to check if bias_grads zero initialized!!
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
    volatile uint32_t idx_eff;
    volatile uint32_t W_idx_eff;

    for(uint32_t out = 0; out < OUT_CH; out++){

        idx_eff = compute_id + ldB * out;

        // printf("TRAINING STEP FP64 baseline: old biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
        // printf("TRAINING STEP FP64 baseline: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(idx_eff > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((double) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        b_checksum += biases[ldB * out];

        for(uint32_t in = 0; in < IN_CH; in++){

            W_idx_eff = compute_id*IN_CH + out * ldW + in;
            
            if(!(W_idx_eff > IN_CH * OUT_CH * 5 - 1)){
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
void feedforward_fp64_ssrn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id,
                uint32_t setup_SSR){

    register double acc = 0.0;

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    volatile uint32_t idx_eff;
    volatile uint32_t W_idx_eff;

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

        idx_eff = compute_id + ldB * out;
        // we need to read the image for every new iteration
        // of a core, because otherwise it will evaluate to
        // all zeros due to the stream semantics
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        // Start of SSR region
        snrt_ssr_enable();
        if(!(idx_eff > OUT_CH * 5 - 1)){
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
        // printf("new FEEDFORWARD FP64 with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    }
    
    snrt_cluster_hw_barrier(); 

}

//// Activation Step
void softmax_activation_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max, uint32_t setup_SSR){

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
            printf("new SOFTMAX FP64 with SSRs: activation[%u] = %f\n", out + 1, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
}

//// Gradient Update
void gradient_update_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, double *loss, uint32_t compute_num, uint32_t setup_SSR){

    
    register double b_grad_update = 0.0;
    register double W_grad_update = 0.0;
    double b_checksum = 0.0;
    double W_checksum = 0.0;
    volatile uint32_t idx_eff;
    volatile uint32_t W_idx_eff;

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


        // SSR setup of activations
        snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                        OUT_CH, 
                        sizeof(double));
    }

    // SSR start address need to be configured each time

    // the effective index is the iteration index of the biases variable
    // across all entries
    for(uint32_t out = 0; out < OUT_CH; out++){

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image); // image stored in ft0
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, &activations[ldB * out]); // stored in ft2
        // printf("bias grads[%u] = %f\n", compute_id + ldB * out, bias_grads[ldB * out]);
        // printf("biases[%u] = %f\n", compute_id + ldB * out, biases[ldB * out]);
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        // b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        const register double one = 1;
        const register double zero = 0;
        // Start of SSR region
        snrt_ssr_enable();
        if(idx_eff > OUT_CH * 5 - 1){
            b_grad_update = 0.0;
        } else {
            asm volatile(
                        "beq          %[idx_eff], %[target_n], 1f\n"
                        "bne          %[idx_eff], %[target_n], 2f\n"
                        "1: \n"
                        "fsub.d       %[b_grad_update], ft2, %[one]\n"
                        "j            3f\n"
                        "2: \n"
                        "fadd.d       %[b_grad_update], ft2, %[zero]\n"
                        "3: \n"
                        : [ b_grad_update ] "+f"(b_grad_update)
                        : [ one ] "f"(one), [ idx_eff ] "r"(idx_eff), [ target_n ] "r"(target_n), 
                        [ zero ] "f"(zero)
                        : "ft0", "ft1", "ft2"
            );
        }
        b_checksum += b_grad_update;


        for(uint32_t in = 0; in < IN_CH; in++){

            W_idx_eff = compute_id*IN_CH + out * ldW + in;
            
            asm volatile(
                        "fmul.d         %[W_grad_update], %[b_grad_update], ft0\n"
                        : [ W_grad_update ] "+f"(W_grad_update)
                        : [ b_grad_update ] "f"(b_grad_update), [ zero ] "f"(zero)
                        : "ft0", "ft1", "ft2"
            );
            // W_grad_update = b_grad_update * image[in];
            
            if(!(W_idx_eff > IN_CH * OUT_CH * 5 - 1)){
                weight_grads[out * ldW + in] += W_grad_update;
                W_checksum += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update; // INFO: "+" only for debugging to check if bias_grads zero initialized!!
        
        // End of the SSR region. 
        snrt_ssr_disable();
    }

    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("GRADIENT UPDATE FP64 with SSRs: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, b_grad_update);
    // }
    printf("new GRADIENT UPDATE FP64 with SSRs: b_checksum = %f\n", b_checksum);
    printf("new GRADIENT UPDATE FP64 with SSRs: W_checksum = %f\n", W_checksum);

    snrt_cluster_hw_barrier();

}

//// Training Step
void training_step_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR){
    
    // FIXME: learning rate should be defined in network struct
    double lr = 0.5;

    double b_checksum = 0.0;
    double W_checksum = 0.0;

    double nimg = ((double)number_of_images);

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    volatile uint32_t idx_eff;
    volatile uint32_t W_idx_eff;

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

        idx_eff = compute_id + ldB * out;
        // SSR start address need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &weight_grads[out*ldW]);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, &weights[out*ldW]);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &bias_grads[out*ldB]);
        // collect the bias gradients in a reg
        register double acc_b = bias_grads[ldB * out];
        // make sure that biases outside of the number of
        // output channels are zero
        if(!(idx_eff > OUT_CH * 5 - 1)){
            asm volatile(
                "fmul.d        %[acc_b], %[lr], ft1 \n"             // acc = lr * bias_grads[ldB * out]
                "fdiv.d        %[acc_b], %[acc_b], %[nimg] \n"      // acc = acc / nimg
                //"fsub.d        %[acc_b], ft2, %[acc_b] \n"        // acc = biases[ldB * out] - acc
                :[ acc_b ] "+f"(acc_b), [ nimg ] "+f"(nimg), [ lr ] "+f"(lr)
                :
                :"ft0", "ft1", "ft2"
                );
            biases[ldB * out] -= acc_b;
            // biases[ldB * out] -= lr * bias_grads[ldB * out] / ((double) number_of_images); reference
        } else {
            biases[ldB * out] = 0;
        }

        b_checksum += biases[ldB * out];

        for(uint32_t in = 0; in < IN_CH; in++){
            
            W_idx_eff = compute_id*IN_CH + out * ldW + in;

            if(!(W_idx_eff > IN_CH * OUT_CH * 5 - 1)){
                register double acc_w = weight_grads[out * ldW + in];
                asm volatile(
                    "fmul.d        %[acc_w], %[lr], ft0 \n"             // acc = lr * weight_grads[out * ldW + in]
                    // "fdiv.d        %[acc_w], %[acc_w], %[nimg] \n"      // acc = acc / nimg
                    // "fsub.d        %[acc_w], %[acc_w], ft2 \n"          // acc = acc - weights[out * ldW + in]
                    :[ acc_w ] "+f"(acc_w), [ nimg ] "+f"(nimg), [ lr ] "+f"(lr)
                    :
                    :"ft0", "ft1", "ft2"
                );

                snrt_ssr_disable();
                weights[out * ldW + in] -= acc_w / nimg;
                W_checksum += weights[out * ldW + in];
                snrt_ssr_enable();
                //weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images); reference
            } 
        }
    }

    // End of the SSR region. 
    snrt_ssr_disable();

    // for(uint32_t out = 0; out < OUT_CH; out++){
    //     printf("TRAINING STEP FP64 with SSRs: updated biases[%u] = %f\n", 1 + compute_id + out * ldB, biases[ldB * out]);
    // }

    printf("new TRAINING STEP FP64 with SSRs: b_checksum = %f\n", b_checksum);
    printf("new TRAINING STEP FP64 with SSRs: W_checksum = %f\n", W_checksum);
}