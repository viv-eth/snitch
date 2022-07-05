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

float my_fabs(float x) {
    if(x < 0) {
        return -x;
    } else {
        return x;
    }
}

float my_exp(float x) 
{ 
    const float epsilon = 1e-7; 
    float sum = 0.0; 
    int n = 0; 
    float factorial = 1; 
    float power=1.0; 
    float term; 
    do { 
        term = power/factorial; 
        sum += term; 
        n += 1; 
        power *= x; 
        factorial *=n; 
    } while (my_fabs(term)>=epsilon); 
    return sum; 
} 

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
        // printf("FEEDFORWARD FP32 Baseline: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);  
    }

    snrt_cluster_hw_barrier();

} // RTL PASS

//// Activation Step
void softmax_activation_fp32n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *activations, uint32_t ldB,
                float *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, float *max){

    float max_core = 0.0;
    float sum = 0.0;
    float max_global;

    volatile uint32_t idx_eff;

    // snrt_ssr_enable();

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

    // printf("FEEDFORWARD FP32 Baseline: max[%u] = %f\n", compute_id, max[compute_id]);
    
    snrt_cluster_hw_barrier();

    max_global = max[0];

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
                // activations[out] = exp(activations[out] - max_global);
                activations[out] = my_exp(activations[out] - max_global);
                sum += activations[out];
            } else {
                activations[out] = 0.0;
            }
        }


        for(uint32_t out = 0; out < OUT_CH*5; out++){
            activations[out] /= sum;
            // printf("SOFTMAX FP32 (no SIMD): activation[%u] = %f\n", out + 1, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
} // RTL PASS

void gradient_update_fp32n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weight_grads, uint32_t ldW, float *bias_grads, float *activations, 
                uint32_t ldB, float *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, float *loss, uint32_t compute_num){

    
    float b_grad_update = 0.0;
    float W_grad_update = 0.0;
    // float b_checksum = 0.0;
    // float W_checksum = 0.0;
    volatile uint32_t idx_eff;
    volatile uint32_t idx_eff_W;

    const uint32_t IN_CH = IN_CH1 * IN_CH2;


    // get the value saved at target address
    uint32_t target_n = *target;
    // compute the loss
    // TODO: change this to my_log instead of log
    // float loss_val = 0.0 - log(activations[target_n -compute_id]);

    // save the value into the loss pointer
    // TODO: update this

    // the effective index is the iteration index of the biases variable
    // across all entries
    for(uint32_t out = 0; out < OUT_CH; out++){
        // printf("bias grads[%u] = %f\n", compute_id + ldB * out, bias_grads[ldB * out]);
        // printf("biases[%u] = %f\n", compute_id + ldB * out, biases[ldB * out]);
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        
        // b_checksum += b_grad_update;

        //printf("b_grad_update = %f\n", b_grad_update);

        for(uint32_t in = 0; in < IN_CH; in++){
            
            W_grad_update = b_grad_update * image[in];

            idx_eff_W = compute_id*IN_CH + out * ldW + in; // computed correctly
            
            if(!(idx_eff_W > IN_CH * OUT_CH * 5 - 1)){
                weight_grads[out * ldW + in] += W_grad_update;
                // W_checksum += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update;
    }

    // printf("GRADIENT UPDATE FP32 Baseline: b_checksum[%u] = %f\n", compute_id + 1, b_checksum);
    // printf("GRADIENT UPDATE FP32 Baseline: W_checksum[%u] = %f\n", compute_id + 1, W_checksum);

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

} // RTL PASS

void training_step_fp32n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, float *weight_grads, uint32_t ldW, float *biases, float *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images){

    float lr = 0.5;
    // float b_checksum = 0.0;
    // float W_checksum = 0.0;

    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    volatile uint32_t idx_eff_W;

    for(uint32_t out = 0; out < OUT_CH; out++){

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((float) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        // b_checksum += biases[ldB * out];

        for(uint32_t in = 0; in < IN_CH; in++){

            idx_eff_W = compute_id*IN_CH + out * ldW + in; 
            
            if(!(idx_eff_W > IN_CH * OUT_CH * 5 - 1)){
                // printf("DEBUG: weight grads[%u] = %f\n", compute_id*IN_CH + out * ldW + in, weight_grads[out*ldW + in]);
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((float) number_of_images);
                // W_checksum += weights[out * ldW + in];
            } 
        }
    }

    // printf("new TRAINING STEP FP32 Baseline: b_checksum = %f\n", b_checksum);
    // printf("new TRAINING STEP FP32 Baseline: W_checksum = %f\n", W_checksum);
} // RTL TODO

// INFO: start of FP32 network implementation using SSRs and SIMD instructions
//// Feedforward Step
void feedforward_fp32_ssr_simd_frep(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *biases, float *activations,
                uint32_t ldB, float *image, uint32_t ldI, uint32_t compute_id,
                uint32_t setup_SSR){

    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    volatile uint32_t idx_eff;

    // SSR strides and bounds only have to be configured
    // once in the beginning
    // WARN In the RTL SSR strides MUST BE of size DOUBLE

    if (setup_SSR) {

        // setup of DATA MOVER input data (MNIST image)
        snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                        IN_CH / 2, 
                        sizeof(double));
        
        // setup of DATA MOVER for weights
        snrt_ssr_loop_2d(SNRT_SSR_DM1, 
                        IN_CH / 2, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW / 2);
    }

    for (uint32_t out = 0; out < OUT_CH; out++) {
        idx_eff = compute_id + ldB * out;
        // we need to read the image for every new iteration
        // of a core, because otherwise it will evaluate to
        // all zeros due to the stream semantics
        // Start of SSR region
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
        snrt_ssr_enable();
        register float acc = 0.0;
        // we need to add a dummy register to fully consume the SSRs
        register float dummy = 0.0;
        register v2f32 reduce_reg;
        register v2f32 sum;
        const register float zero = 0;
        if(!(idx_eff > OUT_CH * 5 - 1)){
            acc = biases[ldB * out];
            // INFO: The zero reg causes Out of Memory accesses - WTF? Discuss with GIM --> Issue was missing ft2 clobber (reserved for SSR)
            asm volatile(
                "vfcpka.s.s        %[reduce_reg], %[acc], %[zero] \n"
                "frep.o            %[n_frep], 1, 0, 0 \n"
                "vfmac.s           %[reduce_reg], ft0, ft1 \n"                     // load two values from image and weights into SIMD vector
                "vfcpka.s.s        %[sum], %[zero], %[zero] \n"
                "vfsum.s           %[sum], %[reduce_reg] \n"
                "vfcpka.s.s        %[acc], %[sum], %[zero] \n"
                : [ reduce_reg ] "+&f"(reduce_reg), [ acc ] "+&f"(acc), [ sum ] "+&f"(sum)
                : [ zero ] "f"(zero), [ n_frep ] "r"(IN_CH / 2 - 1)
                : "ft0", "ft1", "ft2");

        } else {
            acc = 0.0;
            // GIM: which instructions have least overhead?
            asm volatile(
                "frep.o        %[n_frep], 1, 0, 0 \n"
                "vfadd.s       %[dummy], ft0, ft1 \n"
            : [ dummy ] "+f"(dummy)
            : [ n_frep ] "r"(IN_CH / 2 - 1)
            : "ft0", "ft1", "ft2");
        }

        activations[ldB * out] = acc;
        acc = 0.0;

        // End of SSR region.
        snrt_fpu_fence();
        snrt_ssr_disable();
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

    }

    // for (uint32_t out = 0; out < OUT_CH; out++) {
    //     printf("FEEDFORWARD FP32 SIMD with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    // }

    snrt_cluster_hw_barrier();

} // RTL PASS


//// Gradient Update
void gradient_update_fp32_ssr_simdn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weight_grads, uint32_t ldW, float *bias_grads, float *activations, 
                uint32_t ldB, float *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, float *loss, uint32_t compute_num, uint32_t setup_SSR){

    // INFO: due to a compiler bug we need to reserve the registers for the SSR
    //       otherwise it will use them for stack operations breaking the stream(s)
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    float b_grad_update = 0.0;
    float W_grad_update = 0.0;
    // float b_checksum = 0.0;
    // float loss_val = 0.0;
    // register float W_checksum = 0.0;
    volatile uint32_t idx_eff;
    volatile uint32_t idx_eff_W;

    // get the value saved at target address
    uint32_t target_n = *target;
    
    // compute the loss
    // if(!compute_id){
    //     loss_val = 0.0 - log(activations[target_n - compute_id]);
    // }

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;


    // SSR strides and bounds only have to be configured
    // once in the beginning
    if (setup_SSR) {

        // SSR read setup of input data (MNIST image)
        snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                        IN_CH / 2, 
                        sizeof(double));
        
        // SSR read setup of weight gradients 
        snrt_ssr_loop_2d(SNRT_SSR_DM1, 
                        IN_CH / 2, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW / 2);

        // SSR write setup of weight gradients
        snrt_ssr_loop_2d(SNRT_SSR_DM2, 
                        IN_CH / 2, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW / 2);



    }


    for(uint32_t out = 0; out < OUT_CH; out++){

        b_grad_update = 0.0;
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        // INFO: we need to add this, since after DMA transfer out of bound activations
        //       evaluate to -INF
        if(idx_eff  > OUT_CH * 5 - 1){
            b_grad_update = 0.0;
        } else {
            b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];
        }

        // b_checksum += b_grad_update;
        // SSR start address need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, image);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weight_grads[out*ldW]);
        // TODO: add 3rd stream with write
        // GIM: the SSR write fails
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_2D, &weight_grads[out*ldW]);
        snrt_ssr_enable();  
        register v2f32 reduce_reg;
        register float sum = 0.0;
        // dummy register in order to fully consume the SSRs
        register float dummy = 0.0;
        register const float zero = 0;
        register v2f32 zero_reg;
        // zero initialze the reduce register for each loop
        asm volatile(
                "vfcpka.s.s      %[reduce_reg], %[zero], %[zero] \n"
                "vfcpka.s.s      %[zero_reg], %[zero], %[zero] \n"
                : [ reduce_reg ] "+&f"(reduce_reg), [ zero_reg ] "+&f"(zero_reg)
                : [ zero ] "f"(zero)
                : "ft0", "ft1", "ft2"
        ); 
        for(uint32_t in = 0; in < IN_CH;){
            
            // calculate the effective index at which we are writing the weights
            // idx_eff_W = <compute core offset> + <out offset> + <in offset>
            idx_eff_W = compute_id*IN_CH + out * ldW + in;
            // add plus one to the effective index, since we are writing two values at once
            // GIM: discuss how to incorporate if statement into frep loop
            if(!(idx_eff_W > IN_CH * OUT_CH * 5 - 1)){
                // TODO: the second vfadd is actually not needed, but added in order to be able
                // to compute the checksum of the weights ---> remove it for benchmarking
                asm volatile(
                    //"frep.o             %[n_frep], 3, 0, 0 \n"
                    "vfcpka.s.s         %[reduce_reg], %[b_grad], %[b_grad] \n"       // load the bias gradient for each vector
                    "vfmul.s            %[reduce_reg], %[reduce_reg], ft0 \n"         // compute weight update b_grad * image
                    "vfadd.s            %[reduce_reg], %[reduce_reg], ft1 \n"         // add weight update to weight gradient
                    "vfadd.s            ft2, %[reduce_reg], %[zero_reg] \n"           // write the values into the weight gradients
                    : [reduce_reg] "+&f"(reduce_reg), [zero_reg] "+&f"(zero_reg)
                    : [b_grad] "f"(b_grad_update), [ n_frep ] "r"(IN_CH / 2 - 1), [zero] "f"(zero)
                    : "ft0", "ft1", "ft2"
                );


                // INFO: need to disable SSRs, as they do not have access to 
                //       other FP registers
                // snrt_ssr_disable();
                // W_checksum += reduce_reg[0] + reduce_reg[1]; // this works
                // snrt_ssr_enable();
                
            } else {
                asm volatile(
                    "vfadd.s       %[dummy], ft0, ft1 \n"
                    "vfadd.s       %[dummy], ft2, %[zero_reg] \n"
                    // "vfadd.s       ft2, %[zero_reg], %[zero_reg]\n" // GIM: why does this screw up the checksum?
                    : [reduce_reg] "+&f"(reduce_reg), [dummy] "+&f"(dummy), [zero_reg] "+&f"(zero_reg)
                    : [zero] "f"(zero)
                    : "ft0", "ft1", "ft2"
                );
            }

            in += 2;
        }

        bias_grads[ldB * out] = b_grad_update;
        // NOTE: potentially remove FPU fence
        snrt_fpu_fence();
        snrt_ssr_disable();
        // INFO: after disabling the SSRs we can free the registers
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

    }


    // printf("GRADIENT UPDATE FP32 SIMD with SSRs: b_checksum[%u] = %f\n", 1 + compute_id, b_checksum);
    // printf("GRADIENT UPDATE FP32 SIMD with SSRs: W_checksum[%u] = %f\n", 1 + compute_id, W_checksum);

    snrt_cluster_hw_barrier();
} // RTL PASS

//// Training Step
// FIXME: not giving correct weight checksum compared to baseline 
void training_step_fp32_ssr_simdn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, float *weight_grads, uint32_t ldW, float *biases, float *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR){

    // INFO: due to a compiler bug we need to reserve the registers for the SSR
    //       otherwise it will use them for stack operations breaking the stream(s)
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    float lr = 0.5;

    // float b_checksum = 0.0;
    // float W_checksum = 0.0;

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

    uint32_t volatile idx_eff_W;

    // SSR strides and bounds only have to be configured
    // once in the beginning
    // WARN In the RTL SSR strides MUST BE of size DOUBLE

    // SSR strides and bounds only have to be configured
    // once in the beginning
    if (setup_SSR) {
        
        // SSR read setup of weight gradients
        snrt_ssr_loop_2d(SNRT_SSR_DM0, 
                        IN_CH / 2, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW / 2);

        // // SSR read setup of weights
        // snrt_ssr_loop_2d(SNRT_SSR_DM1, 
        //                 IN_CH / 2, 
        //                 OUT_CH, 
        //                 sizeof(double), 
        //                 sizeof(double) * ldW / 2);

        // SSR write setup of weights
        snrt_ssr_loop_2d(SNRT_SSR_DM2, 
                        IN_CH / 2, 
                        OUT_CH, 
                        sizeof(double), 
                        sizeof(double) * ldW / 2);
    }


    for(uint32_t out = 0; out < OUT_CH; out++){
        // NOTE: we don't use SSRs for biases, as we don't have enough data movers
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((float) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        // b_checksum += biases[ldB * out];
        
        // SSR start addresses need to be configured each time
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &weight_grads[out*ldW]); // weight gradients stored in ft0
        // snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]); // weights stored in ft1
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_2D, &weights[out*ldW]);

        snrt_ssr_enable();
        register v2f32 reduce_reg;
        const register float zero = 0.0;
        register v2f32 zero_reg;
        // dummy register to consume SSRs 
        register float dummy = 0.0;
        // snrt_cluster_hw_barrier();
        for(uint32_t in = 0; in < IN_CH;){

            idx_eff_W = compute_id*IN_CH + out * ldW + in;
            
            if(!(idx_eff_W  > IN_CH * OUT_CH * 5)){
                // snrt_ssr_disable();
                // printf("DEBUG: weight_grads[%u] = %f\n", idx_eff_W, weight_grads[out * ldW + in]);
                // printf("DEBUG: weight_grads[%u] = %f\n", idx_eff_W, weight_grads[out * ldW + in + 1]);
                // snrt_ssr_enable();
                asm volatile(
                    "vfmul.s              %[reduce_reg], %[lr_vec], ft0 \n"                 // compute the weight update
                    "vfdiv.s              %[reduce_reg], %[reduce_reg], %[nimg_vec] \n"     // divde by the size of the dataset --> TODO: banshee: add floating point exception for divide by zero
                    "vfadd.s              ft2, %[reduce_reg], %[zero] \n"                   // write the value into the weights
                : [reduce_reg] "+&f"(reduce_reg), [zero_reg] "+&f"(zero_reg)
                : [lr_vec] "f"(lr_vec), [nimg_vec] "f"(nimg_vec), [zero] "f"(zero)
                : "ft0", "ft1", "ft2"
                ); 

                // discuss with GIM: can I FREP this somehow?
                // snrt_ssr_disable(); // Discuss with GIM: why do we need to disable SSRs?
                // weights[out*ldW + in] -= reduce_reg[0];
                // weights[out*ldW + in + 1] -= reduce_reg[1];
                // W_checksum += reduce_reg[0] + reduce_reg[1];
                // snrt_ssr_enable();
            } else {
                // TODO: add dummy instruction to consume SSRs
                asm volatile(
                    "vfadd.s              %[dummy], ft0, ft2 \n"
                : [dummy] "+&f"(dummy)
                :
                : "ft0", "ft1", "ft2"
                );
            }

            in += 2;

        }

        snrt_ssr_disable();
        // INFO: after disabling the SSRs we can free the registers
        asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

    }

    // printf("FP32 with SSRs and SIMD: b_checksum = %f\n", b_checksum);
    // printf("FP32 with SSRs and SIMD: W_checksum = %f\n", W_checksum);

} // RTL TODO