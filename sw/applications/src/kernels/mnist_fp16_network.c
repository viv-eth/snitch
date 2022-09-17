// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist_fp16_network.h"

#include "printf.h"
#include "snrt.h"

#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

typedef float v2f32 __attribute__((vector_size(8)));
typedef union fp32_v2f32_u { 
        v2f32 v2; 
        float v[2]; 
} fp32_v2f32_u;
typedef __fp16 v4f16 __attribute__((vector_size(8)));
union fp16_v4f16_u { 
        v4f16 v4; 
        __fp16 v[4]; 
};

typedef union {
    double f64;
    v4f16 vec;
} v4s;

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

//#define MINDIFF 2.2250738585072014e-308   // smallest positive double
#define MINDIFF 2.25e-308                   // use for convergence check

double my_sqrt(double square)
{
    double root=square/3, last, diff=1;
    if (square <= 0) return 0;
    do {
        last = root;
        root = (root + square / root) / 2;
        diff = root - last;
    } while (diff > MINDIFF || diff < -MINDIFF);
    return root;
}

static float loadFromF16(const __fp16 *pointer) { return *(const __fp16 *)pointer; } 

// bit returned at location
int bit_return(int a, int loc)   
{
    int buf = a & 1<<loc;

    if (buf == 0) return 0;
    else return 1; 
}


// INFO: start of FP16 baseline network implementation
//// Feedforward Step
void feedforward_fp16n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *biases, __fp16 *activations,
                uint32_t ldB, __fp16 *image, uint32_t ldI, uint32_t compute_id){

    const uint32_t IN_CH = IN_CH1*IN_CH2;
    // Linear layer: OUT = X * W^T + B
    for (uint32_t out = 0; out < OUT_CH; out++) {
        __fp16 acc = biases[ldB * out];
        //printf("FP16 baseline init: acc[%u] = %f\n", 1 + compute_id + out * ldB, acc);
        for(uint32_t in = 0; in < IN_CH; in++){
            // if(!compute_id){
            //     printf("image[%u] = %f, weights[%u] = %f\n", in, image[in], in + out * ldW, weights[in + out * ldW]);
            // }
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
    }

    snrt_cluster_hw_barrier();

} // RTL PASS

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
        for(uint32_t out = 0; out < OUT_CH*5; out++){
            act_ptr = &activations[0];
            //printf("DEBUG: act_ptr[%u] = %f\n", out + 1, act_ptr[out]);
            if(act_ptr[out] != 0.0f){
                //printf("DEBUG NON ZERO: act_ptr[%u] = %f\n", out, act_ptr[out]);
                act_ptr[out] = exp(act_ptr[out] - max_global);
                // act_ptr[out] = my_exp(act_ptr[out] - max_global);
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
            printf("FENGA SOFTMAX FP16 (no SIMD): activation[%u] = %f\n", out + 1, activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
} // RTL PASS

void gradient_update_fp16n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weight_grads, uint32_t ldW, __fp16 *bias_grads, __fp16 *activations, 
                uint32_t ldB, __fp16 *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, __fp16 *loss, uint32_t compute_num){

    
    __fp16 b_grad_update = 0.0;
    __fp16 W_grad_update = 0.0;
    __fp16 b_checksum = 0.0;
    __fp16 W_checksum = 0.0;
    volatile uint32_t idx_eff;

    double loss_val = 0.0;

    // get the value saved at target address
    uint32_t target_n = *target;
    // compute the loss
    if(!compute_id){
        loss_val = 0.0 - log(activations[target_n - compute_id]);
        printf("GU current loss[%u] = %f\n", target_n - compute_id, loss_val);
        printf("GU activation[target = %u] = %.15f\n", target_n - compute_id, activations[target_n - compute_id]);
        loss[0] += loss_val;
    } 

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    

    // the effective index is the iteration index of the biases variable
    // across all entries
    for(uint32_t out = 0; out < OUT_CH; out++){
        W_checksum = 0.0;
        idx_eff = compute_id + ldB * out;
        // Gradient Calculation for SoftMax activation with Cross Entropy Loss
        b_grad_update = (idx_eff == *target) ? activations[ldB * out] - 1 : activations[ldB * out];

        printf("FENGA GRADIENT UPDATE FP16 (no SIMD): bias_grads[%u] = %f\n", idx_eff, b_grad_update);

        for(uint32_t in = 0; in < IN_CH; in++){
            
            W_grad_update = b_grad_update * image[in];
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH * OUT_CH * 5 - 1)){	
                weight_grads[out * ldW + in] = W_grad_update;
                W_checksum += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update;
        
        printf("FENGA GRADIENT UPDATE FP16 (no SIMD): W_checksum[%u] = %f\n", idx_eff, W_checksum);
    }

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

} // RTL TODO

void training_step_fp16n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, __fp16 *weight_grads, uint32_t ldW, __fp16 *biases, __fp16 *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images){

    __fp16 lr = 0.5;
    __fp16 b_checksum = 0.0;
    __fp16 W_checksum = 0.0;

    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    uint32_t idx_eff;
    uint32_t idx_eff_W;

    for(uint32_t out = 0; out < OUT_CH; out++){

        idx_eff = compute_id + ldB * out;

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((__fp16) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        printf("FENGA TRAINING STEP FP16 (no SIMD): biases[%u] = %f\n", idx_eff, biases[out * ldB]);

        for(uint32_t in = 0; in < IN_CH; in++){

            idx_eff_W = compute_id*IN_CH + out * ldW + in;
            
            if(!(idx_eff_W > IN_CH * OUT_CH * 5 - 1)){
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in];
                W_checksum += weights[out * ldW + in];
            } 
        }

        printf("FENGA TRAINING STEP FP16 (no SIMD): W_checksum[%u] = %f\n", idx_eff, W_checksum);
    }
}

void feedforward_fp16_ssr_simd_frep(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *biases, __fp16 *activations,
                uint32_t ldB, __fp16 *image, uint32_t ldI, uint32_t compute_id,
                uint32_t setup_SSR, uint32_t curr_img){
    
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));
    
    
    // INFO: for simplicity image is converted to dtype __fp16 
    const register float zero = 0.0f;

    __fp16 acc = 0.0f;
    __fp16 image_scaling_factor; 

    // No scaling for the first iteration 
    // if(curr_img == 0){
    //     image_scaling_factor = 1.0f;
    // } else {
    //     image_scaling_factor = 8.0f;
    // }

    // if(!compute_id){
    //     printf("INFO: image_scaling_factor[%u] = %f\n", curr_img, image_scaling_factor);
    // }

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    volatile uint32_t idx_eff;



    for (uint32_t out = 0; out < OUT_CH; out++) {

        idx_eff = compute_id + ldB * out;

        if(!(idx_eff > OUT_CH * 5 - 1)){
            
            register v2f32 reduce_reg;
            const uint16_t unroll = 4;
            register v2f32 reduce_reg_0;
            register v2f32 reduce_reg_1;
            register v2f32 reduce_reg_2;
            register v2f32 reduce_reg_3;
            register float sum;
            register v2f32 test;
            register v4f16 dotp;
            register v4f16 zero_reg;
            register float tacc;

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
                                OUT_CH - 1, 
                                sizeof(double), 
                                sizeof(double) * ldW / 4);
            }

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weights[out*ldW]);
            
            // Start of SSR region
            snrt_ssr_enable();

            acc = biases[ldB * out];
            
            // TODO: vfpack with .s.s to avoid twice the same op
            asm volatile(
                "vfcpka.s.s    %[tacc], %[zero], %[zero]\n" // zero initialize accumulator
                // "vfcpka.h.s    %[zero_reg], %[zero], %[zero] \n"
                // "vfcpkb.h.s    %[zero_reg], %[zero], %[zero] \n"
                "vfcpka.s.s    %[zero_reg], %[zero], %[zero]\n"
                // "vfcpka.h.s    %[sum], %[zero], %[zero] \n"
                // "vfcpkb.h.s    %[sum], %[zero], %[zero] \n"
                "vfcpka.s.s    %[sum], %[zero], %[zero]\n"
                : [tacc] "+&f"(tacc), [zero_reg] "+&f"(zero_reg), [sum] "+&f"(sum)
                : [zero] "f"(zero)
            );
            
                
            // calculate the dot product of the image and the weights (increment by four columns in each iteration)
            // asm volatile(
            //     "frep.o           %[n_frep], 4, 0, 0\n"
            //     "vfadd.s          %[reduce_reg], %[zero_reg], %[zero_reg] \n"
            //     "vfadd.s          %[sum], %[zero_reg], %[zero_reg] \n"
            //     "vfdotpex.s.h     %[reduce_reg], ft1, ft0 \n"
            //     "vfsum.s          %[sum], %[reduce_reg]\n"
            //     "fadd.s           %[tacc], %[tacc], %[sum] \n"
            //     //"vfcpka.s.s       %[sum], %[zero], %[zero] \n" // GIM: why is this not freped? --> instruction not supported for FREP
            // : [sum] "+f"(sum), [dotp] "+f"(dotp), [tacc] "+f"(tacc), 
            //   [zero_reg] "+&f"(zero_reg), [reduce_reg] "+&f"(reduce_reg)
            // : [zero] "f"(zero), [n_frep] "r"(IN_CH / 4 - 1)
            // : "ft0", "ft1", "ft2"
            // );

            asm volatile(
                "frep.o           %[n_frep], 13, 0, 0\n"
                "vfadd.s          %[reduce_reg_0], %[zero_reg], %[zero_reg] \n"
                "vfadd.s          %[reduce_reg_1], %[zero_reg], %[zero_reg] \n"
                "vfadd.s          %[reduce_reg_2], %[zero_reg], %[zero_reg] \n"
                "vfadd.s          %[reduce_reg_3], %[zero_reg], %[zero_reg] \n"
                "vfadd.s          %[sum], %[zero_reg], %[zero_reg] \n"
                "vfdotpex.s.h     %[reduce_reg_0], ft1, ft0 \n"
                "vfdotpex.s.h     %[reduce_reg_1], ft1, ft0 \n"
                "vfdotpex.s.h     %[reduce_reg_2], ft1, ft0 \n"
                "vfdotpex.s.h     %[reduce_reg_3], ft1, ft0 \n"
                "vfsum.s          %[sum], %[reduce_reg_0]\n"
                "vfsum.s          %[sum], %[reduce_reg_1]\n"
                "vfsum.s          %[sum], %[reduce_reg_2]\n"
                "vfsum.s          %[sum], %[reduce_reg_3]\n"
                "fadd.s           %[tacc], %[tacc], %[sum] \n"
                //"vfcpka.s.s       %[sum], %[zero], %[zero] \n" // GIM: why is this not freped? --> instruction not supported for FREP
            : [sum] "+f"(sum), [dotp] "+f"(dotp), [tacc] "+f"(tacc), 
              [zero_reg] "+&f"(zero_reg), 
              [reduce_reg_0] "+&f"(reduce_reg_0), [reduce_reg_1] "+&f"(reduce_reg_1),
              [reduce_reg_2] "+&f"(reduce_reg_2), [reduce_reg_3] "+&f"(reduce_reg_3)
            : [zero] "f"(zero), [n_frep] "r"(IN_CH / (4 * unroll) - 1)
            : "ft0", "ft1", "ft2"
            );

            // End of SSR region.
            snrt_ssr_disable();
            // snrt_fpu_fence();
            acc += tacc;
            // printf("FEEDFORWARD DEBUG acc[%u]: %f\n", 1 + compute_id + out * ldB, acc);
            activations[ldB * out] = acc;
            acc = 0.0;
            asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

        } else {
            acc = 0.0;
            activations[ldB * out] = acc;
        }

    }


    for (uint32_t out = 0; out < OUT_CH; out++) {
        printf("FENGA FP16 SIMD with SSRs: acc[%u] = %f\n", 1 + compute_id + out * ldB, activations[ldB * out]);
    }

    snrt_cluster_hw_barrier();

} // RTL PASS

//// Gradient Update
void gradient_update_fp16_ssr_simdn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weight_grads, uint32_t ldW, __fp16 *bias_grads, __fp16 *activations, 
                uint32_t ldB, __fp16 *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, __fp16 *loss, uint32_t compute_num, uint32_t setup_SSR){

    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));

    register float b_grad_update = 0.0;
    double loss_val = 0.0;
    register v4f16 b_grad_update_reg;
    register v4f16 W_grad_update_reg;

    register v4f16 image_scaling_reg;
    register v4f16 scaled_image_reg;
    float image_scaling_factor = 8;

    register v4f16 zero_reg = {0.0, 0.0, 0.0, 0.0};
    

    __fp16 W_checksum = 0.0;
    __fp16 W_grad_update = 0.0;
    uint16_t idx_eff;
    uint32_t idx_eff_W;
    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    

    // get the value saved at target address
    uint16_t target_n = target[0];

    // compute the loss
    if(!compute_id){
        loss_val = 0.0 - log(activations[target_n - compute_id]);
        // printf("loss activation[target] = %f\n", activations[target_n - compute_id]);
        printf("GU current loss [%u] = %.15f\n", target_n - compute_id, loss_val);
        printf("GU activation[target = %u] = %.15f\n", target_n - compute_id, activations[target_n - compute_id]);
        loss[0] += loss_val;
    } 

    for(uint32_t out = 0; out < OUT_CH; out++){

        W_checksum = 0.0;
        
        idx_eff = compute_id + ldB * out;

        if(!(idx_eff > OUT_CH * 5 - 1)) {
            b_grad_update = (idx_eff == target_n) ? activations[ldB * out] - 1 : activations[ldB * out];
            bias_grads[ldB * out] = b_grad_update;
            if (setup_SSR) {

                // READ setup of DATA MOVER input data (MNIST image)
                snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                                IN_CH / 4, 
                                sizeof(double));
                
                // WRITE setup of DATA MOVER for image
                snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                                IN_CH / 4,  
                                sizeof(double));

                // WRITE setup of DATA MOVER for weight gradients
                snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                                IN_CH / 4,  
                                sizeof(double));
            }

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image); // read image ft0
            // snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weight_grads[out*ldW]); // read image ft1
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weight_grads[out*ldW]); // write weight gradients ft2
            

            for(uint32_t in = 0; in < IN_CH;){
                idx_eff_W = compute_id*IN_CH + out * ldW + in;
                if(!(idx_eff_W  > IN_CH * OUT_CH * 5 - 1)){ 

                    // if(weight_grads[out*ldW + in] > 1000.0 || weight_grads[out*ldW + in] < -1000.0 || weight_grads[out * ldW + in + 1] > 1000.0 || weight_grads[out * ldW + in + 1] < -1000.0 || weight_grads[out * ldW + in + 2] > 1000.0 || weight_grads[out * ldW + in + 2] < -1000.0 || weight_grads[out * ldW + in + 3] > 1000.0 || weight_grads[out * ldW + in + 3] < -1000.0){
                    //     printf("GU weight_grads[%u] = %f\n", out*ldW + in, weight_grads[out*ldW + in]);
                    // }

                    // Start of SSR region
                    snrt_ssr_enable();  

                    asm volatile(
                        "vfcpka.h.s      %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
                        "vfcpkb.h.s      %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
                        "vfcpka.h.s      %[image_scaling_reg], %[image_scaling_factor], %[image_scaling_factor] \n"
                        "vfcpkb.h.s      %[image_scaling_reg], %[image_scaling_factor], %[image_scaling_factor] \n"
                        "vfmul.h         %[scaled_image_reg], %[image_scaling_reg], ft0 \n"
                        "vfmul.h         %[W_grad_update_reg], %[scaled_image_reg], %[b_grad_update_reg] \n"
                        "vfdiv.h         ft2, %[W_grad_update_reg], %[image_scaling_reg] \n"
                        // "vfdiv.h         ft2, %[W_grad_update_reg], %[image_scaling_reg] \n"
                        // "vfadd.h         ft2, %[W_grad_update_reg], %[zero_reg] \n"
                        // "vfadd.h         ft2, %[W_grad_update_reg], ft1 \n"
                        : [b_grad_update_reg] "+&f"(b_grad_update_reg), [W_grad_update_reg] "+&f"(W_grad_update_reg), 
                          [image_scaling_reg] "+&f"(image_scaling_reg), [scaled_image_reg] "+&f"(scaled_image_reg),
                          [zero_reg] "+&f"(zero_reg)
                        : [b_grad_update] "f"(b_grad_update), [image_scaling_factor] "f"(image_scaling_factor)
                        : "ft0", "ft1", "ft2"
                    );

                    // weight_grads[out*ldW + in + 0] = W_grad_update_reg[0];
                    // weight_grads[out*ldW + in + 1] = W_grad_update_reg[1];
                    // weight_grads[out*ldW + in + 2] = W_grad_update_reg[2];
                    // weight_grads[out*ldW + in + 3] = W_grad_update_reg[3];
                    snrt_ssr_disable();
                    // INFO: after disabling the SSRs we can free the registers
                    asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));
                    // if(weight_grads[out*ldW + in] > 1000.0 || weight_grads[out*ldW + in] < -1000.0 || weight_grads[out * ldW + in + 1] > 1000.0 || weight_grads[out * ldW + in + 1] < -1000.0 || weight_grads[out * ldW + in + 2] > 1000.0 || weight_grads[out * ldW + in + 2] < -1000.0 || weight_grads[out * ldW + in + 3] > 1000.0 || weight_grads[out * ldW + in + 3] < -1000.0){
                    //     printf("GU weight_grads[%u] = %f\n", out*ldW + in, weight_grads[out*ldW + in]);
                    // }
                    // if(!compute_id) {
                    //     printf("GU weight_grads[%u] = %f\n", out*ldW + in, weight_grads[out*ldW + in]);
                    //     printf("GU weight_grads[%u] = %f\n", out*ldW + in + 1, weight_grads[out*ldW + in + 1]);
                    //     printf("GU weight_grads[%u] = %f\n", out*ldW + in + 2, weight_grads[out*ldW + in + 2]);
                    //     printf("GU weight_grads[%u] = %f\n", out*ldW + in + 3, weight_grads[out*ldW + in + 3]);
                    // }
                    W_checksum += weight_grads[out*ldW + in] + weight_grads[out*ldW + in + 1] + weight_grads[out*ldW + in + 2] + weight_grads[out*ldW + in + 3];
                    // account for the scaling factor
                    // W_checksum /= image_scaling_factor;
                } else {
                    W_checksum = 0.0;
                    weight_grads[out*ldW + in + 0] = 0.0;
                    weight_grads[out*ldW + in + 1] = 0.0;
                    weight_grads[out*ldW + in + 2] = 0.0;
                    weight_grads[out*ldW + in + 3] = 0.0;
                }

                in += 4;
            }

        } else {
            bias_grads[ldB * out] = 0.0;
        }
        
        // printf("FENGA GRADIENT UPDATE FP16 SIMD with SSRs: bias_grads[%u] = %f\n", compute_id + out * ldB, bias_grads[ldB * out]);
        // printf("FENGA GRADIENT UPDATE FP16 SIMD with SSRs: W_checksum[%u] = %f\n", idx_eff, W_checksum );

    }

    snrt_cluster_hw_barrier();
}

//// Training Step
void training_step_fp16_ssr_simdn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, volatile __fp16 *weight_grads, uint32_t ldW, volatile __fp16 *biases, __fp16 *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR, uint32_t curr_img){

    // INFO: due to a compiler bug we need to reserve the registers for the SSR
    //       otherwise it will use them for stack operations breaking the stream(s)
    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));

    register v4f16 image_scaling_reg;
    register v4f16 scaled_image_reg;
    register v4f16 scaled_weights_reg;
    float image_scaling_factor; 

    if(curr_img == 0){
        image_scaling_factor = 8.0;
    } else {
        image_scaling_factor = 8.0;
    }

    register v4f16 zero_reg = {0.0, 0.0, 0.0, 0.0};

    
    register float lr = 0.5;
    register v4f16 lr_vec;

    __fp16 W_checksum = 0.0f;

    uint32_t idx_eff = 0;
    uint32_t idx_eff_W = 0;

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    for(uint32_t out = 0; out < OUT_CH; out++){

        W_checksum = 0.0;

        asm volatile(
            "vfcpka.h.s          %[lr_vec], %[lr], %[lr] \n"
            "vfcpkb.h.s          %[lr_vec], %[lr], %[lr] \n"
            "vfcpka.h.s          %[image_scaling_reg], %[image_scaling_factor], %[image_scaling_factor] \n"
            "vfcpkb.h.s          %[image_scaling_reg], %[image_scaling_factor], %[image_scaling_factor] \n"
            : [lr_vec] "+&f"(lr_vec), [image_scaling_reg] "+&f"(image_scaling_reg)
            : [lr] "f"(-lr), [image_scaling_factor] "f"(image_scaling_factor)
            : "ft0", "ft1", "ft2"
        );

        idx_eff = compute_id + out * ldB;

        if(!(idx_eff > OUT_CH * 5 - 1)){

            if (setup_SSR) {
                
                // SSR read setup of weight gradients
                snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                                IN_CH / 4,
                                sizeof(double));
                
                // SSR read setup of weights
                snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                                IN_CH / 4, 
                                sizeof(double));

                // SSR write setup of weights
                snrt_ssr_loop_1d(SNRT_SSR_DM2, 
                                IN_CH / 4, 
                                sizeof(double));
                
            }
            
            // SSR start addresses need to be configured each time
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &weight_grads[out*ldW]); // weight gradients stored in ft0
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]); // weights stored in ft1 for read
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, &weights[out*ldW]); // weights stored in ft2 for write

            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((__fp16) number_of_images);

	        // b_checksum += biases[ldB * out];
        } else {
            biases[ldB * out] += 0.0f;
        }

        printf("FENGA TRAINING STEP FP16 with SSRs: biases[%u] = %f\n", idx_eff, biases[ldB * out]);
        
        register v4f16 reduce_reg;
        uint16_t unroll = 4;
        register v4f16 unroll_reg[4];
        const register float zero = 0.0;
        register v2f32 zero_reg;
        register float sum = 0.0f;
        // zero initialize the reduce register
        // asm volatile (
        //     "vfcpka.h.s          %[reduce_reg], %[zero], %[zero] \n"
        //     "vfcpkb.h.s          %[reduce_reg], %[zero], %[zero] \n"
        //     : [reduce_reg] "+&f"(reduce_reg), [zero_reg] "+&f"(zero_reg)
        //     : [zero] "f"(zero)
        //     : "ft0", "ft1", "ft2"
        // );
        for(uint32_t in = 0; in < IN_CH;){
            idx_eff_W = compute_id*IN_CH + out * ldW + in;
            snrt_ssr_enable();
            if(!(idx_eff_W  > IN_CH * OUT_CH * 5 - 1)){ 
                
                asm volatile(
                    // "vfmul.h             %[scaled_image_reg], %[image_scaling_reg], ft0 \n"
                    // "vfmul.h             %[scaled_weights_reg], %[image_scaling_reg], ft1 \n"
                    "vfmul.h             %[reduce_reg], %[lr_vec], ft0 \n"
                    // "vfmul.h             %[reduce_reg], %[lr_vec], ft0 \n"
                    // "vfadd.h             %[scaled_weights_reg], %[scaled_weights_reg], %[reduce_reg] \n"
                    "vfadd.h             ft2, ft1, %[reduce_reg] \n"
                    // "vfdiv.h              ft2, %[scaled_weights_reg], %[image_scaling_reg] \n"     // divde by the size of the dataset --> banshee: add floating point exception for divide by zero
                : [reduce_reg] "+&f"(reduce_reg), [scaled_image_reg] "+&f"(scaled_image_reg), [scaled_weights_reg] "+&f" (scaled_weights_reg)
                : [zero] "f"(zero), [lr] "f"(-lr), [lr_vec] "f"(lr_vec), [image_scaling_reg] "f"(image_scaling_reg)
                : "ft0", "ft1", "ft2"
                ); 
            } 

            snrt_ssr_disable();
            // INFO: after disabling the SSRs we can free the registers
            asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));
            W_checksum += weights[out*ldW + in + 0] + weights[out*ldW + in + 1] 
                        + weights[out*ldW + in + 2] + weights[out*ldW + in + 3];

            in += 4;
            // in += 4 * unroll;
        }

        printf("FENGA TRAINING STEP FP16 SIMD with SSRs: W_checksum[%u] = %f\n", compute_id, W_checksum);
    }

}

// TODO: implement norm gradient clipping
void gradient_norm_clip(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                        volatile __fp16 *weight_grads, uint32_t ldW, uint32_t ldB, 
                        uint32_t compute_id, uint32_t compute_num,
                        __fp16 *sum, float max_norm, float *scaling) {

    // determine the sum over all weight gradients
    float sum_core = 0.0f;
    float sum_global = 0.0f;
    float grad_norm;
    uint32_t idx_eff = 0;
    uint32_t idx_eff_W = 0;

    uint32_t IN_CH = IN_CH1 * IN_CH2;

    for(uint32_t out = 0; out < OUT_CH; out++){

        idx_eff = compute_id + out * ldB;

        for(uint32_t in = 0; in < IN_CH; in++){

            idx_eff_W = compute_id*IN_CH + out * ldW + in;
            if(!(idx_eff_W  > IN_CH * OUT_CH * 5 - 1)) {
                sum_core += pow(weight_grads[out*ldW + in], 2);
            }
        }
    }

    sum[compute_id] = sum_core;

    printf("FENGA GRADCLIP FP16: sum[%u] = %f\n", compute_id, sum[compute_id]);

    snrt_cluster_hw_barrier();

    // sum_global += sum[compute_id];

    printf("FENGA GRADCLIP STEP FP16 after HW barrier: sum[%u] = %f\n", compute_id, sum[compute_id]);

    // reduction on single core
    if(compute_id == 0){ 
        for(uint32_t core = 0; core < compute_num; core++) {
            printf("FENGA TRAINING STEP FP16: sum[%u] = %f\n", core, sum[core]);
            sum_global += sum[core];
        }

        printf("FENGA GRADIENT NORM CLIPPING FP16: sum_global = %f\n", sum_global);

        // manually compute the square root of the sum
        grad_norm = my_sqrt(sum_global);


        printf("FENGA GRADIENT NORM CLIPPING FP16: grad_norm = %f\n", grad_norm);

        // if the norm is greater than the max norm, scale the gradients
        if(grad_norm > max_norm) {
            float scaling_factor;
            // scaling = MIN(max_norm / grad_norm, 1.0f);
            if (max_norm / grad_norm < 1.0f) {
                scaling_factor = max_norm / grad_norm;
            } else {
                scaling_factor = 1.0f;
            }
            // TODO: make scaling factor available on all cores
            // printf("FENGA GRADIENT NORM CLIPPING FP16: gradient norm clipping by scaling = %f\n", scaling_factor);
            for(uint32_t core = 0; core < compute_num; core++) {
                scaling[core] = scaling_factor;
            }   
            // for(uint32_t out = 0; out < OUT_CH*5; out++){
            //     for(uint32_t in = 0; in < IN_CH; in++){
            //         weight_grads[out*ldW + in] *= scaling;
            //     }
            // }
        }
    }

    snrt_cluster_hw_barrier();

    printf("FENGA GRADIENT NORM CLIPPING FP16: scaling[%u] = %f\n", compute_id, scaling[compute_id]);

    for(uint32_t out = 0; out < OUT_CH; out++){

        idx_eff = compute_id + out * ldB;

        for(uint32_t in = 0; in < IN_CH; in++){

            idx_eff_W = compute_id*IN_CH + out * ldW + in;
            if(!(idx_eff_W  > IN_CH * OUT_CH * 5 - 1)) {
                weight_grads[out*ldW + in] *= scaling[compute_id];
            }
        }
    }


    snrt_cluster_hw_barrier();
}    
