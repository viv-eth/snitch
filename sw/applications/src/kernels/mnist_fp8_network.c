// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include <string.h>

#include "mnist_fp8_network.h"

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

double my_exp(float x) 
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

float get_float_from_byte(char value) {
    
    int i, s, m, e = 0;

    float sign;

    int sign_val = 0;
    for(s = 0; s < 1; s++) {
        sign_val = !!((value << s) & 0x80);
    }

    sign = sign_val ? -1.0 : 1.0;

    int exp_acc = 0;
    uint32_t exp_len = 4; // start counting from zero, so it's actually 5 bits
    char exponent = 0b00000000;

    for(e = 1; e < 6; e++) {
        int exp_val = !!((value << e) & 0x80);
        // get the exponent value
        if(exp_val) {
            // account for 2s complement
            if(e == 1) {
                exp_acc -= (int)(pow(2, exp_len));
            } else {
                exp_acc += (int)(pow(2, exp_len));
            }
        }
        exp_len -= 1;
        // if(e == 1 && exp_val != 0) {
        //     exp_acc -= pow(2, exp_len);
        // } else if (e != 1 && exp_val != 0) {
        //     exp_acc += pow(2, exp_len);
        // }
    }
    int man_acc = 0;
    uint32_t man_len = 1; // start counting from zero, so it's actually 2 bits
 
    for(m = 6; m < 8; m++) {
        int man_val = !!((value << m) & 0x80); 
        if(man_val != 0) {
            man_acc += (int)(pow(2, man_len));
        }

        man_len -= 1;
    }

    int fp8_bias = 15;

    //printf("Sign val = %d, Mantissa = %d, Exponent = %d, Exponent Bias = %d\n", sign_val, man_acc, exp_acc, fp8_bias);

    float float_num = sign * (1 + man_acc/pow(2, 2)) * pow(2, exp_acc - fp8_bias);

    if(sign_val == 0 && man_acc == 0 && exp_acc == 0){
        float_num = 0;
    }

    // printf("float_val = %.10f\n", float_num);

    return float_num;
}

// function to print the bits of a char
void print_byte(char value){

    for (int i = 0; i < 8; i++) {
      printf("%d", !!((value << i) & 0x80));
    }

}

// INFO: start of FP8 baseline network implementation
//// Feedforward Step
void feedforward_fp8n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *biases, char *activations,
                uint32_t ldB, char *image, uint32_t ldI, uint32_t compute_id, float *activations_fp32){
    
    const uint32_t IN_CH = IN_CH1*IN_CH2;
    // Linear layer: OUT = X * W^T + B
    for (uint32_t out = 0; out < OUT_CH; out++) {
        char acc = biases[ldB * out];
        for(uint32_t in = 0; in < IN_CH; in++){
            // if(!compute_id) {
            //     printf("image[%d] = %d\n", in, image[in]);
            // }
            acc += image[in] * weights[out * ldW + in];
            // INFO: If this is not set harts start reading outside the mem map
            // FIXME: Next harts should start computation of the subsequent image
            if(compute_id + out * ldB > OUT_CH * 5 - 1){
                acc = 0b00000000;
            }
        }
        // OUT is accumulated in activations 
        activations[ldB * out] = acc;
        printf("FEEDFORWARD FP8 Baseline: acc[%u] = ", 1 + compute_id + out * ldB);
        print_byte(activations[ldB * out]);  
        // printf("\n");
        printf(" = %d\n", activations[ldB * out]);
        // get_float_from_byte(activations[ldB * out]);
    }

    snrt_cluster_hw_barrier();

}

//// Activation Step
void softmax_activation_fp8n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *activations, uint32_t ldB,
                char *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, char *max){

    char max_core;
    char sum = 0.0f;
    char *act_ptr;

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

    char max_global = max[0];

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
            // printf("DEBUG: act_ptr[%u] = %d\n", out + 1 + compute_id, act_ptr[out]);
            if(act_ptr[out] != 0b00000000){
                // printf("DEBUG NON ZERO: act_ptr[%u] = %d\n", out + 1 + compute_id, act_ptr[out]);
                // act_ptr[out] = exp(act_ptr[out] - max_global);
                act_ptr[out] = my_exp(act_ptr[out] - max_global);
                printf("DEBUG: act_ptr[%u] = %d\n", out + 1 + compute_id, act_ptr[out]);
                sum += act_ptr[out];
                //printf("DEBUG: sum = %f\n", sum);
            } else {
                //printf("DEBUG ZERO: act_ptr[%u] = %f\n", out, act_ptr[out]);
                act_ptr[out] = 0b00000000;
            }
        }


        for(uint32_t out = 0; out < OUT_CH*5; out++){
            act_ptr[out] /= sum;
            activations[out] = act_ptr[out];
            printf("SOFTMAX FP8 (no SIMD): activation[%u] = ", out + 1);
            print_byte(activations[out]);
            printf(" = %d\n", activations[out]);
            // printf(" = ");
            // get_float_from_byte(activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
}

void softmax_activation_fp32_ex(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH,
                float *activations_fp32, char *activations, uint32_t ldB, uint32_t compute_id, 
                uint32_t compute_num, float *max){

    float max_core = 0.0;
    float sum = 0.0;
    float max_global;

    volatile uint32_t idx_eff;

    // snrt_ssr_enable();

    max_core = activations_fp32[0];

    for(uint32_t out = 0; out < OUT_CH; out++){
        idx_eff = compute_id + ldB * out;
        if(!(idx_eff > OUT_CH * 5 - 1)){
            if(activations_fp32[ldB * out] > max_core) {
                max_core = activations_fp32[ldB * out];
            }
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
        
        // FIXME: actually OUT_CH should be multiplied by number of compute cores
        for(uint32_t out = 0; out < OUT_CH*5; out++){
            if(activations_fp32[out]){
                // activations_fp32[out] = exp(activations_fp32[out] - max_global);
                // printf("DEBUG: activations_fp32[%u] - max_global = %.10f\n", out, activations_fp32[out] - max_global);
                activations_fp32[out] = my_exp(activations_fp32[out] - max_global);
                sum += activations_fp32[out];
            } else {
                activations_fp32[out] = 0.0;
            }
        }


        for(uint32_t out = 0; out < OUT_CH*5; out++){
            activations_fp32[out] /= sum;
            float act_fp32 = activations_fp32[out];
            // printf("activations_fp32[%u] = %.10f\n", out, act_fp32);
            register v8f8 act_fp8_ptr;
            register v2f32 sum;
            register float sum_f;
            // printf("activations_fp8[%u] = ", out);
            // print_byte(activations[out]);
            // printf(" = %d\n", activations[out]);
            asm volatile(
                "vfcpka.b.s       %[act_fp8_ptr], %[act_fp32], %[zero] \n"
                "vfcpkb.b.s       %[act_fp8_ptr], %[zero], %[zero] \n"
                "vfcpkc.b.s       %[act_fp8_ptr], %[zero], %[zero] \n"
                "vfcpkd.b.s       %[act_fp8_ptr], %[zero], %[zero] \n"
                : [act_fp8_ptr] "+f" (act_fp8_ptr), [act_fp32] "+f" (act_fp32)
                : [zero] "f" (0.0f)
                : "ft0", "ft1", "ft2"
            );
            activations[out] = act_fp8_ptr[0];
            float fp8_float = get_float_from_byte(activations[out]);
            // printf("SOFTMAX FP32 expanding (no SIMD): activation[%u] = %.10f\n", out + 1, activations_fp32[out]);
            printf("SOFTMAX FP32 expanding (no SIMD): activation[%u] = ", out + 1);
            print_byte(activations[out]);
            printf(" = %d = %f\n", activations[out], fp8_float);

        }
    }

    snrt_cluster_hw_barrier();
}

void gradient_update_fp8n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weight_grads, uint32_t ldW, char *bias_grads, char *activations, 
                uint32_t ldB, char *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, char *loss, uint32_t compute_num){

    
    char b_grad_update = 0.0;
    char W_grad_update = 0.0;
    char b_checksum = 0.0;
    char W_checksum = 0.0;
    volatile uint32_t idx_eff;


    // get the value saved at target address
    uint32_t target_n = *target;
    // compute the loss
    // TODO: update loss computation
    // __fp16 loss_val = 0.0 - log(activations[target_n - compute_id]);

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
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH * OUT_CH * 5 - 1)){	
                weight_grads[out * ldW + in] += W_grad_update;
                W_checksum += W_grad_update;
            }
        }
            
        bias_grads[ldB * out] = b_grad_update;
        // printf("GRADIENT UPDATE FP16 Baseline: bias_grads[%u] = %f\n", 1 + compute_id + out * ldB, bias_grads[ldB * out]);
    }

    printf("GRADIENT UPDATE FP8 Baseline: W_checksum[%u] = ", 1 + compute_id);
    print_bits(W_checksum);
    printf("GRADIENT UPDATE FP8 Baseline: b_checksum[%u] = ", 1 + compute_id);
    print_bits(b_checksum);

    snrt_cluster_hw_barrier(); // INFO: target variable lost after HW barrier

}

void training_step_fp8n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, char *weight_grads, uint32_t ldW, char *biases, char *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images){

    char lr = 0.5;
    char b_checksum = 0.0;
    char W_checksum = 0.0;

    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    for(uint32_t out = 0; out < OUT_CH; out++){

        // make sure that biases outside of the number of
        // output channels are zero
        if(!(compute_id + out * ldB > OUT_CH * 5 - 1)){
            biases[ldB * out] -= lr * bias_grads[ldB * out] / ((char) number_of_images);
        } else {
            biases[ldB * out] = 0;
        }

        b_checksum += biases[ldB * out];

        for(uint32_t in = 0; in < IN_CH; in++){
            
            if(!(compute_id*IN_CH + out * ldW + in > IN_CH * OUT_CH * 5 - 1)){
                // if(compute_id*IN_CH + out * ldW + in % 4 == 0){
                //     printf("DEBUG: weight_grad[%u] = %f\n", compute_id*IN_CH + out * ldW + in, weight_grads[out * ldW + in]);
                // }
                weights[out * ldW + in] -= lr * weight_grads[out * ldW + in] / ((char) number_of_images);
                W_checksum += weights[out * ldW + in];
            } 
        }
    }

    printf("TRAINING STEP FP8 Baseline: W_checksum[%u] = ", 1 + compute_id);
    print_bits(W_checksum);
    printf("TRAINING STEP FP8 Baseline: b_checksum[%u] = ", 1 + compute_id);
    print_bits(b_checksum);
}

// INFO: start of FP8 optimized network implementation
//// Feedforward Step
void feedforward_fp8n_opt(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *biases, char *activations,
                uint32_t ldB, char *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t setup_SSR, float *activations_fp32){

    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" : "=f"(ft0), "=f"(ft1), "=f"(ft2));


    char acc = 0b00000000;

    const register float zero = 0.0;

    // get the total number of input features
    const uint32_t IN_CH = IN_CH1 * IN_CH2;
    uint32_t idx_eff;

    for (uint32_t out = 0; out < OUT_CH; out++) {
        // get the output activation index
        idx_eff = compute_id + ldB * out;
        // check if we are out of bound
        if(!(idx_eff > OUT_CH * 5 - 1)){

            // SSR strides and bounds only have to be configured
            // once in the beginning
            // WARN In the RTL SSR strides MUST BE of size DOUBLE
            // For each SSR stride we load eight FP8 values

            if (setup_SSR) {

                // setup of DATA MOVER input data (MNIST image)
                snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                                IN_CH / 8, 
                                sizeof(double));
                
                // setup of DATA MOVER for weights
                // snrt_ssr_loop_2d(SNRT_SSR_DM1, 
                //                 IN_CH / 8, 
                //                 OUT_CH - 1, 
                //                 sizeof(double), 
                //                 sizeof(double) * ldW / 8);
                snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                                IN_CH / 8, 
                                sizeof(double));
            }

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &weights[out*ldW]);

            register float reduce_reg;
            const uint16_t unroll = 4;
            register v2f32 sum;
            register v8f8 dotp;
            register v4f16 zero_reg;
            register float tacc;
            register v4f16 c;
            register v8f8 test;

            // Start of SSR region
            // snrt_ssr_enable();
            // printf("FP8opt DEBUG: before computation acc = ");
            // print_byte(acc);
            // printf("\n");

            snrt_ssr_enable();

            acc = biases[ldB * out];

            asm volatile(
                "vfcpka.s.s    %[tacc], %[zero], %[zero]\n" // zero initialize accumulator
                "vfcpka.h.s    %[zero_reg], %[zero], %[zero] \n"
                "vfcpkb.h.s    %[zero_reg], %[zero], %[zero] \n"
                "vfcpka.h.s    %[sum], %[zero], %[zero] \n"
                "vfcpkb.h.s    %[sum], %[zero], %[zero] \n"
                "vfadd.s       %[c], %[zero_reg], %[zero_reg] \n"
                "vfadd.s       %[test], %[zero_reg], %[zero_reg] \n"
                : [tacc] "+&f"(tacc), [zero_reg] "+&f"(zero_reg), [sum] "+&f"(sum), 
                  [c] "+&f"(c), [test] "+&f"(test)
                : [zero] "f"(zero)
                : "ft0", "ft1", "ft2"
            );

            // calculate the dot product of the image and the weights (increment by four columns in each iteration)
            asm volatile(
                "frep.o           %[n_frep], 1, 0, 0\n"
                "vfdotpex.h.b     %[c], ft1, ft0 \n"
                "vfsumex.s.h      %[sum], %[c]\n"
                "vfsum.s          %[tacc], %[sum]\n"
                "vfcpka.b.s       %[test], %[tacc], %[zero] \n"
                "vfcpkb.b.s       %[test], %[zero], %[zero] \n"
                "vfcpkc.b.s       %[test], %[zero], %[zero] \n"
                "vfcpkd.b.s       %[test], %[zero], %[zero] \n"
            //     // "fadd.s           %[tacc], %[tacc], %[sum] \n"
            //     //"vfcpka.s.s       %[sum], %[zero], %[zero] \n" // GIM: why is this not freped? --> instruction not supported for FREP
            : [sum] "+f"(sum), [dotp] "+f"(dotp), [tacc] "+f"(tacc), 
              [zero_reg] "+&f"(zero_reg), [reduce_reg] "+&f"(reduce_reg),
              [c] "+&f"(c), [test] "+&f"(test)
            : [zero] "f"(zero), [n_frep] "r"(IN_CH / 8 - 1)
            : "ft0", "ft1", "ft2"
            );

            // End of SSR region.
            snrt_ssr_disable();
            // printf("test[%u] = ", compute_id + ldB * out + 1);
            // print_bits(test[0]);
            // snrt_fpu_fence();
            acc += test[0];
            activations[ldB * out] = acc;
            float acc_float = get_float_from_byte(acc);
            // activations_fp32[ldB * out] = 1000*acc_float;
            activations_fp32[ldB * out] = acc_float;
            printf("FEEDFORWARD FP8 OPT: acc[%u] = ", 1 + compute_id + out * ldB);
            print_byte(activations[ldB * out]);  
            printf(" = %d = %0.10f\n", activations[ldB * out], activations_fp32[ldB * out]);
            acc = 0b00000000;

        }
    }

    snrt_cluster_hw_barrier();
}

gradient_update_fp8n_opt(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                        char *weight_grads, uint32_t ldW, char *bias_grads,
                        float *activations_fp32, uint32_t ldB, char *image, 
                        uint32_t *target, uint32_t ldI, uint32_t compute_id, 
                        char *loss, uint32_t compute_num, uint32_t setup_SSR){

    register volatile double ft0 asm("ft0");
    register volatile double ft1 asm("ft1");
    register volatile double ft2 asm("ft2");
    asm volatile("" ::"f"(ft0), "f"(ft1), "f"(ft2));
    
    float b_grad_update = 0.0;
    float b_checksum = 0.0;
    float W_checksum = 0.0;

    uint32_t idx_eff;
    uint32_t idx_eff_W;

    // get the value saved at target address
    uint32_t target_n = *target;

    const uint32_t IN_CH = IN_CH1 * IN_CH2;

    for(uint32_t out = 0; out < OUT_CH; out++){
        idx_eff = compute_id + ldB * out;
        // printf("idx_eff = %u\n", idx_eff);
        if(!(idx_eff > OUT_CH * 5 - 1)){
            // printf("idx_eff = %u\n", idx_eff);
            b_grad_update = (idx_eff == *target) ? activations_fp32[ldB * out] - 1 : activations_fp32[ldB * out];
            b_checksum += b_grad_update;
            // now we pack the b_grad_update into the bias_grads vector
            register v8f8 b_grad_update_reg;
            // we define a reduce register 
            register v8f8 reduce_reg;
            register v4f16 sum_reduce_reg_v4;
            register v2f32 sum_reduce_reg_v2;
            register float sum = 0.0;
            asm volatile(
                "vfcpka.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
                "vfcpkb.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
                "vfcpkc.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
                "vfcpkd.b.s       %[b_grad_update_reg], %[b_grad_update], %[b_grad_update] \n"
                "vfcpka.s.s       %[reduce_reg], %[zero], %[zero] \n"
                : [b_grad_update_reg] "+&f"(b_grad_update_reg), [reduce_reg] "+&f"(reduce_reg)
                : [b_grad_update] "f"(b_grad_update), [zero] "f"(0.0)
                : "ft0", "ft1", "ft2"
            );

            if (setup_SSR) {

                // setup of DATA MOVER input data (MNIST image)
                snrt_ssr_loop_1d(SNRT_SSR_DM0, 
                                IN_CH / 8, 
                                sizeof(double));
                
                // SSR READ setup of weight grads
                snrt_ssr_loop_1d(SNRT_SSR_DM1, 
                                IN_CH / 8, 
                                sizeof(double));

            }

            // Start of SSR region
            snrt_ssr_enable();

            // SSR start address need to be configured each time
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, image);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &weight_grads[out*ldW]);

            for(uint32_t in = 0; in < IN_CH;){
                idx_eff_W = compute_id * IN_CH + out * ldW + in;
                if(!(idx_eff_W + 7 > IN_CH * OUT_CH * 5 - 1)){

                    asm volatile(
                        "vfcpka.s.s       %[reduce_reg], %[zero], %[zero] \n"
                        "vfmul.b          %[reduce_reg], %[b_grad_update_reg], ft0 \n"
                        "vfadd.b          %[reduce_reg], %[reduce_reg], ft1 \n"
                        // INFO: below lines for debugging only
                        "vfcpka.s.s       %[sum_reduce_reg_v4], %[zero], %[zero]\n"
                        "vfcpka.s.s       %[sum_reduce_reg_v2], %[zero], %[zero]\n"
                        "vfcpka.s.s       %[sum], %[zero], %[zero]\n"
                        "vfsumex.h.b      %[sum_reduce_reg_v4], %[reduce_reg] \n"
                        "vfsumex.s.h      %[sum_reduce_reg_v2], %[sum_reduce_reg_v4] \n"
                        "vfsum.s          %[sum], %[sum_reduce_reg_v2] \n"
                        : [b_grad_update_reg] "+&f"(b_grad_update_reg), [reduce_reg] "+&f"(reduce_reg),
                          [sum_reduce_reg_v4] "+&f"(sum_reduce_reg_v4), [sum_reduce_reg_v2] "+&f"(sum_reduce_reg_v2),
                          [sum] "+&f"(sum)
                        : [b_grad_update] "f"(b_grad_update), [zero] "f"(0.0)
                        : "ft0", "ft1", "ft2"
                    );

                    snrt_ssr_disable(); 
                    weight_grads[out*ldW + in + 0] += reduce_reg[0];
                    weight_grads[out*ldW + in + 1] += reduce_reg[1];
                    weight_grads[out*ldW + in + 2] += reduce_reg[2];
                    weight_grads[out*ldW + in + 3] += reduce_reg[3];
                    weight_grads[out*ldW + in + 4] += reduce_reg[4];
                    weight_grads[out*ldW + in + 5] += reduce_reg[5];
                    weight_grads[out*ldW + in + 6] += reduce_reg[6];
                    weight_grads[out*ldW + in + 7] += reduce_reg[7];
                    W_checksum += sum;
                    snrt_ssr_enable();

                    in += 8;
                }
            }


            snrt_ssr_disable();
        }
    }


    

    printf("GRADIENT UPDATE FP8 SIMD with SSRs: W_checksum[%u] = %f\n", compute_id, W_checksum);
    printf("GRADIENT UPDATE FP8 SIMD with SSRs: b_checksum[%u] = %f\n", compute_id, b_checksum);

    snrt_cluster_hw_barrier();

}


//// Activation Step
void softmax_activation_fp8_ex(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *activations, uint32_t ldB,
                char *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, float *max){

    // char max_core;
    register float max_core;
    float sum = 0.0f;
    char *act_ptr = activations[0];
    v4f16 sum_m = {0.0, 0.0, 0.0, 0.0};
    v4f16 sum_s = {0.0, 0.0, 0.0, 0.0};
    v4f16 sum_c0 = {0.0, 0.0, 0.0, 0.0};
    float max_core_f = 0.0f;
    float curr_activation = 0.0f;
    float curr_activation_c0 = 0.0f;
    

    v8f8 acts = {activations[0], 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000};
    v8f8 one_vec = {0b00000001, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000};

    // First of all we need to expand the activations to float precision

    asm volatile(
        "vfdotpex.h.b     ft3, %[acts], %[one_vec] \n"
        "vfsumex.s.h      %[sum_m], ft3 \n"
        "vfsum.s          %[max_core_f], %[sum_m] \n"
        : [acts] "+&f"(acts), [one_vec] "+&f"(one_vec), [sum_m] "+&f"(sum_m), [max_core_f] "+&f"(max_core_f)
        :
        : "ft0", "ft1", "ft2"
    );

    printf("max_core = %f\n", max_core_f);

    // max_core = activations[0];

    volatile uint32_t idx_eff; 

    for(uint32_t out = 0; out < OUT_CH; out++){
        idx_eff = compute_id + ldB * out;
        if(!(idx_eff > OUT_CH * 5 - 1)){ 
            v8f8 curr_act = {activations[out * ldB], 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000};
            asm volatile(
                "vfdotpex.h.b     ft4, %[curr_act], %[one_vec] \n"
                "vfsumex.s.h      %[sum_s], ft4 \n"
                "vfsum.s          %[curr_activation], %[sum_s] \n"
                : [curr_act] "+&f"(curr_act), [one_vec] "+&f"(one_vec), [sum_s] "+&f"(sum_s), [curr_activation] "+&f"(curr_activation)
                :
                : "ft0", "ft1", "ft2"
            );
            // printf("curr_activation = %f\n", curr_activation);
            // check if max_core is infinity
            if(!isinf(max_core) & !isnan(max_core) && !isinf(curr_activation) && !isnan(curr_activation)){
                if(curr_activation > max_core){
                    max_core = curr_activation;
                }
            }
            else{
                printf("ERROR: over/underflow\n");
                max_core = -10000.0f;
            }
        }
    }

    max[compute_id] = max_core;
    
    snrt_cluster_hw_barrier();

//     //printf("Max value of compute core %u is %f\n", compute_id, max_core);

    float max_global = max[0];

    // Reduction on single core
    if(compute_id == 0){
        for(uint32_t core = 0; core < compute_num; core++){
            if(max[core] > max_global){
                max_global = max[core];
            }
        }

        printf("Max value of all cores is %f\n", max_global);

        // FIXME: actually OUT_CH should be multiplied by number of compute cores
        for(uint32_t out = 0; out < OUT_CH*5; out++){
            act_ptr = &activations[0];
            // printf("act[%u] = %d\n", out + 1 + compute_id, activations[out * ldB]);
            v8f8 curr_act_c0 = {activations[out * ldB], 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000};
            asm volatile(
                "vfdotpex.h.b     ft5, %[curr_act_c0], %[one_vec] \n"
                "vfsumex.s.h      %[sum_c0], ft5 \n"
                "vfsum.s          %[curr_activation_c0], %[sum_c0] \n"
                : [curr_act_c0] "+&f"(curr_act_c0), [one_vec] "+&f"(one_vec), [sum_c0] "+&f"(sum_c0), [curr_activation_c0] "+&f"(curr_activation_c0)
                :
                : "ft0", "ft1", "ft2"
            );
            // printf("DEBUG: curr_activation_c0[%u] = %f\n", out + 1 + compute_id, curr_activation_c0);
            if(curr_activation_c0 != 0.0f & !isinf(curr_activation_c0) && !isnan(curr_activation_c0)){
                // printf("DEBUG NON ZERO: curr_activation_c0[%u] = %f\n", out + 1 + compute_id, curr_activation_c0);
                // printf("DEBUG: curr_activation_c0 - max_global = %f\n", curr_activation_c0 - max_global);
                curr_activation_c0 = my_exp(curr_activation_c0 - max_global);
                // printf("DEBUG after EXP: curr_activation_c0[%u] = %f\n", out + 1 + compute_id, curr_activation_c0);
                // printf("DEBUG: act_ptr[%u] = %d\n", out + 1 + compute_id, act_ptr[out]);
                if(!isinf(curr_activation_c0) && !isnan(curr_activation_c0)){
                    sum += curr_activation_c0;
                }
                //printf("DEBUG: sum = %f\n", sum);
            } else {
                //printf("DEBUG ZERO: act_ptr[%u] = %f\n", out, act_ptr[out]);
                curr_activation_c0 = 0.0f;
            }
        }


        for(uint32_t out = 0; out < OUT_CH*5; out++){
            curr_activation_c0 /= sum;
            printf("SOFTMAX FP8 (expanding): activation[%u] = %f\n", out + 1, curr_activation_c0);
            // activations[out] = act_ptr[out];
            // printf("SOFTMAX FP8 (no SIMD): activation[%u] = ", out + 1);
            // print_byte(activations[out]);
            // printf(" = %d\n", activations[out]);
            // printf(" = ");
            // get_float_from_byte(activations[out]);
        }
    }

    snrt_cluster_hw_barrier();
}