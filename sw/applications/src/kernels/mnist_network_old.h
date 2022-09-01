// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>


// INFO: Function calls for FP64 baseline network 

/**
 * @brief implementation of the feedforward part of a simple neural network for FP64
 * @param IN_CH1 first input dimension
 * @param IN_CH2 second input dimension
 * @param OUT_CH output dimension
 * @param weights initial weights of the network (obtained from GM)
 * @param ldW row stride of the weights matrix
 * @param activations of the NN
 * @param biases initial biases of the network (obtained from GM)
 * @param ldB row stride of the bias matrix
 * @param image input data (MNIST image)
 * @param ldI row stride of the image INFO: (not really necessary since it's a single row...)
 */

void feedforward_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double* activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id); //WORKS

/**
 * @brief implementation of the softmax activation for FP64
 * @param IN_CH1 first input dimension
 * @param IN_CH2 second input dimension
 * @param OUT_CH output dimension
 * @param weights neural network weights
 * @param ldW row stride of the weights matrix
 * @param activations of the NN
 * @param biases neural network biases
 * @param ldB row stride of the bias matrix
 * @param image input data (MNIST image)
 * @param ldI row stride of the image INFO: (not really necessary since it's a single row...)
 * @param compute_id ID of the current compute core
 * @param compute_num Number of total compute cores
 * @param max pointer to the memory location containing the maxima of each compute core
 */

void softmax_activation_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max, uint32_t* core_sync);


//TODO: add description
void gradient_update_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, uint32_t ldI, uint32_t compute_id, 
                double *loss, uint32_t compute_num);

//TODO: add description
void training_step_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, 
                double *biases, double *bias_grads, uint32_t ldB, 
                uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images);


// INFO: Function calls for FP64 network with SSRs
void feedforward_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync,
                uint32_t setup_SSR);

void softmax_activation_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max, uint32_t* core_sync, uint32_t setup_SSR);

void gradient_update_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, double *loss, uint32_t compute_num, uint32_t setup_SSR);

void training_step_fp64_ssr(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR);

// INFO: Function calls for FP32 network with SSRs and SIMD instructions
void feedforward_fp32_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *biases, float *activations,
                uint32_t ldB, float *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync,
                uint32_t setup_SSR);

void softmax_activation_fp32(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *activations, uint32_t ldB,
                float *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, float *max, uint32_t* core_sync);

void gradient_update_fp32_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weight_grads, uint32_t ldW, float *bias_grads, float *activations, 
                uint32_t ldB, float *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, float *loss, uint32_t compute_num, uint32_t setup_SSR);

void training_step_fp32_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, float *weight_grads, uint32_t ldW, float *biases, float *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR);

// INFO Function calls for FP32 WITHOUT SSRs and SIMD
void feedforward_fp32(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, uint32_t ldW, float *biases, float* activations,
                uint32_t ldB, float *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync);


void gradient_update_fp32(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weight_grads, uint32_t ldW, float *bias_grads, float *activations, 
                uint32_t ldB, float *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, float *loss, uint32_t compute_num);

void training_step_fp32(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                float *weights, float *weight_grads, uint32_t ldW, float *biases, float *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images);


// INFO Function calls for FP16 using SSRs and SIMD
void feedforward_fp16_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *biases, __fp16 *activations,
                uint32_t ldB, __fp16 *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync,
                uint32_t setup_SSR);

void gradient_update_fp16_ssr_simd(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weight_grads, uint32_t ldW, __fp16 *bias_grads, __fp16 *activations, 
                uint32_t ldB, __fp16 *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, __fp16 *loss, uint32_t compute_num, uint32_t setup_SSR);

// INFO Function calls for FP16 WITHOUT SSRs and SIMD
void feedforward_fp16(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *biases, __fp16 *activations,
                uint32_t ldB, __fp16 *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync);

void softmax_activation_fp16(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, uint32_t ldW, __fp16 *activations, uint32_t ldB,
                __fp16 *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, __fp16 *max, uint32_t* core_sync);

void gradient_update_fp16(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weight_grads, uint32_t ldW, __fp16 *bias_grads, __fp16 *activations, 
                uint32_t ldB, __fp16 *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, __fp16 *loss, uint32_t compute_num);

void training_step_fp16(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                __fp16 *weights, __fp16 *weight_grads, uint32_t ldW, __fp16 *biases, __fp16 *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images);

// INFO Function calls for FP8 baseline
void feedforward_fp8(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *biases, char *activations,
                uint32_t ldB, char *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync);

void softmax_activation_fp8(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *activations, uint32_t ldB,
                char *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, char *max, uint32_t* core_sync);

void gradient_update_fp8(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weight_grads, uint32_t ldW, char *bias_grads, char *activations, 
                uint32_t ldB, char *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, char *loss, uint32_t compute_num);

void training_step_fp8(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, char *weight_grads, uint32_t ldW, char *biases, char *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images);