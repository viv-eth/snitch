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
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id, uint32_t* core_sync); //WORKS

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