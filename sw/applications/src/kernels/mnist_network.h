// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

/**
 * @brief implementation of the feedforward part of a simple neural network for FP64
 * @param IN_CH1 first input dimension
 * @param IN_CH2 second input dimension
 * @param OUT_CH output dimension
 * @param weights initial weights of the network (obtained from GM)
 * @param ldW row stride of the weights matrix
 * @param biases initial biases of the network (obtained from GM)
 * @param ldB row stride of the bias matrix
 * @param image input data (MNIST image)
 * @param ldI row stride of the image INFO: (not really necessary since it's a single row...)
 */

void feedforward_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id); //WORKS

/**
 * @brief implementation of the softmax activation for FP64
 * @param IN_CH1 first input dimension
 * @param IN_CH2 second input dimension
 * @param OUT_CH output dimension
 * @param weights neural network weights
 * @param ldW row stride of the weights matrix
 * @param biases neural network biases
 * @param ldB row stride of the bias matrix
 * @param image input data (MNIST image)
 * @param ldI row stride of the image INFO: (not really necessary since it's a single row...)
 * @param compute_id ID of the current compute core
 * @param compute_num Number of total compute cores
 * @param max pointer to the memory location containing the maxima of each compute core
 */

void softmax_activation_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max);


//TODO: add description & update arguments
void gradient_update_fp64(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *biases, 
                uint32_t ldB, double *image, uint32_t *target, uint32_t ldI, uint32_t compute_id, 
                double *loss, uint32_t compute_num);