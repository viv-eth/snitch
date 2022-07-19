// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

void feedforward_fp64n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double* activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id);

void softmax_activation_fp64n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max);

void gradient_update_fp64n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, double *loss, uint32_t compute_num);

void training_step_fp64n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images);

void feedforward_fp64_ssrn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *biases, double *activations,
                uint32_t ldB, double *image, uint32_t ldI, uint32_t compute_id,
                uint32_t setup_SSR);

void softmax_activation_fp64_ssrn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, uint32_t ldW, double *activations, uint32_t ldB,
                double *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, double *max, uint32_t setup_SSR);

void gradient_update_fp64_ssrn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weight_grads, uint32_t ldW, double *bias_grads, double *activations, 
                uint32_t ldB, double *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, double *loss, uint32_t compute_num, uint32_t setup_SSR);

void training_step_fp64_ssrn(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                double *weights, double *weight_grads, uint32_t ldW, double *biases, double *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR);