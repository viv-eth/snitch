// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

void feedforward_fp8n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *biases, char *activations,
                uint32_t ldB, char *image, uint32_t ldI, uint32_t compute_id, float *activations_fp32);

void softmax_activation_fp8n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *activations, uint32_t ldB,
                char *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, char *max);

void gradient_update_fp8n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weight_grads, uint32_t ldW, char *bias_grads, char *activations, 
                uint32_t ldB, char *image, uint32_t *target, uint32_t ldI, 
                uint32_t compute_id, char *loss, uint32_t compute_num);

void training_step_fp8n(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, char *weight_grads, uint32_t ldW, char *biases, char *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images);

void feedforward_fp8n_opt(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *biases, char *activations,
                uint32_t ldB, char *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t setup_SSR, float *activations_fp32);

void softmax_activation_fp8_ex(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, uint32_t ldW, char *activations, uint32_t ldB,
                char *image, uint32_t ldI, uint32_t compute_id, 
                uint32_t compute_num, float *max);

void softmax_activation_fp32_ex(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH,
                float *activations_fp32, char *activations, uint32_t ldB, uint32_t compute_id, 
                uint32_t compute_num, float *max);

void gradient_update_fp8n_opt(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                        char *weight_grads, uint32_t ldW, char *bias_grads,
                        float *activations_fp32, uint32_t ldB, char *image, 
                        char *target, uint32_t ldI, uint32_t compute_id, 
                        char *loss, uint32_t compute_num, uint32_t setup_SSR);

void training_step_fp8_opt(uint32_t IN_CH1, uint32_t IN_CH2, uint32_t OUT_CH, 
                char *weights, char *weight_grads, uint32_t ldW, char *biases, char *bias_grads,
                uint32_t ldB, uint32_t compute_id, uint32_t compute_num,
                uint32_t number_of_images, uint32_t setup_SSR);

    