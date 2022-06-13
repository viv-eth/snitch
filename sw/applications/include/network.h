// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "layer.h"

//typedef enum { FP64 = 8, FP32 = 4, FP16 = 2, FP8 = 1 } precision_t;

/**
 * @struct network_t_
 * @brief This structure contains all parameters necessary for building a simple neural netowork.
 * @var network_t_::IN_CH1
 * First dimension of the input data matrix (first channel)
 * @var network_t_::IN_CH2
 * Second dimension of the input data matrix (second channel)
 * @var network_t_::OUT_CH
 * Dimension of input matix along which we perform SoftMax
 * @var network_t_::b
 * Pointer to biases of the network
 * @var network_t_::W
 * Pointer to weights of the network
 * @var network_t_::b_grad
 * Pointer to bias gradients of the network
 * @var network_t_::W_grad
 * Pointer to weight gradients of the network
 * @var network_t_::dtype
 * Precision of the neural network (uniform for now)
 */

typedef struct network_t_ {
    uint32_t IN_CH1;
    uint32_t IN_CH2;
    uint32_t OUT_CH;

    float *b;
    float *W;
    float *b_grad;
    float *W_grad;

    // double *b;
    // double *W;
    // double *b_grad;
    // double *W_grad;

    // double *images;
    float *images;
    uint32_t *targets;

    precision_t dtype;
    
} network_t;
