// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

# pragma once

#include "math.h"

#include "printf.h"
#include "snrt.h"
#include "utils.h"

typedef float v2f32 __attribute__((vector_size(8)));
typedef __fp16 v4f16 __attribute__((vector_size(8)));
typedef char v8f8 __attribute__((vector_size(8)));

typedef union {
    double f64;
    v2f32 vec;
} v2s;
typedef union {
    double f64;
    v4f16 vec;
} v4s;
typedef union {
    double f64;
    v8f8 vec;
} v8s;

/**
 * Baseline kernels for a single core execution
*/

#define NUM_CLASSES 10
#define IN_CH 784
#define BATCH_SIZE 256

/**
 * SoftMax calculation 
*/

static inline void SoftMax_fp16s(__fp16 *activations, int length) {
// int length = LEN(activations);
    printf("============= SoftMax feedforward start =============\n");
    float sum = 0;
    __fp16 max = activations[0];
    int correct, predict = 0;

    // Get the maximum value of all activations
    for (int i = 1; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    // normalize 
    for (int i = 0; i < length; i++) {
        activations[i] = exp(activations[i]- max);
        sum += activations[i];
    }

    // compute softmax activations
    for (int i = 0; i < length; i++) {
        activations[i] /= sum;
        printf("activations[%d] = %f\n", i, activations[i]);
    }

    printf("============= SoftMax feedforward end =============\n");

    // snrt_cluster_hw_barrier();
}

/**
 * FeedForward calculation 
*/

static inline void FeedForward_fp16s(__fp16 *image, __fp16 *activations, __fp16 *biases, __fp16 *weights) {

    printf("============= Feedforward pass start =============\n");

    // float checksum = 0;
    // float img_checksum = 0;
    // float weight_checksum = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        // printf("biases[%d] = %f\n", i, biases[i]);
        activations[i] = biases[i];
        for (int j = 0; j < IN_CH; j++) {
            // printf("image[%d] = %f\n", j, image[j]);
            // img_checksum += image[j];
            // weight_checksum += weights[i * IN_CH + j];
            __fp16 tmp = activations[i];
            activations[i] += weights[i * IN_CH + j] * image[j];
            if (activations[i] > 100 || activations[i] < -100) {
                printf("weights[%d] = %f\n", i * IN_CH + j, weights[i * IN_CH + j]);
                // print weights in hex
                uint32_t *p = (uint32_t *) &weights[i * IN_CH + j];
                printf("weights[%d] = %08x\n", i * IN_CH + j, *p);
                printf("image[%d] = %f\n", j, image[j]);
                uint32_t *p2 = (uint32_t *) &image[j];
                printf("image[%d] = %08x\n", j, *p2);
                printf("activations[%d] = %f\n", i-1, tmp);
                uint32_t *p3 = (uint32_t *) &tmp;
                printf("activations[%d] = %08x\n", i-1, *p3);
                tmp += weights[i * IN_CH + j] * image[j];
                printf("tmp = %f\n", tmp);
                printf("activations[%d] = %f\n", i, activations[i]);
                uint32_t *p4 = (uint32_t *) &activations[i];
                printf("activations[%d] = %08x\n", i, *p4);
                printf("weights[%d] * image[%d] = %f\n", i * IN_CH + j, j, (weights[i * IN_CH + j] * image[j]));
            }
        }

        // if(activations[i] > 100 || activations[i] < -100) {
        //     printf("biases[%d] = %f\n", i, biases[i]);
        //     for (int j = 0; j < IN_CH; j++) {
        //         printf("weights[%d] = %f\n", i * IN_CH + j, weights[i * IN_CH + j]);
        //     }
        // }

        // checksum += activations[i];

        printf("activations[%d] = %f\n", i, activations[i]);
    }
    
    // printf("Activation checksum = %f\n", checksum);
    // printf("Image FeedForward checksum = %f\n", img_checksum);
    // printf("Weight FeedForward checksum = %f\n", weight_checksum);

    printf("============= Feedforward pass end =============\n");

    // snrt_cluster_hw_barrier();
    
    SoftMax_fp16s(activations, NUM_CLASSES);
}

/**
 * Gradient update calculation
*/

static inline void GradientUpdate_fp16s(
            __fp16 *image, __fp16 *activations, __fp16 *biases, 
            __fp16 *weights, __fp16 *W_gradients, __fp16 *b_gradients,
            uint32_t label, __fp16 *loss) {
    

    FeedForward_fp16s(image, activations, biases, weights);

    loss[0] = 0.0f - log(activations[label]);
    printf("loss = %f, label = %u, activation = %f\n", loss[0], label, activations[label]);
    
    snrt_cluster_hw_barrier();
    
    __fp16 b_grad, W_grad;
    for (int i = 0; i < NUM_CLASSES; i++) {
        b_grad = (i == label) ? (activations[i] - 1) : activations[i];
        for (int j = 0; j < IN_CH; j++) {
            W_grad = b_grad * image[j];
            W_gradients[i * IN_CH + j] += W_grad;
        }

        b_gradients[i] += b_grad;
    }
    
    // return loss;
    snrt_cluster_hw_barrier();
    
}


/**
 * Training step calculation
*/

static inline void TrainingStep_fp16s(
            __fp16 *biases, __fp16 *weights, __fp16 *W_gradients, __fp16 *b_gradients,
            float learning_rate) {

    float b_checksum = 0;
    float W_checksum = 0;

    for(int i = 0; i < NUM_CLASSES; i++) {
        // printf("biases before [%d] = %f\n", i, biases[i]);
        // printf("biases gradients [%d] = %f\n", i, b_gradients[i]);
        biases[i] -= learning_rate * b_gradients[i] / BATCH_SIZE;
        // printf("biases after [%d] = %f\n", i, biases[i]);
        // b_grad_checksum += b_gradients[i];
        b_checksum += biases[i];
        for(int j = 0; j < IN_CH; j++) {
            weights[i * IN_CH + j] -= learning_rate * W_gradients[i * IN_CH + j] / BATCH_SIZE;
            W_checksum += weights[i * IN_CH + j];
            // W_grad_checksum += W_gradients[i * IN_CH + j];
        }
    }

    printf("b_checksum = %f\n", b_checksum);
    printf("W_checksum = %f\n", W_checksum);
    // printf("b_grad_checksum = %f\n", b_grad_checksum);
    // printf("W_grad_checksum = %f\n", W_grad_checksum);

    snrt_cluster_hw_barrier();
}