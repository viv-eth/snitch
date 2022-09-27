// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "single_cluster.h"
#include "network.h"
#include "math.h"
#include "perf_cnt.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

// define in which precision to run the network
#define PREC 64

// define whether to use MNIST dataset size or full TCDM size
#define HIGH_DIM 0

// define which kernel to run
#define FEEDFORWARD 0
#define GRADIENT_UPDATE 0
#define TRAINING_STEP 1

# if PREC == 64
    # if HIGH_DIM == 0
        #include "data_fp64_all_mnist.h"
    # endif
# elif PREC == 32

    # if HIGH_DIM == 0
        #include "data_fp32_benchmark.h" //--> For FP32 tests
    # else
        #include "data_fp32_benchmark_high_dim.h" //--> For FP32 tests
    # endif
# elif PREC == 16
    # if HIGH_DIM == 0
        #include "data_fp16_benchmark.h" //--> For FP16 tests
    # else
        #include "data_fp16_benchmark_high_dim.h" //--> For FP16 tests
    # endif
# elif PREC == 8
    # if HIGH_DIM == 0
        #include "data_fp8_benchmark.h" //--> For FP8 tests
    # else
        #include "data_fp8_benchmark_high_dim.h" //--> For FP8 tests
    # endif
# endif

int main(){

    mnist_t.W = (void *)mnist_weights_dram;
    mnist_t.W_grads = (void *)mnist_weight_grads_dram;
    mnist_t.b = (void *)mnist_biases_dram;
    mnist_t.b_grads = (void *)mnist_bias_grads_dram;

    // NOTE At the moment we are using five MNIST images only
    // for simulation purposes
    mnist_t.images = (void *)mnist_images_dram;
    mnist_t.targets = (void *)mnist_labels_dram;
    
    single_cluster(&mnist_t);
    
    snrt_global_barrier();

    return 0;
}