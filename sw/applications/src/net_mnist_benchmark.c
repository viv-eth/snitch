// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SW testbench for profiling linear kernels in different
// floating point precisions (fp64, fp32, fp16), as well as
// different memory layouts for matrices (transposed/not-transposed)
// Correctness of results are checked automatically

#include "mnist_benchmark.h"
#include "network.h"
#include "math.h"
#include "perf_cnt.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

#define PREC 16
#define HIGH_DIM 0

# if PREC == 64
    #include "data_fp64_benchmark.h" //--> For FP64 tests
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
    mnist_t.b = (void *)mnist_biases_dram;

    // NOTE At the moment we are using five MNIST images only
    // for simulation purposes
    mnist_t.images = (void *)mnist_images_dram;
    mnist_t.targets = (void *)mnist_labels_dram;
    
    mnist_benchmark(&mnist_t);
    
    snrt_global_barrier();

    return 0;
}