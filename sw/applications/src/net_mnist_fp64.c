// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SW testbench for profiling linear kernels in different
// floating point precisions (fp64, fp32, fp16), as well as
// different memory layouts for matrices (transposed/not-transposed)
// Correctness of results are checked automatically

#include "mnist_fp64.h"
#include "data_fp64_mnist.h" //--> For FP64 tests
#include "network.h"
#include "math.h"
#include "perf_cnt.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

int main(){

    mnist_t.W = (void *)mnist_weights_dram;
    mnist_t.b = (void *)mnist_biases_dram;

    // NOTE At the moment we are using five MNIST images only
    // for simulation purposes
    mnist_t.images = (void *)mnist_images_dram;
    mnist_t.targets = (void *)mnist_labels_dram;
    
    mnist_fp64(&mnist_t);

    snrt_global_barrier();

    return 0;
}