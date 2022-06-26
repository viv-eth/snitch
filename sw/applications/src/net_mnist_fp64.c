// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SW testbench for profiling linear kernels in different
// floating point precisions (fp64, fp32, fp16), as well as
// different memory layouts for matrices (transposed/not-transposed)
// Correctness of results are checked automatically

#include "mnist_fp64.h"
#include "data_five_mnist.h" //--> For FP64 tests
#include "network.h"
#include "math.h"
#include "perf_cnt.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

int main(){

    mini_mnist_t.W = (void *)mini_mnist_weights_dram;
    mini_mnist_t.b = (void *)mini_mnist_biases_dram;

    // NOTE At the moment we are using five MNIST images only
    // for simulation purposes
    mini_mnist_t.images = (void *)mini_mnist_images_dram;
    mini_mnist_t.targets = (void *)mini_mnist_labels_dram;
    
    mnist_fp64(&mini_mnist_t);

    // snrt_global_barrier();
    // INFO: replacing global barrier with custom barrier for RTL sims
    uint32_t cluster_num = snrt_cluster_num();
    uint32_t cluster_core_num = snrt_cluster_core_num();
    snrt_generic_cluster_barrier(cluster_num*cluster_core_num);

    return 0;
}