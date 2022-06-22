// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SW testbench for profiling Conv2d Layer
// Automatically checks the correctness of the results

#include "mnist_cnn.h"
#include "data_cnn_mnist.h"
#include "layer.h"
#include "math.h"
#include "perf_cnt.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

int main() {

    mnist_cnn_t.image = (double*)mnist_cnn_image_dram;
    mnist_cnn_t.conv1_weights = (double*)mnist_cnn_conv1_weights_dram;
    mnist_cnn_t.conv1_biases = (double*)mnist_cnn_conv1_bias_dram;
    
    mnist_cnn(&mnist_cnn_t);

    // snrt_global_barrier();
    // INFO: replacing global barrier with custom barrier for RTL sims
    // TODO: check whether this is necessary
    // uint32_t cluster_num = snrt_cluster_num();
    // uint32_t cluster_core_num = snrt_cluster_core_num();
    // snrt_generic_cluster_barrier(cluster_num*cluster_core_num);

    return 0;

}