// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SW testbench for profiling linear kernels in different
// floating point precisions (fp64, fp32, fp16), as well as
// different memory layouts for matrices (transposed/not-transposed)
// Correctness of results are checked automatically

#include "nnlinear_backend_fp16s.h"
#include "data_fp16_nnlinear.h"
#include "network.h"
#include "math.h"
#include "perf_cnt.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

int main(){

    nn_linear_fp16s_t.W = (void *)nn_linear_fp16s_weights_dram;
    nn_linear_fp16s_t.b = (void *)nn_linear_fp16s_biases_dram;
    
    nnlinear_backend_fp16s(&nn_linear_fp16s_t);

    // INFO: replacing global barrier with custom barrier for RTL sims
    uint32_t cluster_num = snrt_cluster_num();
    uint32_t cluster_core_num = snrt_cluster_core_num();

    // snrt_generic_cluster_barrier(cluster_num*cluster_core_num);
    snrt_global_barrier();

    return 0;
}