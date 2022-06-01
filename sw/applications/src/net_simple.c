// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SW testbench for profiling linear kernels in different
// floating point precisions (fp64, fp32, fp16)

#include "math.h"
#include "perf_cnt.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"
#include "data_simple.h"

int main(){

    simple_l.A = (void *)simple_A_dram;
    simple_l.B = (void *)simple_B_dram;
    simple_l.C = (void *)simple_C_dram;

    simple_layer(&simple_l, &simple_checksum);

    snrt_global_barrier();

    return 0;
}
