// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SW testbench for profiling linear kernels in different
// floating point precisions (fp64, fp32, fp16), as well as
// different memory layouts for matrices (transposed/not-transposed)
// Correctness of results are checked automatically

#include "celoss_layer.h"
#include "data_celoss.h"
#include "layer.h"
#include "math.h"
#include "perf_cnt.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

// Padding of innermost dimension of a Matrix
// Useful for preventing banking conflicts between cores
// that are accessing different rows of the matrix
#define MAT_ROW_PADDING 4

// Padding in between matrices A, B for preventing
// banking conflicts in the beginning
#define MAT_PADDING 8

int main(){
    // load activation data from the golden model
    celoss_l.IN = (void *)celoss_fwout_dram;
    
    celoss_layer(&celoss_l, &celoss_checksum);

    snrt_global_barrier();

    return 0;
}