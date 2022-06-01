// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "simple.h"

#include "printf.h"
#include "snrt.h"

typedef float v2f32 __attribute__((vector_size(8)));
typedef __fp16 v4f16 __attribute__((vector_size(8)));
typedef char v8f8 __attribute__((vector_size(8)));

void simple_fp64(uint32_t M, uint32_t N, uint32_t K, double* A, uint32_t ldA,
                double* B, uint32_t ldB, double* C, uint32_t ldC, uint32_t ALPHA, uint32_t compute_id) {
    
        for (uint32_t m = 0; m < M; m++) {
            for (uint32_t n = 0; n < N; n++) {
                register double c0 = ALPHA * C[m * ldC + n];
                // if(!compute_id){
                //     printf("init c0 = %f\n", c0 / 3);
                // }
                for (uint32_t k = 0; k < K; k++) {
                    c0 += A[k + m * ldA] * B[k * ldB + n];
                    // if(!compute_id){
                    //     printf("A[%u][%u] = %f\n", m, k, A[k + m * ldA]);
                    //     printf("B[%u][%u] = %f\n", k, n, B[k * ldB + n]);
                    //     printf("c0 = %f\n", c0);
                    // }
                }
                C[m * ldC + n] = c0;

            }
        }
} 


        

