// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

/**
 * @brief naive implementation of a FP64 linear layer
 *
 * @param M number of rows of matrix W (weights)
 * @param N number of columns of matrix X (input data)
 * @param K number of columns of matrix W (weights)
 * @param W pointer to weight matrix W
 * @param ldW row stride in matrix W
 * @param tw transposed memory layout for matrix W
 * @param X pointer to input data matrix/vector X
 * @param ldX row stride in input data matrix X
 * @param tx transposed memory layout for matrix X
 * @param B pointer to bias matrix B
 * @param tb transposed memory layout for matrix B
 * @param ldB row stride in bias matrix B
 */
 
void linear_fp64(uint32_t M, uint32_t N, uint32_t K, double* W, uint32_t ldW,
               uint32_t tw, double* X, uint32_t ldX, uint32_t tx, double* B, uint32_t tb,
               uint32_t ldB);

/**
 * @brief Implementation of a FP64 linear layer with SSR and FREP
 *
 * @param M number of rows of matrix W (weights)
 * @param N number of columns of matrix X (input data)
 * @param K number of columns of matrix W (weights)
 * @param W pointer to weight matrix W
 * @param ldW row stride in matrix W
 * @param tw transposed memory layout for matrix W
 * @param X pointer to input data matrix/vector X
 * @param ldX row stride in input data matrix X
 * @param tx transposed memory layout for matrix X
 * @param B pointer to bias matrix B
 * @param tb transposed memory layout for matrix B
 * @param ldB row stride in bias matrix B
 * @param setup_SSR setup SSR bounds and strides
 */

void linear_fp64_ssr_frep(uint32_t M, uint32_t N, uint32_t K, double* W,
                        uint32_t ldW, uint32_t tw, double* X, uint32_t ldX,
                        uint32_t tx, double* B, uint32_t ldB, uint32_t tb, 
                        uint32_t setup_SSR);

/**
 * @brief Implementation of a FP32 linear layer with SSR and FREP
 *
 * @param M number of rows of matrix W (weights)
 * @param N number of columns of matrix X (input data)
 * @param K number of columns of matrix W (weights)
 * @param W pointer to weight matrix W
 * @param ldW row stride in matrix W
 * @param tw transposed memory layout for matrix W
 * @param X pointer to input data matrix/vector X
 * @param ldX row stride in input data matrix X
 * @param tx transposed memory layout for matrix X
 * @param B pointer to bias matrix B
 * @param tb transposed memory layout for matrix B
 * @param ldB row stride in bias matrix B
 * @param setup_SSR setup SSR bounds and strides
 */

void linear_fp32simd_ssr_frep(uint32_t M, uint32_t N, uint32_t K, float* W,
                        uint32_t ldW, uint32_t tw, float* X, uint32_t ldX,
                        uint32_t tx, float* B, uint32_t ldB, uint32_t tb, 
                        uint32_t setup_SSR);

/**
 * @brief Implementation of a FP16 linear layer with SSR and FREP
 *
 * @param M number of rows of matrix W (weights)
 * @param N number of columns of matrix X (input data)
 * @param K number of columns of matrix W (weights)
 * @param W pointer to weight matrix W
 * @param ldW row stride in matrix W
 * @param tw transposed memory layout for matrix W
 * @param X pointer to input data matrix/vector X
 * @param ldX row stride in input data matrix X
 * @param tx transposed memory layout for matrix X
 * @param B pointer to bias matrix B
 * @param tb transposed memory layout for matrix B
 * @param ldB row stride in bias matrix B
 * @param setup_SSR setup SSR bounds and strides
 */

void linear_fp16simd_ssr_frep(uint32_t M, uint32_t N, uint32_t K, __fp16* W,
                        uint32_t ldW, uint32_t tw, __fp16* X, uint32_t ldX,
                        uint32_t tx, __fp16* B, uint32_t ldB, uint32_t tb, 
                        uint32_t setup_SSR);