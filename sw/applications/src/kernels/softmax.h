// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

/**
 * @brief implementation of a FP64 softmax activation
 * @param l SoftMax layer struct //TODO: remove from function call, not needed
 * @param dim1 First dimension of the input matrix
 * @param dim2 Second dimension of the input matrix
 * @param IN Input slice for each core
 * @param ldIn Row stride in matrix IN
 * @param compute_id ID of the compute core
 * @param max Pointer to where cores store their max values
 */
 
void softmax_fp64(const sm_layer *l, uint32_t dim1, uint32_t dim2, double* IN, uint32_t ldIn, uint32_t compute_id, double *max);

/**
 * @brief implementation of a FP64 with SSRs softmax activation (uses clang SSR inbuilt functions)
 * @param l SoftMax layer struct TODO: check how to account properly for parallelization when reducing on single core
 * @param dim1 First dimension of the input matrix
 * @param dim2 Second dimension of the input matrix
 * @param IN Input slice for each core
 * @param ldIn Row stride in matrix IN
 * @param compute_id ID of the compute core
 * @param max Pointer to where cores store their max values
 * @param SSR_setup Whether or not to setup SSRs //TODO: add inside function, not used yet
 */
void softmax_fp64_ssr(const sm_layer *l, uint32_t dim1, uint32_t dim2, double* IN, uint32_t ldIn, uint32_t compute_id, double *max, uint32_t SSR_setup);

/**
 * @brief implementation of a FP32 softmax activation with SSRs
 * @param dim1 First dimension of the input matrix
 * @param dim2 Second dimension of the input matrix
 * @param IN Input slice for each core
 * @param ldIn Row stride in matrix IN
 * @param compute_id ID of the compute core
 * @param max Pointer to where cores store their max values
 * @param SSR_setup Whether or not to setup SSRs //TODO: add inside function, not used yet
 */
void softmax_fp32_ssr(uint32_t dim1, uint32_t dim2, float* IN, uint32_t ldIn, uint32_t compute_id, float *max, uint32_t SSR_setup);

/** 
 * @brief implementation of a FP16 with SSRs softmax activation
 * @param dim1 First dimension of the input matrix
 * @param dim2 Second dimension of the input matrix
 * @param IN Input slice for each core
 * @param ldIn Row stride in matrix IN
 * @param compute_id ID of the compute core
 * @param max Pointer to where cores store their max values
 * @param SSR_setup Whether or not to setup SSRs //TODO: add inside function, not used yet
 */
void softmax_fp16(uint32_t dim1, uint32_t dim2, __fp16* IN, uint32_t ldIn, uint32_t compute_id, __fp16 *max, uint32_t SSR_setup);