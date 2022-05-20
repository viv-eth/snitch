// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

//TODO: add description
void celoss_fp64(const cel_layer *l, uint32_t dim1, uint32_t dim2, 
                 uint32_t dim3, double* IN, uint32_t ldIn, 
                 uint32_t compute_id, double* sum); // INFO: works

//TODO: add description
void celoss_fp64_ssr(const cel_layer *l, uint32_t dim1, uint32_t dim2, 
                 uint32_t dim3, double* IN, uint32_t ldIn, 
                 uint32_t compute_id, double* sum);

//TODO: add description
void celoss_fp32(const cel_layer *l, uint32_t dim1, uint32_t dim2, 
                 uint32_t dim3, float* IN, uint32_t ldIn, 
                 uint32_t compute_id, float* sum); 

//TODO: add description
void celoss_fp16(const cel_layer *l, uint32_t dim1, uint32_t dim2, 
                 uint32_t dim3, __fp16* IN, uint32_t ldIn, 
                 uint32_t compute_id, __fp16* sum); 