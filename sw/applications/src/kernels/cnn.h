// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "snrt.h"

// TODO: add description
void pad_image(double *image, double *padded_image, uint16_t H, uint16_t W, uint16_t padding);

// TODO: add description
void conv2d_fp64(double *padded_image, double *weights, double *biases, uint16_t ci, uint16_t co, 
                uint16_t H, uint16_t W, uint16_t K, uint16_t padding, uint16_t stride, 
                uint16_t dim_out_x, uint16_t dim_out_y, uint16_t row_stride, double *output);