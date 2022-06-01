// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// TODO: add description
void simple_fp64(uint32_t M, uint32_t N, uint32_t K, double* A, uint32_t ldA,
                double* B, uint32_t ldB, double* C, uint32_t ldC, uint32_t ALPHA, uint32_t compute_id);