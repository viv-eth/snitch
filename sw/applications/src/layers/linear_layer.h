// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "layer.h"

/**
 * @brief Simple linear layer performing y=W*X+B 
 *
 * @param l linear_layer struct that holds addresses and parameters
 * 
 */
void linear_layer(const lin_layer *l, void *l_checksum);
