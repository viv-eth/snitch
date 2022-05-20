// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "layer.h"

/**
 * @brief SoftMax layer SoftMax(IN_i) = exp(IN_i)/sum_j[exp(x_j)]
 *
 * @param l softmax_layer struct that holds addresses and parameters
 */
void softmax_layer(const sm_layer *l, void *l_checksum);