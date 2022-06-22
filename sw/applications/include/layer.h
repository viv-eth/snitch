// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

typedef enum { FP64 = 8, FP32 = 4, FP16 = 2, FP8 = 1 } precision_t;

/**
 * @struct gemm_layer_struct
 * @brief This structure contains all parameters necessary for GEMM.
 * @var gemm_layer_struct::M
 * Dimension of matrix product MxK * KxN
 * @var gemm_layer_struct::M_p
 * M divided by number of compute cores
 * @var gemm_layer_struct::N
 * Dimension of matrix product MxK * KxN
 * @var gemm_layer_struct::K
 * Dimension of matrix product MxK * KxN
 * @var gemm_layer_struct::TA
 * Transpose matrix A
 * @var gemm_layer_struct::TB
 * Transpose matrix B
 * @var gemm_layer_struct::TILE_M
 * Tile factor across M dimension
 * @var gemm_layer_struct::TILE_N
 * Tile factor across N dimension
 * @var gemm_layer_struct::TILE_K
 * Tile factor across K dimension
 * @var gemm_layer_struct::A
 * Pointer to matrix A
 * @var gemm_layer_struct::B
 * Pointer to matrix B
 * @var gemm_layer_struct::C
 * Pointer to matrix C
 * @var gemm_layer_struct::ALPHA
 * constant factor: A * B + ALPHA * C
 * @var gemm_layer_struct::dtype
 * Precision of GEMM
 * @var gemm_layer_struct::expand
 * Use expanding DOTP instructions
 */
typedef struct gemm_layer_struct {
    uint32_t M;
    uint32_t M_p;
    uint32_t N;
    uint32_t K;

    uint32_t TA;
    uint32_t TB;

    uint32_t TILE_M;
    uint32_t TILE_N;
    uint32_t TILE_K;

    double *A;
    double *B;
    double *C;

    uint32_t ALPHA;

    precision_t dtype;
    uint32_t expand;
} gemm_layer;

/**
 * @struct linear_layer_struct
 * @brief This structure contains all parameters necessary for a linear layer.
 * The structure is based on the GEMM layer (which is a generalized linear layer)
 * @var linear_layer_struct::M
 * Dimension of matrix product MxK * KxN + MxN
 * @var linear_layer_struct::K
 * Dimension of matrix product MxK * KxN + MxN
 * @var linear_layer_struct::N
 * Dimension of matrix product MxK * KxN + MxN
 * @var linear_layer_struct::K
 * Dimension of matrix product MxK * KxN + MxN
 * @var linear_layer_struct::TW
 * Transpose matrix W
 * @var linear_layer_struct::TX
 * Transpose matrix X
 * @var linear_layer_struct::TB
 * Transpose matrix B
 * @var linear_layer_struct::W
 * Pointer to matrix W (weights)
 * @var linear_layer_struct::X
 * Pointer to matrix X (inputs)
 * @var linear_layer_struct::B
 * Pointer to matrix B (biases)
 * @var linear_layer_struct::dtype
 * Precision of the linear layer
 */
typedef struct linear_layer_struct {
    uint32_t M;
    uint32_t N;
    uint32_t K;

    uint32_t TW;
    uint32_t TX;
    uint32_t TB;

    double *W;
    double *X;
    double *B;
    precision_t dtype;
    
} lin_layer;

// TODO: add description

typedef struct simple_layer_struct {
    
    uint32_t M;
    uint32_t N;
    uint32_t K;

    double *A;
    double *B;
    double *C;

    uint32_t ALPHA;

    precision_t dtype;
    
} simpl_layer;

/**
 * @struct softmax_layer_struct
 * @brief This structure contains all parameters necessary for a softmax layer.
 * @var softmax_layer_struct::dim1
 * First dimension of the input data matrix (and output data matrix)
 * @var softmax_layer_struct::dim2
 * Second dimension of the input data matrix (and output data matrix)
 * @var softmax_layer_struct::D
 * Dimension of input matix along which we perform SoftMax
 * @var softmax_layer_struct::IN
 * Pointer to input data matrix
 * @var softmax_layer_struct::dtype
 * Precision of the softmax layer
 */
typedef struct softmax_layer_struct {
    uint32_t dim1;
    uint32_t dim2;
    uint32_t D;

    double *IN;
    double *OUT;
    precision_t dtype;
    
} sm_layer;

//TODO: add description here
typedef struct celoss_layer_struct {
    uint32_t IN_CH1;
    uint32_t IN_CH2;
    uint32_t OUT_CH;

    uint32_t target;

    double *IN;
    precision_t dtype;
    
} cel_layer;

//TODO: add description here
typedef struct sgd_bw_layer_struct {
    // dimensions
    uint32_t IN_CH1;
    uint32_t IN_CH2;
    uint32_t OUT_CH; 

    uint32_t target;
    double *ACTIVATIONS;
    double *INIT_BIAS_GRAD;
    double *INIT_WEIGHT_GRAD;
    // golden model output
    double *OUT;

    precision_t dtype;
    
} sgd_bw_layer;

/**
 * @struct conv_layer_struct
 * @brief This structure contains all parameters necessary for Convolutional
 * layers
 * @var conv_layer_struct::CO
 * Number of output channels
 * @var conv_layer_struct::CI
 * Number of input channels
 * @var conv_layer_struct::IH
 * Height of input feature map
 * @var conv_layer_struct::IW
 * Width of input feature map
 * @var conv_layer_struct::OH
 * Height of output feature map
 * @var conv_layer_struct::OW
 * Width of output feature map
 * @var conv_layer_struct::FH
 * Height of filter
 * @var conv_layer_struct::FW
 * Width of filter
 * @var conv_layer_struct::pad
 * Padding on all sides
 * @var conv_layer_struct::ifmap
 * Pointer to input feature map
 * @var conv_layer_struct::weights
 * Pointer to weights
 * @var conv_layer_struct::ofmap
 * Pointer to output feature map
 * @var conv_layer_struct::TILE_CI
 * Tiling factor of input channel
 * @var conv_layer_struct::cluster2cluster
 * Flag for enabling cluster 2 cluster communication
 * @var conv_layer_struct::im2col
 * Flag for enabling im2col + GEMM
 * @var conv_layer_struct::gamma
 * Pointer to gamma for BatchNorm
 * @var conv_layer_struct::beta
 * Pointer to beta for BatchNorm
 * @var gemm_layer_struct::dtype
 * Precision of Convolution layer
 */
typedef struct conv_layer_struct {
    // CONV2D
    uint32_t CO;
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    uint32_t OH;
    uint32_t OW;
    uint32_t FH;
    uint32_t FW;
    uint32_t pad;

    double *ifmap;
    double *weights;
    double *ofmap;

    uint32_t TILE_CI;
    uint32_t cluster2cluster;
    uint32_t im2col;

    // BATCHNORM
    double *gamma;
    double *beta;

    precision_t dtype;
} conv_layer;

typedef struct cnn_layer_struct {
    // CNN
    uint32_t CO;
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    uint32_t OH;
    uint32_t OW;
    uint32_t FH;
    uint32_t FW;
    uint32_t pad;

    double *ifmap;
    double *weights;
    double *biases;
    double *ofmap;

    uint32_t TILE_CI;
    uint32_t cluster2cluster;
    uint32_t im2col;

    // BATCHNORM
    double *gamma;
    double *beta;

    precision_t dtype;
} cnn_layer;
