// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef MNIST_FILE_H_
#define MNIST_FILE_H_

#include <stdint.h>
#include <stdbool.h>
#include "layer.h"

#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT
#define MNIST_LABELS 10

/**
 * @struct mnist_label_file_header_t_
 * @brief Struct to retrieve number of labels and magic number from
 * the raw MNIST label data 
 */
typedef struct mnist_label_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_labels;
} __attribute__((packed)) mnist_label_file_header_t;

/**
 * @struct mnist_image_file_header_t_
 * @brief Struct to retrieve number of magic number, images, 
 * number of columns, and number of rows from
 * the raw MNIST image data 
 */
typedef struct mnist_image_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t number_of_rows;
    uint32_t number_of_columns;
} __attribute__((packed)) mnist_image_file_header_t;

/**
 * @struct mnist_image_t_
 * @brief Struct to represent a single MNIST image
 * of size MNIST_IMAGE_SIZE
 */
typedef struct mnist_image_t_ {
    uint8_t pixels[MNIST_IMAGE_SIZE];
} __attribute__((packed)) mnist_image_t;

/**
 * @struct mnist_dataset_t_
 * @brief Dataset with images and their respective labels
 * @var images The 28x28 MNIST images
 * @var labels The respective labels, i.e. the number represented by images
 * @var size Size of the dataset
 */
typedef struct mnist_dataset_t_ {
    mnist_image_t * images;
    uint8_t * labels;
    uint32_t size;
} mnist_dataset_t;

mnist_dataset_t * mnist_get_dataset(const char * image_path, const char * label_path);
mnist_dataset_t * mnist_get_chunks(mnist_dataset_t * dataset, int size, int offset);
void mnist_free_dataset(mnist_dataset_t * dataset);
int mnist_batch(mnist_dataset_t * dataset, mnist_dataset_t * batch, int batch_size, int batch_number);

#endif
