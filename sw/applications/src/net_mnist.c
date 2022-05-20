// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SW testbench for profiling linear kernels in different
// floating point precisions (fp64, fp32, fp16), as well as
// different memory layouts for matrices (transposed/not-transposed)
// Correctness of results are checked automatically

#include "mnist.h"
#include "data_mnist.h"
#include "mnist_data.h"
#include "network.h"
#include "math.h"
#include "perf_cnt.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

// const char * train_images_file = "MNIST/train-images-idx3-ubyte";
// const char * train_labels_file = "MNIST/train-labels-idx1-ubyte";
// const char * test_images_file = "MNIST/t10k-images-idx3-ubyte";
// const char * test_labels_file = "MNIST/t10k-labels-idx1-ubyte";

int main(){
    // get cluster & core ID for debugging purposes
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_id = snrt_cluster_compute_core_idx();
    
    // load inut data from the golden model 
    // give the start address you defined in the SnitchUtilities argument flags
    // uint32_t *dataset_dram = (void *)0x8004000;
    // now we check the output -- > uncomment if you are unsure if your DRAM is loaded correctly
    // if(!(cluster_id || compute_id)){
    //     // we check only the first image
    //     for(uint32_t i = 0; i < 785; i++){
    //         if(i == 0){
    //             printf("Address: %p, Label: %u\n", &test_dram_preload[i], test_dram_preload[i]);
    //         } else{
    //             printf("Address: %p, Pixel Value: %u\n", &test_dram_preload[i], test_dram_preload[i]);
    //         }
    //     }
    // }
    mini_mnist_t.W = (void *)mini_mnist_weights_dram;
    mini_mnist_t.b = (void *)mini_mnist_biases_dram;
    // mini_mnist_t.W_grad = (void *)mini_mnist_weight_grads_dram; --> zero initialized, don't have to be pre-loaded
    // mini_mnist_t.b_grad = (void *)mini_mnist_bias_grads_dram; --> zero initialized, don't have to be pre-loaded

    mini_mnist_t.images = (void *)mini_mnist_images_dram;
    mini_mnist_t.targets = (void *)mini_mnist_labels_dram;
    
    mnist(&mini_mnist_t);

    //printf("after doin network shit\n");

    snrt_global_barrier();
    //cluster_global_barrier(18);
    //printf("success\n");

    return 0;
}