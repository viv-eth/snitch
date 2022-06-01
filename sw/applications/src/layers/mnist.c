// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "mnist.h"

#include "network.h"
#include "mnist_network.h"
#include "printf.h"
#include "snrt.h"
#include "mnist_data.h"
#include "utils.h"


// Padding in between matrices for preventing
// banking conflicts in the beginning
#define MAT_PADDING 8

#define MAT_ROW_PADDING 4

// const char * train_images_file = "MNIST/train-images-idx3-ubyte";
// const char * train_labels_file = "MNIST/train-labels-idx1-ubyte";

void mnist(const network_t *n){

    uint32_t cluster_num = snrt_cluster_num(); // returns 2
    uint32_t cluster_core_num = snrt_cluster_core_num(); // returns 9
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_num = snrt_cluster_compute_core_num(); // Number of compute cores per cluster --> returns 8
    uint32_t global_compute_num = snrt_global_core_num(); // Total cores incl. DM core per cluster --> returns 18
    uint32_t compute_id = snrt_cluster_compute_core_idx();
    uint32_t global_compute_id = snrt_global_core_idx(); // Core ID of each core on all clusters

    //printf("Cluster Num: %u\n", cluster_num);

    uint32_t IN_CH = n->IN_CH1*n->IN_CH2; // number of total input channels for a flattened image
    // determine a load stride for loading chunks of data from DRAM to local cluster memory
    // --> should be a multiple of IN_CH
    uint32_t dram_stride = n->IN_CH1;
    // this defines at which DRAM stride of the image we are
    // range is 0*28 ... 27*28
    uint32_t dram_stride_offset = 0; 

    // size of the weight matrix, same for weight gradients
    // on cluster 0 we store the weights, on cluster 1 the
    // weight gradients at this location
    // uint32_t weight_mat_size = (n->OUT_CH * IN_CH + MAT_PADDING) * n->dtype;
    uint32_t weight_mat_size = (n->OUT_CH * IN_CH + MAT_PADDING) * n->dtype;
    // size of the bias matrix, same for bias gradients
    // on cluster 0 we store the biases, on cluster 1 the
    // bias gradients at this location
    uint32_t bias_mat_size = n->OUT_CH * n->dtype;
    // size of activations (same as biases), which we need to synchronize the clusters
    uint32_t act_mat_size = bias_mat_size;
    // size of a single MNIST image (28x28 = 784 pixels)
    uint32_t image_size = IN_CH * n->dtype;
    // size for storing the maximum on each core of a cluster (only used on cluster 0)
    uint32_t max_size = n->dtype;//compute_num * n->dtype;
    // size of the target for image classification (0...9)
    uint32_t target_size = sizeof(uint32_t);
    // result of the cross entropy loss calculation
    uint32_t loss_size = n->dtype;
    // synchronization flags for the compute cores on among clusters
    uint32_t core_sync_flag_size = compute_num*sizeof(uint32_t);
    // synchronization flags for the compute cores on each cluster
    uint32_t cluster_sync_flag_size = cluster_num*sizeof(uint32_t);
    // sync flag value of the last compute core in the forward pass
    uint32_t last_core_ff_size = sizeof(uint32_t);
    // sync flag which is asserted when the final core starts its final computation
    uint32_t is_last_core_size = sizeof(uint32_t);

    // FP64 cluster memory setup
    // @brief Cluster Memory Structure for each cluster to ensure
    // we can access the data of both by using the constant
    // cluster base offset
    void *ptr = (double *)snrt_cluster_memory().start;
    void *ptr_start = ptr;
    uint32_t *cluster_sync = ptr;
    ptr += cluster_sync_flag_size;
    uint32_t *core_sync = ptr; // zero initialized
    ptr += core_sync_flag_size;
    uint32_t *last_core_ff = ptr;
    ptr += last_core_ff_size;
    uint32_t is_last_core = ptr;
    ptr += is_last_core_size;
    double *max= ptr; // zero initialized
    ptr += max_size;
    uint32_t *target = ptr;
    ptr += target_size;
    double *loss = ptr; // zero initialized
    ptr += loss_size;
    double *image = ptr;
    ptr += image_size;
    double *biases = ptr; // bias GRADIENTS zero initialized
    ptr += bias_mat_size;
    double *activations = ptr;
    ptr += act_mat_size;
    // INFO: setting weights as last element so 
    // when we iterate over data we can zero out
    // excessive rows
    double *weights = ptr; // weight GRADIENTS zero initialized
    ptr += weight_mat_size;
    void *ptr_end = (double *)snrt_cluster_memory().end;
    if(compute_id == 0){   
        printf("Start address of cluster %u memory: 0x%p\n", cluster_id, ptr_start);
        printf("End address of cluster %u memory: 0x%p\n", cluster_id, ptr_end);
        printf("Available memory on cluster %u: %u KB\n", cluster_id, (ptr_end - ptr_start) / 1000);
        printf("Total cluster memory occupation on cluster %u: %u KB\n", cluster_id, (ptr - ptr_start) / 1000);
    }

    //printf("I am alive and my memory range is: [0x%p, 0x%p]\n", ptr_start, ptr_end);

//     // For every new epoch we load the (updated) weights
//     // and biases into the cluster 0 memory
//     // TODO: add the epochs
//     if (snrt_is_dm_core() && cluster_id == 0) {

//                 // load initial biases from Golden Model into Cluster 0 memory
//                 snrt_dma_txid_t txid_B = 
//                     snrt_dma_start_1d(biases,                 // destination
//                                     n->b,                     // source
//                                     n->dtype * n->OUT_CH);    // size

//                 // load image data into Cluster 0 memory
//                 snrt_dma_txid_t txid_W =
//                     snrt_dma_start_2d(weights,                // destination
//                                     n->W,                     // source
//                                     n->dtype * IN_CH ,        // size
//                                     n->dtype * IN_CH ,        // destination stride
//                                     n->dtype * IN_CH ,        // source stride
//                                     n->OUT_CH);               // repetitions
                
//                 // wait until each DMA transfer done
//                 snrt_dma_wait_all();
//                 //printf("Data transfer done.\n");

//     }

//     // Global memory access
//     uint32_t *global_mem = (void *)snrt_global_memory().start;
    
//     // DRAM dataset memory start address
//     uint32_t *dataset_dram = (void *)0x8004000;
//     // get the size of one data slice (label + image)
//     uint32_t data_slice = 1 + IN_CH;
//     uint32_t number_of_images = 2;
//     uint32_t image_count = 0;

//     //printf("First Value entry of DRAM: %u\n", dataset_dram[0]);

//     // create buffers for the DRAM data that we
//     // want to load into the cluster local memory
//     double image_buffer_double[IN_CH];
//     double image_buffer_strided_double[IN_CH / dram_stride];
//     uint32_t label_buffer;

//     // write the values from DRAM into the label and image buffer
//     // TODO: we will have to increase the dataset_dram pointer by
//     // the datawidth (785- 1 for label and 784 for image data)
//     // for each iteration of the training 
//     //if(!(cluster_id || compute_id)){printf("Value of label = %u\n", label_buffer);}
//     for(; image_count < number_of_images; image_count++){
//         // if we are at the first image, we load the entire image into the cluster memory
//         if(image_count == 0){
//             label_buffer = dataset_dram[0]; // this has to be only written once for every image
//             // first we fill the write buffer
//             // pixel count starts at 1, since label is at 0
//             for(uint32_t pixel = 0; pixel < IN_CH ; pixel++){
//                 image_buffer_double[pixel] = (double) dataset_dram[pixel + 1] / 255;
//             }
//             // start DMA transfer for cluster 0
//             if (snrt_is_dm_core() && cluster_id == 0) {

//                 // load image data into Cluster 0 memory
//                 snrt_dma_txid_t txid_IMG =
//                     snrt_dma_start_2d(image,                  // destination
//                                     image_buffer_double,      // source 
//                                     n->dtype * IN_CH,         // size   
//                                     n->dtype * IN_CH,         // destination stride
//                                     n->dtype * IN_CH,         // source stride
//                                     1);                       // repetitions
                
//                 // wait until each DMA transfer done
//                 snrt_dma_wait_all();
//                 //printf("Data transfer done.\n");

//             }
//             // after reading in the image  we increment the
//             // image count 
//             image_count += 1;


//         } else { 
//             // we load the correct label 
//             // We update the write buffer with the strided version
//             label_buffer = dataset_dram[image_count*data_slice + 1];
//             for(uint32_t pixel = 0; pixel < dram_stride; pixel++){
//                 image_buffer_strided_double[pixel] = (double) dataset_dram[image_count*data_slice + pixel + 1] / 255;
//             }
            
//             // afterward we check the sync flag of the last compute core working on the data
//             // we always update the image data with a chunk of new data corresponding to a full row
//             // we only start updating when our computation is at the last core
//             if(cluster_id == 0 && is_last_core){
//                 // we only update after 28 pixels, i.e. one row
//                 // we don't update if we are in the first 28 pixels
//                 if(last_core_ff[0] % 28 == 0 && last_core_ff[0]){
//                     if(snrt_is_dm_core()){
//                         // load image chunk into the right position of the image
//                         // TODO: determine image offset where to store the strided data
//                         snrt_dma_txid_t txid_IMG =
//                             snrt_dma_start_2d(image,                  // destination
//                                             image_buffer_strided_double,      // source 
//                                             n->dtype * IN_CH,         // size   
//                                             n->dtype * IN_CH,         // destination stride
//                                             n->dtype * IN_CH,         // source stride
//                                             1);                       // repetitions
                        
//                         // wait until each DMA transfer done
//                         snrt_dma_wait_all();
//                     }
//                 }
//             }
//         }

//     }
    


//     // cluster offset in an Occamy quadrant
//     uint32_t cluster_offset = 0x00040000;

//     snrt_global_barrier(); 

//     // start DMA transfer for cluster 0
//     if (snrt_is_dm_core() && cluster_id == 0) {

//         // load initial biases from Golden Model into Cluster 0 memory
//         snrt_dma_txid_t txid_B = 
//             snrt_dma_start_1d(biases,                   // destination
//                               n->b,                     // source
//                               n->dtype * n->OUT_CH);    // size

//         // load image data into Cluster 0 memory
//         snrt_dma_txid_t txid_IMG =
//             snrt_dma_start_2d(image,                    // destination                
//                               //n->images,                // source
//                               image_buffer_double,      // source INFO: DRAM chunk
//                               n->dtype * IN_CH / dram_stride,         // size   
//                               n->dtype * IN_CH / dram_stride,         // destination stride
//                               n->dtype * IN_CH / dram_stride,         // source stride
//                               1);                       // repetitions

//         // load image data into Cluster 0 memory
//         snrt_dma_txid_t txid_W =
//             snrt_dma_start_2d(weights,                  // destination
//                               n->W,                     // source
//                               n->dtype * IN_CH / dram_stride,         // size
//                               n->dtype * IN_CH / dram_stride,         // destination stride
//                               n->dtype * IN_CH / dram_stride,         // source stride
//                               n->OUT_CH);               // repetitions
        
//         // wait until each DMA transfer done
//         snrt_dma_wait_all();
//         //printf("Data transfer done.\n");

//     } else if(snrt_is_dm_core() && cluster_id == 1){ // DMA transfer for Cluster 1

//         snrt_dma_txid_t txid_target = 
//             snrt_dma_start_1d(target,                   // destination
//                               label_buffer,               // source
//                               sizeof(uint32_t));        // size

//         // wait until each DMA transfer done
//         snrt_dma_wait_all();
//     }


//     // Synchronize cores in a cluster with a hardware barrier
//     snrt_cluster_hw_barrier();

//     // INFO: global barrier test
//     // cluster_global_barrier(cluster_num*cluster_core_num);

//     // if(!(cluster_id)){
//     //     loss[0] = 42;
//     //     printf("Cluster %u Loss = %f\n", cluster_id, loss[0]);
//     // }

//     // cluster_global_barrier(cluster_num*cluster_core_num);

//     // if(cluster_id){
//     //     double *loss_ptr = ((uint32_t)loss) - cluster_offset;
//     //     printf("Cluster %u Loss stolen from Cluster 0 Memory = %f\n", cluster_id, *loss_ptr);
//     //     printf("Cluster %u Loss in Cluster 1 = %f\n", cluster_id, loss[0]);
//     // }


// //     if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {
// //         const uint32_t setup_SSR = 1;
// //         // determine the row offset at which current compute cluster is
// //         volatile uint32_t W_offset =
// //             compute_id * IN_CH;
// //         volatile uint32_t b_offset =
// //             compute_id;

// //         // Calculate number of rows for each compute
// //         // core. If multiples of each other we have to 
// //         // forcefully set it to 1
// //         volatile uint32_t div = n->OUT_CH % compute_num;
// //         if(div == 0){
// //             div = 1;
// //         }


// //         //printf("weights[%u] = %f\n", W_offset, weights[W_offset]);
// //         //printf("biases[%u] = %f\n", b_offset, biases[b_offset]);
// //         //printf("test3: val of target = %u\n", *target);

// //         // determine the row stride of each matrix    
// //         volatile uint32_t ldW = compute_num * IN_CH;
// //         volatile uint32_t ldB = compute_num;
// //         volatile uint32_t ldI = IN_CH;

// //         //printf("ldW: %u, ldB: %u, ldI: %u\n", ldW, ldB, ldI);

// //         if(n->dtype == FP64){
// //             //TODO: handle parallelization properly by wrapping around the cores
// //             //      instead of manually dividing by a convenient number
// //             benchmark_get_cycle();
// //             if(cluster_id==0){
// //                 printf("Feedforward start. \n");
// //                 feedforward_fp64(n->IN_CH1, n->IN_CH2, div, 
// //                                 &weights[W_offset], ldW, &biases[b_offset], ldB,
// //                                 image, ldI, compute_id);
// //                 printf("Feedforward done. \n");
// //                 printf("SoftMax start. \n");
// //                 softmax_activation_fp64(n->IN_CH1, n->IN_CH2, div, 
// //                             &weights[W_offset], ldW, &biases[b_offset], ldB,
// //                             image, ldI, compute_id, compute_num, max);
// //                 printf("SoftMax done. \n");
// //             }
// //             snrt_global_barrier();
// //             //snrt_cluster_hw_barrier();
// //             // only go into BW computation if FW is set to compute_num
// //             // && FW_flag == compute_num
// //             if(cluster_id == 1){
// //                 // FIXME: Cores of Cluster 1 go sometimes into Gradient Update BEFORE Forward Pass is DONE
// //                 printf("Gradient Update start. \n");
// //                 gradient_update_fp64(n->IN_CH1, n->IN_CH2, div, 
// //                                 &weight_grads[W_offset], ldW, 
// //                                 &bias_grads[b_offset], &biases[b_offset], 
// //                                 ldB, image, target, ldI, compute_id, 
// //                                 loss, compute_num);
// //                 printf("Loss: %f\n", loss[0]);
// //                 printf("Gradient Update done. \n");
// //             }
// //             benchmark_get_cycle();
// //         }
// //     } else{
// //         // for ninth core (DMA) core 
// //         // INFO: all cores should have same amount of barriers, HW barrier less computational heavy than others
// //         snrt_cluster_hw_barrier();
// //         snrt_cluster_hw_barrier();
// //         snrt_cluster_hw_barrier();
// //         snrt_cluster_hw_barrier();
// //         //snrt_cluster_hw_barrier();
// //     }
// //     snrt_cluster_hw_barrier();
// //     //snrt_global_barrier();

// //     // TODO: implement proper checking, maybe separate function
// //     // INFO: computed values match the actual weight gradient from the GM

// //     // if(cluster_id == 1 && compute_id == 0){
        
// //     //     printf("Entering check.\n");
        
// //     //     for(uint32_t check = 0; check < n->OUT_CH*IN_CH; check++){
// //     //         printf("GM Weight grad val[%u] = %f\n", check, weight_grad_check[check]);
// //     //         printf("Computed weight grad val[%u] = %f\n", check, weight_grads[check]);
// //     //     }
// //     // }
}