// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "nnlinear_backend_fp16s.h"

#include "network.h"
#include "nnlinear_fp16s.h"
#include "printf.h"
#include "snrt.h"
#include "utils.h"

// define which parts of the network to run
#define RUN_FEEDFORWARD 1
#define RUN_GRADIENT_UPDATE 1
#define RUN_TRAINING_STEP 1 
#define GET_ACCURACY 1
#define GET_LOSS 1
#define RUN_RTL 0
#define NUM_EPOCHS 0
#define BATCH_SIZE 256
#define DATASET_SIZE 256//60000
// #define STEPS(batches) (int)(batches)

void nnlinear_backend_fp16s(const network_fp16_t *n) {
    
    uint32_t cluster_num = snrt_cluster_num(); // Total number of clusters
    uint32_t cluster_core_num = snrt_cluster_core_num(); // Total cores per cluster
    uint32_t cluster_id = snrt_cluster_idx(); // Cluster ID
    uint32_t compute_num = snrt_cluster_compute_core_num(); // Number of compute cores per cluster 
    uint32_t global_compute_num = snrt_global_core_num(); // Total cores incl. DM core per cluster 
    uint32_t compute_id = snrt_cluster_compute_core_idx(); // Core ID of each compute core
    uint32_t dm_id = snrt_cluster_dm_core_idx(); // DM core ID of each cluster
    uint32_t global_compute_id = snrt_global_core_idx(); // Core ID of each core on all clusters

    // if (compute_id == 0) {
    //     printf("======================== System Info ========================\n");
    //     printf("Total number of clusters: %d\n", cluster_num);
    //     printf("Total cores per cluster: %d\n", cluster_core_num);
    //     printf("Number of compute cores per cluster: %d\n", compute_num);
    //     printf("Total cores incl. DM core per cluster: %d\n", global_compute_num);
    //     printf("=============================================================\n");
    // }

    snrt_cluster_hw_barrier();

    uint32_t weights_size = NUM_CLASSES * IN_CH * n->dtype;
    uint32_t biases_size = NUM_CLASSES * n->dtype;
    uint32_t activations_size = NUM_CLASSES * n->dtype;
    uint32_t image_size = IN_CH * n->dtype;
    uint32_t loss_size = n->dtype;
    uint32_t labels_size = sizeof(uint32_t);

    // cluster 0 variabels:
    __fp16 *weights;
    __fp16 *weight_grads;
    __fp16 *biases;
    __fp16 *bias_grads;
    __fp16 *images;
    __fp16 *activations;
    __fp16 *loss;
    uint32_t *targets; 

    void *tcdm_ptr = (__fp16 *)snrt_cluster_memory().start;

    // cluster 0 memory map
    weights = tcdm_ptr;
    tcdm_ptr += weights_size;
    weight_grads = tcdm_ptr;
    tcdm_ptr += weights_size;
    biases = tcdm_ptr;
    tcdm_ptr += biases_size;
    activations = tcdm_ptr;
    tcdm_ptr += activations_size;
    bias_grads = tcdm_ptr;
    tcdm_ptr += biases_size;
    images = tcdm_ptr;
    tcdm_ptr += image_size;
    loss = tcdm_ptr;
    tcdm_ptr += loss_size;
    targets = tcdm_ptr;
    tcdm_ptr += labels_size;

    // DRAM pointers to images and targets
    uint32_t *images_dram = (void *)0x80040000;
    uint32_t *targets_dram = (void *)0x80108000;

    if (snrt_is_dm_core()) {
        snrt_dma_txid_t txid_B = snrt_dma_start_1d(biases, 
                                                    n->b, 
                                                    biases_size);

        // for(uint32_t i = 0; i < NUM_CLASSES; i++){
        //     printf("b[%d] = %f\n", i, biases[i]);
        // }
        snrt_dma_txid_t txid_W = snrt_dma_start_2d(weights,
                                                    n->W,
                                                    IN_CH * n->dtype,
                                                    IN_CH * n->dtype,
                                                    IN_CH * n->dtype,
                                                    NUM_CLASSES);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    uint32_t number_of_images = 256;
    int correct = 0;
    int predict = 0;
    int epoch_count = 0;
    float epoch_loss, epoch_acc = 0;
    float mean_epoch_loss, mean_epoch_acc = 0;
    float batch_acc = 0;
    float batch_loss = 0;
    loss[0] = 0.0f;

    int batches = 0;//DATASET_SIZE / BATCH_SIZE;

    if (compute_id == 0) {
        uint16_t act_hex = 0x000094a2;
        uint16_t img_hex = 0x00002905;
        uint16_t weight_hex = 0x0000275f;
        printf("act hex %u\n", act_hex);
        printf("img hex %u\n", img_hex);
        printf("weight hex %u\n", weight_hex);
        /* WARN: Pointer operations below fail in the RTL 
         if printf is used in the same function.
        */
        __fp16 act_trigger = *((__fp16 volatile *)&act_hex);
        __fp16 img_trigger = *((__fp16 volatile *)&img_hex);
        __fp16 weight_trigger = *((__fp16 volatile *)&weight_hex);
        // __fp16 act_trigger = *((__fp16 *)&act_hex);
        // __fp16 img_trigger = *((__fp16 *)&img_hex);
        // __fp16 weight_trigger = *((__fp16 *)&weight_hex);
        __fp16 volatile mac_trigger;
        float test = 0;
        printf("act trigger %f\n", act_trigger);
        printf("img trigger %f\n", img_trigger);
        printf("weight trigger %f\n", weight_trigger);
        mac_trigger = act_trigger + img_trigger * weight_trigger;
        printf("MAC trigger %f\n", mac_trigger);

        int res0 = 0;
        uint32_t i_a = 0xFFFF4248;   // 3.14
        uint32_t i_an = 0xFFFFC248;  // -3.14
        uint32_t i_b = 0xFFFF3E79;   // 1.618
        uint32_t i_bn = 0xFFFFBE79;  // -1.618
        
        asm volatile(
            "fmv.s.x ft4, %0\n"
            "fmv.s.x ft5, %1\n"
            "fmv.s.x ft6, %2\n"
            // "fmv.s.x ft7, %3\n"
            : "+r"(weight_hex), "+r"(img_hex), "+r"(act_hex)); //, "+r"(i_bn)
        
        asm volatile(
            "fmadd.h ft7, ft4, ft5, ft6\n"
            // "feq.h %0, ft1, ft7\n"
            "fcvt.s.h   ft8, ft7\n"
            : "+f"(test));
    }

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++){
        uint32_t act_hex = 0x000094a2; // __fp16 act_trigger = -0.001131058000000;
        uint32_t img_hex = 0x00002905; // __fp16 img_trigger = 0.039215088000000;
        uint32_t weight_hex = 0x0000275f;//__fp16 weight_trigger = 0.028793335000000;
        __fp16 act_trigger = *((__fp16 volatile *)&act_hex);
        // printf("act_trigger: %f\n", act_trigger);
        // float act_trigger_float = *((__fp16 volatile *)&act_hex);
        // printf("act trigger float: %f\n", act_trigger_float);
        __fp16 img_trigger = *((__fp16 volatile *)&img_hex);
        // float img_trigger_float = *((__fp16 volatile *)&img_hex);
        __fp16 weight_trigger = *((__fp16 volatile *)&weight_hex);
        // float weight_trigger_float = *((__fp16 volatile *)&weight_hex);
        act_trigger += img_trigger * weight_trigger;
        if(snrt_is_compute_core()) {
            printf("MAC trigger: %f\n", act_trigger);
        }
        // asm volatile(
        //     "fmadd.s    %[act_trigger], %[img_trigger], %[weight_trigger], %[act_trigger]\n"
        //     : [act_trigger] "+&f" (act_trigger_float)
        //     : [img_trigger] "f" (img_trigger_float), [weight_trigger] "f" (weight_trigger_float)
        //     :
        // );
        // printf("MAC trigger: %.10f\n", act_trigger_float);

        if(snrt_is_compute_core()) {
            printf("======================== EPOCH [%d/%d] start. ========================\n", (epoch + 1), NUM_EPOCHS);
        }
        for(int batch = 0; batch < batches; batch++){
            batch_loss = 0;
            batch_acc = 0;
            correct = 0;
            if(snrt_is_compute_core()) {
                printf("======================== BATCH [%d/%d] ========================\n", (batch + 1) % batches, batches);
                /* Zero out the gradients 
                * TODO: make this more efficient!
                */
                for (int i = 0; i < NUM_CLASSES; i++) {
                    bias_grads[i] = 0;
                    for (int j = 0; j < IN_CH; j++) {
                        weight_grads[i * IN_CH + j] = 0;
                    }

                }

                printf("INFO: Gradients have been zeroed out.\n");

                snrt_cluster_hw_barrier();

            } else if(!snrt_is_compute_core()) {
                snrt_cluster_hw_barrier();
            }
            for(uint32_t image = 0; image < BATCH_SIZE; image++){
                uint32_t volatile curr_img = image * IN_CH + batch * BATCH_SIZE * IN_CH;
                // printf("======================== Image %d ========================\n", curr_img / 784);
                uint32_t volatile curr_target = image + batch * BATCH_SIZE;
                if (snrt_is_dm_core()) {
                        // float img_checksum = 0;
                        snrt_dma_start_tracking();
                        snrt_dma_txid_t txid_img = 
                                snrt_dma_start_1d(images,                                   // destination
                                                &images_dram[curr_img],                     // source
                                                n->dtype * IN_CH);                          // size
                        snrt_dma_wait_all();
                        snrt_dma_txid_t txid_target = 
                                snrt_dma_start_1d(targets,                                   // destination
                                                &targets_dram[curr_target],                 // source
                                                sizeof(uint32_t));                                  // size
                        
                        // printf("======================== Image %u ========================\n", image);
                        // for(int i = 0; i < IN_CH; i++){
                        //     // printf("image[%u][%u] = %f\n", image, i, images[i]);
                        //     img_checksum += images[i];
                        // }
                        // printf("Image checksum[%d] = %f\n", image, img_checksum);
                        // printf("=============================================================\n");
                        // printf("Image: %u, Target: %u\n", image, targets[0]);
                        snrt_dma_wait_all();
                        // if (curr_img / 784 == 261) {
                        //     for(int i = 0; i < IN_CH; i++){
                        //             printf("image[%u][%u] = %f\n", image, i, images[i]);
                        //         }
                        // }
                }

                snrt_cluster_hw_barrier();

                if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {

                    // if (image == 65) {
                    //     printf(" ============= BEFORE GRADIENT UPDATE =============");
                    //     printf("targets[%d] = %d\n", image, targets[image]);
                    //     printf("loss[%d] = %f\n", image, *loss);
                    //     for(int i = 0; i < NUM_CLASSES; i++){
                    //         printf("biases[%d] = %f\n", i, biases[i]);
                    //         printf("activations[%d] = %f\n", i, activations[i]);
                    //         printf("bias grads[%d] = %f\n", i, bias_grads[i]);
                    //         for (int j = 0; j < IN_CH; j++) {
                    //             if(i == 0) {
                    //                 printf("images[%d] = %f\n", j, images[j]);
                    //             }
                    //             if (weight_grads[i * IN_CH + j] > 10 || weight_grads[i * IN_CH + j] < -10 || weights[i * IN_CH + j] > 10 || weights[i * IN_CH + j] < -10) {
                    //                 printf("weight grads[%d][%d] = %f\n", i, j, weight_grads[i * IN_CH + j]);
                    //                 printf("weights[%d][%d] = %f\n", i, j, weights[i * IN_CH + j]);
                    //             }
                    //         }
                    //     }

                    //     printf("====================== END OF BEFORE GRADIENT UPDATE ======================\n");
                    // }
                    
                    GradientUpdate_fp16s(images, activations, biases, weights, weight_grads, bias_grads, targets[0], loss); 
                    // if (image == 65) {
                    //     printf(" ============= AFTER GRADIENT UPDATE =============");
                    //     printf("targets[%d] = %d\n", image, targets[image]);
                    //     printf("loss[%d] = %f\n", image, *loss);
                    //     for(int i = 0; i < NUM_CLASSES; i++){
                    //         printf("biases[%d] = %f\n", i, biases[i]);
                    //         printf("activations[%d] = %f\n", i, activations[i]);
                    //         printf("bias grads[%d] = %f\n", i, bias_grads[i]);
                    //         for (int j = 0; j < IN_CH; j++) {
                    //             if(i == 0) {
                    //                 printf("images[%d] = %f\n", j, images[j]);
                    //             }
                    //             if(weight_grads[i * IN_CH + j] > 10 || weight_grads[i * IN_CH + j] < -10 || weights[i * IN_CH + j] > 10 || weights[i * IN_CH + j] < -10) {
                    //                 printf("weight grads[%d][%d] = %f\n", i, j, weight_grads[i * IN_CH + j]);
                    //                 printf("weights[%d][%d] = %f\n", i, j, weights[i * IN_CH + j]);
                    //             }
                    //         }
                    //     }
                    //     printf("====================== END OF AFTER GRADIENT UPDATE ======================\n");
                    // }
                    snrt_cluster_hw_barrier();
                    batch_loss += *loss;
                    /* Accuracy Calculation */
                    __fp16 max_activation = activations[0];
                    predict = 0;
                    for (int i = 0; i < NUM_CLASSES; i++) {
                        if(max_activation < activations[i]) {
                            max_activation = activations[i];
                            predict = i;
                        }
                    }

                    if(predict == targets[0]) {
                        correct++;
                    }
                    snrt_cluster_hw_barrier();

                    // printf("pred = %d, target = %d\n", predict, targets[0]);


                } else if (!snrt_is_compute_core()) {
                    snrt_cluster_hw_barrier();
                    snrt_cluster_hw_barrier();
                    snrt_cluster_hw_barrier();
                    snrt_cluster_hw_barrier();
                }
            }

            snrt_cluster_hw_barrier();

            // After one epoch we update the weights
            if (snrt_is_compute_core() && snrt_cluster_compute_core_idx() < compute_num) {
                
                batch_acc = (float)correct / (float)BATCH_SIZE;
                epoch_acc += batch_acc;
                epoch_loss += batch_loss / BATCH_SIZE;
                printf("A total of [%d/%d] images were predicted correctly in batch %d\n", correct, BATCH_SIZE, batch + 1);
                printf("batch acc = %.6f\n", batch_acc * 100);
                printf("batch loss = %.6f\n", batch_loss / BATCH_SIZE);


                TrainingStep_fp16s(biases, weights, weight_grads, bias_grads, n->learning_rate);
                
                if(batch%(batches - 1)==0 && batch!=0) {

                    epoch_count++;
                    mean_epoch_loss = epoch_loss/batches;
                    mean_epoch_acc = epoch_acc/batches;
                    printf("===========================  EPOCH %u done. ===========================\n", epoch_count);
                    printf("===========================  Epoch  Acc %.3f  ===========================\n", mean_epoch_acc * 100);
                    printf("===========================  Epoch  Loss %.3f  ===========================\n", mean_epoch_loss);
                    epoch_loss = 0;
                    epoch_acc = 0;

                }


            } else if (!snrt_is_compute_core()) {

                snrt_cluster_hw_barrier();

            }

            snrt_cluster_hw_barrier();


        }
    }
    snrt_global_barrier();

}