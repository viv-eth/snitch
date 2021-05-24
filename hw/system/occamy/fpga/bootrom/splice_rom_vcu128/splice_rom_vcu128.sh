#!/bin/bash
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0


# Utility script to load MEM contents into BRAM FPGA bitstream file

# Replace bitstream with a new one containing a bootrom initialized with novel bootcode
# New bistream always @ hw/system/occamy/fpga/occamy_vcu128.runs/impl_1/occamy_vcu128_wrapper.bit
set -e

TARGET="../bootrom"
FPGA_IMPL_DIR=../../occamy_vcu128/occamy_vcu128.runs/impl_1
FPGA_BIT_NAME=occamy_vcu128_wrapper

srec_cat "${TARGET}.bin" -binary -offset 0x0 --byte_swap 4 --fill 0xff -within "${TARGET}.bin" -binary -range_pad 4 -o "${TARGET}.brammem" -vmem -Output_Block_Size 4;

#srec_cat "${TARGET}.bin" -binary -offset 0x0 -o "${TARGET}.brammem" \
#         -vmem -Output_Block_Size 4;

python3 addr4x.py -i "${TARGET}.brammem" -o "${TARGET}.mem"

vitis-2020.2 updatemem -force --meminfo ./bram_load.mmi \
  --data "${TARGET}.mem" \
  --bit "${FPGA_IMPL_DIR}/${FPGA_BIT_NAME}.bit"  --proc dummy \
  --out "${FPGA_IMPL_DIR}/${FPGA_BIT_NAME}.splice.bit"

#mv ${FPGA_BUILD_DIR}/${FPGA_BIT_NAME}.bit ${FPGA_BUILD_DIR}/${FPGA_BIT_NAME}.bit.orig
#mv ${FPGA_BUILD_DIR}/${FPGA_BIT_NAME}.splice.bit ${FPGA_BUILD_DIR}/${FPGA_BIT_NAME}.bit
