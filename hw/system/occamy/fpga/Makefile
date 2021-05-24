# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := ${MKFILE_DIR}../../../..
DEBUG       ?= 0

VIVADO := vivado

include $(ROOT)/util/Makefrag

vivado_ips/occamy_xilinx:
	${MAKE} -C vivado_ips occamy_xilinx

bootrom/bootrom.sv:
	${MAKE} -C bootrom bootrom.sv

occamy_vcu128: bootrom/bootrom.sv vivado_ips/occamy_xilinx define_defines_includes_no_simset.tcl
	${VIVADO} -source occamy_vcu128.tcl -tclargs $(DEBUG)

# Generate a new bistream with updated bootcode. The old bistream is replaced (no backup) by the new one @ same location
occamy_update_bistream:
	make -C bootrom occamy_splice_bootrom

define_defines_includes_no_simset.tcl: $(BENDER_FILES)
	${BENDER} script vivado --only-defines --only-includes --no-simset > $@

program:
	vivado -mode batch -source occamy_vcu128_program.tcl

clean:
	${MAKE} -C vivado_ips clean
	rm -rf .Xil occamy_vcu128 define_defines_includes_no_simset.tcl *.jou *.log *.str

.PHONY: program clean
