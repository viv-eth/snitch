# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

INCLUDES = -I./ -I./src
SRCS_C = src/main.c src/uart.c
OBJS_C = $(SRCS_C:.c=.o)

bootrom.coe: bootrom.bin
	bin2coe -i $< -o $@ -w 32

bootrom.elf bootrom.dump bootrom.bin: bootrom.S $(OBJS_C) bootrom.ld
	riscv64-unknown-elf-gcc -mabi=lp64d -march=rv64imafd -static -nostartfiles $(INCLUDES) -Tbootrom.ld $< $(OBJS_C) -o bootrom.elf
	riscv64-unknown-elf-objdump -d bootrom.elf > bootrom.dump
	riscv64-unknown-elf-objcopy -O binary bootrom.elf bootrom.bin

# Generate .vmem bootcode from the binary using srec_cat
bootrom.vmem: bootrom.bin
	python3 gen_rom_vmem.py \
	--basename occamy-bootrom \
	--input bootrom.bin \
	--outdir . \
	--srec_cat /bin/srec_cat

# Generate System Verilog Boot ROM module for Occamy.
bootrom.sv: bootrom.vmem
	python3 gen_rom_sv.py \
	--vmem occamy-bootrom.32.vmem \
	--out occamy_xilinx_bootrom

	mv ./occamy_xilinx_bootrom.sv ../../src/occamy_xilinx_bootrom.sv

# Splice bootcode to existing bitstream without regenerating it from scratch
occamy_splice_bootrom: bootrom.bin
	cd splice_rom_vcu128 && ./splice_rom_vcu128.sh

%.o: %.c
	riscv64-unknown-elf-gcc -Os -ggdb -march=rv64imafd -mabi=lp64d -Wall -mcmodel=medany -mexplicit-relocs -fno-builtin $(INCLUDES) -c $<  -o $@
	@echo "CC    <= $<"

clean:
	rm *.bin *.coe *.dump src/*.o
