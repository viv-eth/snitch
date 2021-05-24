# FPGA Splice flow

This is a FPGA utility script which embedds the generated rom elf file
into FPGA bitstream.  Script assumes there is pre-generated FPGA bit
file in the build directory. The boot rom mem file is auto generated.

## How to run the script

Utility script to load MEM contents into BRAM FPGA bitfile.

### Usage:

```console
$ <path_to_occamy_fpga_root>
$ make splice-bootrom
```

Updated output bitfile located: at the same place as raw vivado bitfile

This directory contains following files:

* `splice_rom_vcu128.sh` - master script
* `bram_load.mmi` - format which vivado tool understands on which FPGA
  BRAM locations the SW contents should go
* `addr4x.py` - utility script used underneath to do address calculation
  to map with FPGA BRAM architecture
