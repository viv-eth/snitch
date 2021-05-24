# Any change in ROM instances path should be updated in following two files
# 1. hw/system/occamy/fpga/vivado_ips/occamy_rom_placement.xdc and
# 2. hw/system/occamy/fpga/vivado_ips/occamy_check_rom_inference.xdc

set_property LOC RAMB36_X10Y51 [get_cells -hierarchical -filter { NAME =~  "*rdata_o_reg" && PRIMITIVE_TYPE =~ BLOCKRAM.*.*}]
