# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Any change in ROM instances path should be updated in following two files
# 1. hw/system/occamy/fpga/vivado_ips/occamy_rom_placement.xdc and
# 2. hw/top_earlgrey/util/vivado_hook_opt_design_post.tcl

send_msg "Designcheck 2-1" INFO "Checking if ROM memory is mapped to BRAM memory."

if {[catch [get_cells -hierarchical -filter { NAME =~  "*rdata_o_reg" && PRIMITIVE_TYPE =~ BLOCKRAM.*.*}]]} {
  send_msg "Designcheck 2-2" INFO "BRAM implementation found for ROM memory."
} else {
  send_msg "Designcheck 2-3" ERROR "BRAM implementation not found for ROM memory."
}
