# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>
#
# Add occamy signals to debug here

# Ariane commited PC
set_property MARK_DEBUG true [get_nets {i_occamy/i_occamy_cva6/i_cva6/pc_commit[*]}]

# Ariane fetched instruction
set_property MARK_DEBUG true [get_nets {i_occamy/i_occamy_cva6/i_cva6/i_frontend/icache_data_q[*]}]

# Ariane exception CSRs
set_property MARK_DEBUG true [get_nets {i_occamy/i_occamy_cva6/i_cva6/csr_regfile_i/mcause_q[*]}]
set_property MARK_DEBUG true [get_nets {i_occamy/i_occamy_cva6/i_cva6/csr_regfile_i/mtval_q[*]}]
set_property MARK_DEBUG true [get_nets {i_occamy/i_occamy_cva6/i_cva6/csr_regfile_i/mepc_q[*]}]

# Boot ROM response, data, address
set_property mark_debug true [get_nets [list occamy_vcu128_i/occamy_xilinx_0/inst/i_occamy_bootrom_wrap/bootrom_req_ready_q_reg_0]]
set_property mark_debug true [get_nets [list occamy_vcu128_i/occamy_xilinx_0/inst/i_occamy_bootrom_wrap/DOUTADOUT[*]]]
set_property mark_debug true [get_nets [list occamy_vcu128_i/occamy_xilinx_0/inst/i_occamy_bootrom_wrap/ADDRARDADDR[*]]]
