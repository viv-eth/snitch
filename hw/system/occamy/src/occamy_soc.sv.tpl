// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>
// Author: Fabian Schuiki <fschuiki@iis.ee.ethz.ch>
//
// AUTOMATICALLY GENERATED by genoccamy.py; edit the script instead.

`include "common_cells/registers.svh"

module occamy_soc
  import occamy_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        test_mode_i,

  /// HBM2e Ports
% for i in range(8):
  output  ${soc_wide_xbar.__dict__["out_hbm_{}".format(i)].req_type()} hbm_${i}_req_o,
  input   ${soc_wide_xbar.__dict__["out_hbm_{}".format(i)].rsp_type()} hbm_${i}_rsp_i,
% endfor

  /// HBI Ports
% for i in range(nr_s1_quadrants+1):
  input   ${soc_wide_xbar.__dict__["in_hbi_{}".format(i)].req_type()} hbi_${i}_req_i,
  output  ${soc_wide_xbar.__dict__["in_hbi_{}".format(i)].rsp_type()} hbi_${i}_rsp_o,
  output  ${wide_xbar_quadrant_s1.out_hbi.req_type()} hbi_${i}_req_o,
  input   ${wide_xbar_quadrant_s1.out_hbi.rsp_type()} hbi_${i}_rsp_i,
% endfor

  /// PCIe Ports
  output  ${soc_wide_xbar.out_pcie.req_type()} pcie_axi_req_o,
  input   ${soc_wide_xbar.out_pcie.rsp_type()} pcie_axi_rsp_i,

  input  ${soc_wide_xbar.in_pcie.req_type()} pcie_axi_req_i,
  output ${soc_wide_xbar.in_pcie.rsp_type()} pcie_axi_rsp_o,

  // Peripheral Ports (to AXI-lite Xbar)
  output  ${soc_narrow_xbar.out_periph.req_type()} periph_axi_lite_req_o,
  input   ${soc_narrow_xbar.out_periph.rsp_type()} periph_axi_lite_rsp_i,

  input   ${soc_narrow_xbar.in_periph.req_type()} periph_axi_lite_req_i,
  output  ${soc_narrow_xbar.in_periph.rsp_type()} periph_axi_lite_rsp_o,

  // Peripheral Ports (to Regbus Xbar)
  output  ${soc_narrow_xbar.out_regbus_periph.req_type()} periph_regbus_req_o,
  input   ${soc_narrow_xbar.out_regbus_periph.rsp_type()} periph_regbus_rsp_i,

  // SoC control register IO
  // FIXME: reg2hw and hw2reg connections are currently unused; this may change, however. 
  input  occamy_soc_reg_pkg::occamy_soc_reg2hw_t soc_ctrl_out_i,
  output occamy_soc_reg_pkg::occamy_soc_hw2reg_t soc_ctrl_in_o,
  output logic [1:0] spm_rerror_o,

  // Interrupts and debug requests
  input  logic [${cores-1}:0] mtip_i,
  input  logic [${cores-1}:0] msip_i,
  input  logic [1:0] eip_i,
  input  logic [0:0] debug_req_i,

  /// SRAM configuration
  input sram_cfgs_t sram_cfgs_i
);

  <% spm_words = cfg["spm"]["length"]//(soc_narrow_xbar.out_spm.dw//8) %>

  typedef logic [${util.clog2(spm_words) + util.clog2(soc_narrow_xbar.out_spm.dw//8)-1}:0] mem_addr_t;
  typedef logic [${soc_narrow_xbar.out_spm.dw-1}:0] mem_data_t;
  typedef logic [${soc_narrow_xbar.out_spm.dw//8-1}:0] mem_strb_t;

  logic spm_req, spm_gnt, spm_we, spm_rvalid;
  mem_addr_t spm_addr;
  mem_data_t spm_wdata, spm_rdata;
  mem_strb_t spm_strb;

  addr_t [${nr_s1_quadrants-1}:0] s1_quadrant_base_addr, s1_quadrant_cfg_base_addr;
  % for i in range(nr_s1_quadrants):
  assign s1_quadrant_base_addr[${i}] = ClusterBaseOffset + ${i} * S1QuadrantAddressSpace;
  assign s1_quadrant_cfg_base_addr[${i}] = S1QuadrantCfgBaseOffset + ${i} * S1QuadrantCfgAddressSpace;
  % endfor

  ///////////////////
  //   CROSSBARS   //
  ///////////////////
  ${module}

  /////////////////////////////
  // Narrow to Wide Crossbar //
  /////////////////////////////
  <% soc_narrow_xbar.out_soc_wide \
        .change_iw(context, soc_wide_xbar.in_soc_narrow.iw, "soc_narrow_wide_iwc") \
        .atomic_adapter(context, 16, "soc_narrow_wide_amo_adapter") \
        .change_dw(context, soc_wide_xbar.in_soc_narrow.dw, "soc_narrow_wide_dw", to=soc_wide_xbar.in_soc_narrow)
  %>

  /////////////////////////////
  // Wide to Narrow Crossbar //
  /////////////////////////////
  <%
    soc_wide_xbar.out_soc_narrow \
      .change_iw(context, soc_narrow_xbar.in_soc_wide.iw, "soc_wide_narrow_iwc") \
      .change_dw(context, soc_narrow_xbar.in_soc_wide.dw, "soc_wide_narrow_dw", to=soc_narrow_xbar.in_soc_wide)
  %>

  //////////
  // PCIe //
  //////////
  <%
    pcie_cuts = 3
    pcie_out = soc_wide_xbar.__dict__["out_pcie"].cut(context, pcie_cuts, name="pcie_out", inst_name="i_pcie_out_cut")
    pcie_in = soc_wide_xbar.__dict__["in_pcie"].copy(name="pcie_in").declare(context)
    pcie_in.cut(context, pcie_cuts, to=soc_wide_xbar.__dict__["in_pcie"])
  %>\

  assign pcie_axi_req_o = ${pcie_out.req_name()};
  assign ${pcie_out.rsp_name()} = pcie_axi_rsp_i;
  assign ${pcie_in.req_name()} = pcie_axi_req_i;
  assign pcie_axi_rsp_o = ${pcie_in.rsp_name()};

  //////////
  // CVA6 //
  //////////
  <%
    cva6_cuts = 1
    cva6_mst = soc_narrow_xbar.__dict__["in_cva6"].copy(name="cva6_mst").declare(context)
    cva6_mst.cut(context, cva6_cuts, to=soc_narrow_xbar.__dict__["in_cva6"])
  %>\

  occamy_cva6 i_occamy_cva6 (
    .clk_i (clk_i),
    .rst_ni (rst_ni),
    .irq_i (eip_i),
    .ipi_i (msip_i[0]),
    .time_irq_i (mtip_i[0]),
    .debug_req_i (debug_req_i[0]),
    .axi_req_o (${cva6_mst.req_name()}),
    .axi_resp_i (${cva6_mst.rsp_name()}),
    .sram_cfg_i (sram_cfgs_i.cva6)
  );

  % for i in range(nr_s1_quadrants):
  ///////////////////
  // S1 Quadrant ${i} //
  ///////////////////
  <%
    quad_widex_cuts = 3
    quad_hbi_cuts = 6
    narrow_in = soc_narrow_xbar.__dict__["out_s1_quadrant_{}".format(i)].cut(context, quad_widex_cuts, name="narrow_in_cut_{}".format(i))
    narrow_out = soc_narrow_xbar.__dict__["in_s1_quadrant_{}".format(i)].copy(name="narrow_out_cut_{}".format(i)).declare(context)
    narrow_out.cut(context, quad_widex_cuts, to=soc_narrow_xbar.__dict__["in_s1_quadrant_{}".format(i)])
    wide_in = soc_wide_xbar.__dict__["out_s1_quadrant_{}".format(i)].cut(context, quad_widex_cuts, name="wide_in_cut_{}".format(i))
    wide_out = soc_wide_xbar.__dict__["in_s1_quadrant_{}".format(i)].copy(name="wide_out_cut_{}".format(i)).declare(context)
    wide_out.cut(context, quad_widex_cuts, to=soc_wide_xbar.__dict__["in_s1_quadrant_{}".format(i)])
    wide_hbi_out = wide_xbar_quadrant_s1.out_hbi.copy(name="wide_hbi_out_cut_{}".format(i), clk="clk_i", rst="rst_ni").declare(context)
    wide_hbi_cut_out = wide_hbi_out.cut(context, quad_hbi_cuts)
  %>
  assign hbi_${i}_req_o = ${wide_hbi_cut_out.req_name()};
  assign ${wide_hbi_cut_out.rsp_name()} = hbi_${i}_rsp_i;

  <%
    nr_cores_s1_quadrant = nr_s1_clusters * nr_cluster_cores
    lower_core = i * nr_cores_s1_quadrant + 1
    ro_addr_regions = cfg["s1_quadrant"].get("ro_cache_cfg", {}).get("address_regions", 1)
  %>

  occamy_quadrant_s1 i_occamy_quadrant_s1_${i} (
    .clk_i (clk_i),
    .rst_ni (rst_ni),
    .test_mode_i (test_mode_i),
    .tile_id_i (6'd${i}),
    // .debug_req_i (debug_req_i[${lower_core + nr_cores_s1_quadrant - 1}:${lower_core}]),
    .debug_req_i ('0),
    .meip_i ('0),
    .mtip_i (mtip_i[${lower_core + nr_cores_s1_quadrant - 1}:${lower_core}]),
    .msip_i (msip_i[${lower_core + nr_cores_s1_quadrant - 1}:${lower_core}]),
    .quadrant_hbi_out_req_o (${wide_hbi_out.req_name()}),
    .quadrant_hbi_out_rsp_i (${wide_hbi_out.rsp_name()}),
    .quadrant_narrow_out_req_o (${narrow_out.req_name()}),
    .quadrant_narrow_out_rsp_i (${narrow_out.rsp_name()}),
    .quadrant_narrow_in_req_i (${narrow_in.req_name()}),
    .quadrant_narrow_in_rsp_o (${narrow_in.rsp_name()}),
    .quadrant_wide_out_req_o (${wide_out.req_name()}),
    .quadrant_wide_out_rsp_i (${wide_out.rsp_name()}),
    .quadrant_wide_in_req_i (${wide_in.req_name()}),
    .quadrant_wide_in_rsp_o (${wide_in.rsp_name()}),
    .sram_cfg_i (sram_cfgs_i.quadrant)
  );

  % endfor

  //////////
  // SPM //
  //////////

  <% narrow_spm_mst = soc_narrow_xbar.out_spm \
                      .serialize(context, "spm_serialize", iw=1) \
                      .atomic_adapter(context, 16, "spm_amo_adapter") \
                      .cut(context, 1)
  %>

  axi_to_mem #(
    .axi_req_t (${narrow_spm_mst.req_type()}),
    .axi_resp_t (${narrow_spm_mst.rsp_type()}),
    .AddrWidth (${util.clog2(spm_words) + util.clog2(narrow_spm_mst.dw//8)}),
    .DataWidth (${narrow_spm_mst.dw}),
    .IdWidth (${narrow_spm_mst.iw}),
    .NumBanks (1),
    .BufDepth (1)
  ) i_axi_to_mem (
    .clk_i (${narrow_spm_mst.clk}),
    .rst_ni (${narrow_spm_mst.rst}),
    .busy_o (),
    .axi_req_i (${narrow_spm_mst.req_name()}),
    .axi_resp_o (${narrow_spm_mst.rsp_name()}),
    .mem_req_o (spm_req),
    .mem_gnt_i (spm_gnt),
    .mem_addr_o (spm_addr),
    .mem_wdata_o (spm_wdata),
    .mem_strb_o (spm_strb),
    .mem_atop_o (),
    .mem_we_o (spm_we),
    .mem_rvalid_i (spm_rvalid),
    .mem_rdata_i (spm_rdata)
  );

  spm_1p_adv #(
    .NumWords (${spm_words}),
    .DataWidth (${narrow_spm_mst.dw}),
    .ByteWidth (8),
    .EnableInputPipeline (1'b1),
    .EnableOutputPipeline (1'b1),
    .sram_cfg_t (sram_cfg_t)
  ) i_spm_cut (
    .clk_i (${narrow_spm_mst.clk}),
    .rst_ni (${narrow_spm_mst.rst}),
    .valid_i (spm_req),
    .ready_o (spm_gnt),
    .we_i (spm_we),
    .addr_i (spm_addr[${util.clog2(spm_words) + util.clog2(narrow_spm_mst.dw//8)-1}:${util.clog2(narrow_spm_mst.dw//8)}]),
    .wdata_i (spm_wdata),
    .be_i (spm_strb),
    .rdata_o (spm_rdata),
    .rvalid_o (spm_rvalid),
    .rerror_o (spm_rerror_o),
    .sram_cfg_i (sram_cfgs_i.spm)
  );

  ///////////
  // HBM2e //
  ///////////

% for i in range(8):
  <%
    hbm_cuts = 3
    hbm_out = soc_wide_xbar.__dict__["out_hbm_{}".format(i)].cut(
      context, hbm_cuts, name="hbm_out_{}".format(i), inst_name="i_hbm_out_cut_{}".format(i))
  %>\

  assign hbm_${i}_req_o = ${hbm_out.req_name()};
  assign ${hbm_out.rsp_name()} = hbm_${i}_rsp_i;

% endfor

  /////////
  // HBI //
  /////////

  // Inputs from HBI to wide Xbar
% for i in range(nr_s1_quadrants+1):
  <%
    hbi_in_cuts = 6
    hbi_in = soc_wide_xbar.__dict__["in_hbi_{}".format(i)].copy(name="in_hbi_{}".format(i)).declare(context)
    hbi_in_trunc = hbi_in.copy(name="in_hbi_trunc_{}".format(i)).declare(context)
    hbi_in.trunc_addr(context, target_aw=40, inst_name="hbi_in_trunc_addr_{}".format(i), to=hbi_in_trunc)
    hbi_in_trunc.cut(context, hbi_in_cuts, to=soc_wide_xbar.__dict__["in_hbi_{}".format(i)])
  %>
  assign ${hbi_in.req_name()} = hbi_${i}_req_i;
  assign hbi_${i}_rsp_o = ${hbi_in.rsp_name()};

% endfor
  // Single output from wide Xbar to HBI
  <%
    hbi_widex_cuts = 6
    soc_wide_hbi_iwc = soc_wide_xbar.__dict__["out_hbi_{}".format(nr_s1_quadrants)] \
        .change_iw(context, wide_xbar_quadrant_s1.out_hbi.iw, "soc_wide_hbi_iwc") \
        .cut(context, hbi_widex_cuts)
  %>
  assign hbi_${nr_s1_quadrants}_req_o = ${soc_wide_hbi_iwc.req_name()};
  assign ${soc_wide_hbi_iwc.rsp_name()} = hbi_${nr_s1_quadrants}_rsp_i;

  /////////////////
  // Peripherals //
  /////////////////
  <%
    periph_regbus_cuts = 3
    periph_axi_lite_cuts = 3
    periph_regbus_out = soc_narrow_xbar.__dict__["out_regbus_periph"].cut(context,
      periph_regbus_cuts, name="periph_regbus_out", inst_name="i_periph_regbus_out_cut")
    periph_axi_lite_out = soc_narrow_xbar.__dict__["out_periph"].cut(context,
      periph_axi_lite_cuts, name="periph_axi_lite_out", inst_name="i_periph_axi_lite_out_cut")
    periph_axi_lite_in = soc_narrow_xbar.__dict__["in_periph"].copy(name="periph_axi_lite_in").declare(context)
    periph_axi_lite_in.cut(context, periph_axi_lite_cuts, to=soc_narrow_xbar.__dict__["in_periph"])
  %>\

  // Inputs
  assign ${periph_axi_lite_in.req_name()} = periph_axi_lite_req_i;
  assign periph_axi_lite_rsp_o = ${periph_axi_lite_in.rsp_name()};

  // Outputs
  assign periph_axi_lite_req_o = ${periph_axi_lite_out.req_name()};
  assign ${periph_axi_lite_out.rsp_name()} = periph_axi_lite_rsp_i;
  assign periph_regbus_req_o = ${periph_regbus_out.req_name()};
  assign ${periph_regbus_out.rsp_name()} = periph_regbus_rsp_i;

endmodule
