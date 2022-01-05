// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>
// Author: Fabian Schuiki <fschuiki@iis.ee.ethz.ch>

// AUTOMATICALLY GENERATED by occamygen.py; edit the script instead.


/// Occamy Stage 1 Quadrant
module occamy_quadrant_s1
  import occamy_pkg::*;
(
    input  logic                                             clk_i,
    input  logic                                             rst_ni,
    input  logic                                             test_mode_i,
    input  tile_id_t                                         tile_id_i,
    input  logic                     [NrCoresS1Quadrant-1:0] debug_req_i,
    input  logic                     [NrCoresS1Quadrant-1:0] meip_i,
    input  logic                     [NrCoresS1Quadrant-1:0] mtip_i,
    input  logic                     [NrCoresS1Quadrant-1:0] msip_i,
    // HBI Connection
    output axi_a48_d512_i5_u0_req_t                          quadrant_hbi_out_req_o,
    input  axi_a48_d512_i5_u0_resp_t                         quadrant_hbi_out_rsp_i,
    // Next-Level
    output axi_a48_d64_i4_u0_req_t                           quadrant_narrow_out_req_o,
    input  axi_a48_d64_i4_u0_resp_t                          quadrant_narrow_out_rsp_i,
    input  axi_a48_d64_i8_u0_req_t                           quadrant_narrow_in_req_i,
    output axi_a48_d64_i8_u0_resp_t                          quadrant_narrow_in_rsp_o,
    output axi_a48_d512_i4_u0_req_t                          quadrant_wide_out_req_o,
    input  axi_a48_d512_i4_u0_resp_t                         quadrant_wide_out_rsp_i,
    input  axi_a48_d512_i9_u0_req_t                          quadrant_wide_in_req_i,
    output axi_a48_d512_i9_u0_resp_t                         quadrant_wide_in_rsp_o,
    // SRAM configuration
    input  sram_cfg_quadrant_t                               sram_cfg_i
);

  // Calculate cluster base address based on `tile id`.
  addr_t [3:0] cluster_base_addr;
  assign cluster_base_addr[0] = ClusterBaseOffset + tile_id_i * NrClustersS1Quadrant * ClusterAddressSpace + 0 * ClusterAddressSpace;
  assign cluster_base_addr[1] = ClusterBaseOffset + tile_id_i * NrClustersS1Quadrant * ClusterAddressSpace + 1 * ClusterAddressSpace;
  assign cluster_base_addr[2] = ClusterBaseOffset + tile_id_i * NrClustersS1Quadrant * ClusterAddressSpace + 2 * ClusterAddressSpace;
  assign cluster_base_addr[3] = ClusterBaseOffset + tile_id_i * NrClustersS1Quadrant * ClusterAddressSpace + 3 * ClusterAddressSpace;

  // Signals from Controller
  logic clk_quadrant, rst_quadrant_n;
  logic [4:0] isolate, isolated;
  logic ro_enable, ro_flush_valid, ro_flush_ready;
  logic [3:0][47:0] ro_start_addr, ro_end_addr;

  ///////////////////
  //   CROSSBARS   //
  ///////////////////

  /// Address map of the `wide_xbar_quadrant_s1` crossbar.
  xbar_rule_48_t [4:0] WideXbarQuadrantS1Addrmap;
  assign WideXbarQuadrantS1Addrmap = '{
  '{ idx: 1, start_addr: 48'h10000000000, end_addr: 48'h20000000000 },
  '{ idx: 2, start_addr: cluster_base_addr[0], end_addr: cluster_base_addr[0] + ClusterAddressSpace },
  '{ idx: 3, start_addr: cluster_base_addr[1], end_addr: cluster_base_addr[1] + ClusterAddressSpace },
  '{ idx: 4, start_addr: cluster_base_addr[2], end_addr: cluster_base_addr[2] + ClusterAddressSpace },
  '{ idx: 5, start_addr: cluster_base_addr[3], end_addr: cluster_base_addr[3] + ClusterAddressSpace }
};

  wide_xbar_quadrant_s1_in_req_t   [4:0] wide_xbar_quadrant_s1_in_req;
  wide_xbar_quadrant_s1_in_resp_t  [4:0] wide_xbar_quadrant_s1_in_rsp;
  wide_xbar_quadrant_s1_out_req_t  [5:0] wide_xbar_quadrant_s1_out_req;
  wide_xbar_quadrant_s1_out_resp_t [5:0] wide_xbar_quadrant_s1_out_rsp;

  axi_xbar #(
      .Cfg          (WideXbarQuadrantS1Cfg),
      .Connectivity (30'b011111101111110111111011111110),
      .AtopSupport  (0),
      .slv_aw_chan_t(axi_a48_d512_i2_u0_aw_chan_t),
      .mst_aw_chan_t(axi_a48_d512_i5_u0_aw_chan_t),
      .w_chan_t     (axi_a48_d512_i2_u0_w_chan_t),
      .slv_b_chan_t (axi_a48_d512_i2_u0_b_chan_t),
      .mst_b_chan_t (axi_a48_d512_i5_u0_b_chan_t),
      .slv_ar_chan_t(axi_a48_d512_i2_u0_ar_chan_t),
      .mst_ar_chan_t(axi_a48_d512_i5_u0_ar_chan_t),
      .slv_r_chan_t (axi_a48_d512_i2_u0_r_chan_t),
      .mst_r_chan_t (axi_a48_d512_i5_u0_r_chan_t),
      .slv_req_t    (axi_a48_d512_i2_u0_req_t),
      .slv_resp_t   (axi_a48_d512_i2_u0_resp_t),
      .mst_req_t    (axi_a48_d512_i5_u0_req_t),
      .mst_resp_t   (axi_a48_d512_i5_u0_resp_t),
      .rule_t       (xbar_rule_48_t)
  ) i_wide_xbar_quadrant_s1 (
      .clk_i                (clk_quadrant),
      .rst_ni               (rst_quadrant_n),
      .test_i               (test_mode_i),
      .slv_ports_req_i      (wide_xbar_quadrant_s1_in_req),
      .slv_ports_resp_o     (wide_xbar_quadrant_s1_in_rsp),
      .mst_ports_req_o      (wide_xbar_quadrant_s1_out_req),
      .mst_ports_resp_i     (wide_xbar_quadrant_s1_out_rsp),
      .addr_map_i           (WideXbarQuadrantS1Addrmap),
      .en_default_mst_port_i('1),
      .default_mst_port_i   ('0)
  );

  /// Address map of the `narrow_xbar_quadrant_s1` crossbar.
  xbar_rule_48_t [3:0] NarrowXbarQuadrantS1Addrmap;
  assign NarrowXbarQuadrantS1Addrmap = '{
  '{ idx: 1, start_addr: cluster_base_addr[0], end_addr: cluster_base_addr[0] + ClusterAddressSpace },
  '{ idx: 2, start_addr: cluster_base_addr[1], end_addr: cluster_base_addr[1] + ClusterAddressSpace },
  '{ idx: 3, start_addr: cluster_base_addr[2], end_addr: cluster_base_addr[2] + ClusterAddressSpace },
  '{ idx: 4, start_addr: cluster_base_addr[3], end_addr: cluster_base_addr[3] + ClusterAddressSpace }
};

  narrow_xbar_quadrant_s1_in_req_t   [4:0] narrow_xbar_quadrant_s1_in_req;
  narrow_xbar_quadrant_s1_in_resp_t  [4:0] narrow_xbar_quadrant_s1_in_rsp;
  narrow_xbar_quadrant_s1_out_req_t  [4:0] narrow_xbar_quadrant_s1_out_req;
  narrow_xbar_quadrant_s1_out_resp_t [4:0] narrow_xbar_quadrant_s1_out_rsp;

  axi_xbar #(
      .Cfg          (NarrowXbarQuadrantS1Cfg),
      .Connectivity (25'b0111110111110111110111110),
      .AtopSupport  (1),
      .slv_aw_chan_t(axi_a48_d64_i4_u0_aw_chan_t),
      .mst_aw_chan_t(axi_a48_d64_i7_u0_aw_chan_t),
      .w_chan_t     (axi_a48_d64_i4_u0_w_chan_t),
      .slv_b_chan_t (axi_a48_d64_i4_u0_b_chan_t),
      .mst_b_chan_t (axi_a48_d64_i7_u0_b_chan_t),
      .slv_ar_chan_t(axi_a48_d64_i4_u0_ar_chan_t),
      .mst_ar_chan_t(axi_a48_d64_i7_u0_ar_chan_t),
      .slv_r_chan_t (axi_a48_d64_i4_u0_r_chan_t),
      .mst_r_chan_t (axi_a48_d64_i7_u0_r_chan_t),
      .slv_req_t    (axi_a48_d64_i4_u0_req_t),
      .slv_resp_t   (axi_a48_d64_i4_u0_resp_t),
      .mst_req_t    (axi_a48_d64_i7_u0_req_t),
      .mst_resp_t   (axi_a48_d64_i7_u0_resp_t),
      .rule_t       (xbar_rule_48_t)
  ) i_narrow_xbar_quadrant_s1 (
      .clk_i                (clk_quadrant),
      .rst_ni               (rst_quadrant_n),
      .test_i               (test_mode_i),
      .slv_ports_req_i      (narrow_xbar_quadrant_s1_in_req),
      .slv_ports_resp_o     (narrow_xbar_quadrant_s1_in_rsp),
      .mst_ports_req_o      (narrow_xbar_quadrant_s1_out_req),
      .mst_ports_resp_i     (narrow_xbar_quadrant_s1_out_rsp),
      .addr_map_i           (NarrowXbarQuadrantS1Addrmap),
      .en_default_mst_port_i('1),
      .default_mst_port_i   ('0)
  );


  ///////////////////////////////
  // Narrow In + IW Converter //
  ///////////////////////////////
  axi_a48_d64_i8_u0_req_t  narrow_cluster_in_iwc_req;
  axi_a48_d64_i8_u0_resp_t narrow_cluster_in_iwc_rsp;

  axi_a48_d64_i8_u0_req_t  narrow_cluster_in_iwc_cut_req;
  axi_a48_d64_i8_u0_resp_t narrow_cluster_in_iwc_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d64_i8_u0_aw_chan_t),
      .w_chan_t(axi_a48_d64_i8_u0_w_chan_t),
      .b_chan_t(axi_a48_d64_i8_u0_b_chan_t),
      .ar_chan_t(axi_a48_d64_i8_u0_ar_chan_t),
      .r_chan_t(axi_a48_d64_i8_u0_r_chan_t),
      .req_t(axi_a48_d64_i8_u0_req_t),
      .resp_t(axi_a48_d64_i8_u0_resp_t)
  ) i_narrow_cluster_in_iwc_cut (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(narrow_cluster_in_iwc_req),
      .slv_resp_o(narrow_cluster_in_iwc_rsp),
      .mst_req_o(narrow_cluster_in_iwc_cut_req),
      .mst_resp_i(narrow_cluster_in_iwc_cut_rsp)
  );
  axi_a48_d64_i8_u0_req_t  narrow_cluster_in_isolate_req;
  axi_a48_d64_i8_u0_resp_t narrow_cluster_in_isolate_rsp;

  axi_isolate #(
      .NumPending(16),
      .TerminateTransaction(1),
      .AtopSupport(1),
      .AxiIdWidth(8),
      .AxiAddrWidth(48),
      .AxiDataWidth(64),
      .AxiUserWidth(1),
      .req_t(axi_a48_d64_i8_u0_req_t),
      .resp_t(axi_a48_d64_i8_u0_resp_t)
  ) i_narrow_cluster_in_isolate (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(narrow_cluster_in_iwc_cut_req),
      .slv_resp_o(narrow_cluster_in_iwc_cut_rsp),
      .mst_req_o(narrow_cluster_in_isolate_req),
      .mst_resp_i(narrow_cluster_in_isolate_rsp),
      .isolate_i(isolate[0]),
      .isolated_o(isolated[0])
  );

  axi_id_remap #(
      .AxiSlvPortIdWidth(8),
      .AxiSlvPortMaxUniqIds(16),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(4),
      .slv_req_t(axi_a48_d64_i8_u0_req_t),
      .slv_resp_t(axi_a48_d64_i8_u0_resp_t),
      .mst_req_t(axi_a48_d64_i4_u0_req_t),
      .mst_resp_t(axi_a48_d64_i4_u0_resp_t)
  ) i_narrow_cluster_in_iwc (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_cluster_in_isolate_req),
      .slv_resp_o(narrow_cluster_in_isolate_rsp),
      .mst_req_o(narrow_xbar_quadrant_s1_in_req[NARROW_XBAR_QUADRANT_S1_IN_TOP]),
      .mst_resp_i(narrow_xbar_quadrant_s1_in_rsp[NARROW_XBAR_QUADRANT_S1_IN_TOP])
  );

  assign narrow_cluster_in_iwc_req = quadrant_narrow_in_req_i;
  assign quadrant_narrow_in_rsp_o  = narrow_cluster_in_iwc_rsp;

  ///////////////////////////////
  // Narrow Out + IW Converter //
  ///////////////////////////////
  axi_a48_d64_i4_u0_req_t  narrow_cluster_out_iwc_req;
  axi_a48_d64_i4_u0_resp_t narrow_cluster_out_iwc_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(7),
      .AxiSlvPortMaxUniqIds(16),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(4),
      .slv_req_t(axi_a48_d64_i7_u0_req_t),
      .slv_resp_t(axi_a48_d64_i7_u0_resp_t),
      .mst_req_t(axi_a48_d64_i4_u0_req_t),
      .mst_resp_t(axi_a48_d64_i4_u0_resp_t)
  ) i_narrow_cluster_out_iwc (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_xbar_quadrant_s1_out_req[NARROW_XBAR_QUADRANT_S1_OUT_TOP]),
      .slv_resp_o(narrow_xbar_quadrant_s1_out_rsp[NARROW_XBAR_QUADRANT_S1_OUT_TOP]),
      .mst_req_o(narrow_cluster_out_iwc_req),
      .mst_resp_i(narrow_cluster_out_iwc_rsp)
  );
  axi_a48_d64_i4_u0_req_t  narrow_cluster_out_isolate_req;
  axi_a48_d64_i4_u0_resp_t narrow_cluster_out_isolate_rsp;

  axi_isolate #(
      .NumPending(16),
      .TerminateTransaction(0),
      .AtopSupport(1),
      .AxiIdWidth(4),
      .AxiAddrWidth(48),
      .AxiDataWidth(64),
      .AxiUserWidth(1),
      .req_t(axi_a48_d64_i4_u0_req_t),
      .resp_t(axi_a48_d64_i4_u0_resp_t)
  ) i_narrow_cluster_out_isolate (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_cluster_out_iwc_req),
      .slv_resp_o(narrow_cluster_out_iwc_rsp),
      .mst_req_o(narrow_cluster_out_isolate_req),
      .mst_resp_i(narrow_cluster_out_isolate_rsp),
      .isolate_i(isolate[1]),
      .isolated_o(isolated[1])
  );

  axi_a48_d64_i4_u0_req_t  narrow_cluster_out_isolate_cut_req;
  axi_a48_d64_i4_u0_resp_t narrow_cluster_out_isolate_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d64_i4_u0_aw_chan_t),
      .w_chan_t(axi_a48_d64_i4_u0_w_chan_t),
      .b_chan_t(axi_a48_d64_i4_u0_b_chan_t),
      .ar_chan_t(axi_a48_d64_i4_u0_ar_chan_t),
      .r_chan_t(axi_a48_d64_i4_u0_r_chan_t),
      .req_t(axi_a48_d64_i4_u0_req_t),
      .resp_t(axi_a48_d64_i4_u0_resp_t)
  ) i_narrow_cluster_out_isolate_cut (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(narrow_cluster_out_isolate_req),
      .slv_resp_o(narrow_cluster_out_isolate_rsp),
      .mst_req_o(narrow_cluster_out_isolate_cut_req),
      .mst_resp_i(narrow_cluster_out_isolate_cut_rsp)
  );


  assign quadrant_narrow_out_req_o = narrow_cluster_out_isolate_cut_req;
  assign narrow_cluster_out_isolate_cut_rsp = quadrant_narrow_out_rsp_i;

  /////////////////////////////////////////
  // Wide Out + RO Cache + IW Converter  //
  /////////////////////////////////////////
  axi_a48_d512_i6_u0_req_t  snitch_ro_cache_req;
  axi_a48_d512_i6_u0_resp_t snitch_ro_cache_rsp;

  snitch_read_only_cache #(
      .LineWidth(1024),
      .LineCount(128),
      .SetCount(2),
      .AxiAddrWidth(48),
      .AxiDataWidth(512),
      .AxiIdWidth(5),
      .AxiUserWidth(1),
      .MaxTrans(8),
      .NrAddrRules(4),
      .slv_req_t(axi_a48_d512_i5_u0_req_t),
      .slv_rsp_t(axi_a48_d512_i5_u0_resp_t),
      .mst_req_t(axi_a48_d512_i6_u0_req_t),
      .mst_rsp_t(axi_a48_d512_i6_u0_resp_t),
      .sram_cfg_data_t(sram_cfg_t),
      .sram_cfg_tag_t(sram_cfg_t)
  ) i_snitch_ro_cache (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .enable_i(ro_enable),
      .flush_valid_i(ro_flush_valid),
      .flush_ready_o(ro_flush_ready),
      .start_addr_i(ro_start_addr),
      .end_addr_i(ro_end_addr),
      .axi_slv_req_i(wide_xbar_quadrant_s1_out_req[WIDE_XBAR_QUADRANT_S1_OUT_TOP]),
      .axi_slv_rsp_o(wide_xbar_quadrant_s1_out_rsp[WIDE_XBAR_QUADRANT_S1_OUT_TOP]),
      .axi_mst_req_o(snitch_ro_cache_req),
      .axi_mst_rsp_i(snitch_ro_cache_rsp),
      .sram_cfg_data_i(sram_cfg_i.rocache_data),
      .sram_cfg_tag_i(sram_cfg_i.rocache_tag)
  );

  axi_a48_d512_i4_u0_req_t  wide_cluster_out_iwc_req;
  axi_a48_d512_i4_u0_resp_t wide_cluster_out_iwc_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(6),
      .AxiSlvPortMaxUniqIds(16),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(4),
      .slv_req_t(axi_a48_d512_i6_u0_req_t),
      .slv_resp_t(axi_a48_d512_i6_u0_resp_t),
      .mst_req_t(axi_a48_d512_i4_u0_req_t),
      .mst_resp_t(axi_a48_d512_i4_u0_resp_t)
  ) i_wide_cluster_out_iwc (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(snitch_ro_cache_req),
      .slv_resp_o(snitch_ro_cache_rsp),
      .mst_req_o(wide_cluster_out_iwc_req),
      .mst_resp_i(wide_cluster_out_iwc_rsp)
  );
  axi_a48_d512_i4_u0_req_t  wide_cluster_out_isolate_req;
  axi_a48_d512_i4_u0_resp_t wide_cluster_out_isolate_rsp;

  axi_isolate #(
      .NumPending(16),
      .TerminateTransaction(0),
      .AtopSupport(0),
      .AxiIdWidth(4),
      .AxiAddrWidth(48),
      .AxiDataWidth(512),
      .AxiUserWidth(1),
      .req_t(axi_a48_d512_i4_u0_req_t),
      .resp_t(axi_a48_d512_i4_u0_resp_t)
  ) i_wide_cluster_out_isolate (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_cluster_out_iwc_req),
      .slv_resp_o(wide_cluster_out_iwc_rsp),
      .mst_req_o(wide_cluster_out_isolate_req),
      .mst_resp_i(wide_cluster_out_isolate_rsp),
      .isolate_i(isolate[3]),
      .isolated_o(isolated[3])
  );

  axi_a48_d512_i4_u0_req_t  wide_cluster_out_isolate_cut_req;
  axi_a48_d512_i4_u0_resp_t wide_cluster_out_isolate_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d512_i4_u0_aw_chan_t),
      .w_chan_t(axi_a48_d512_i4_u0_w_chan_t),
      .b_chan_t(axi_a48_d512_i4_u0_b_chan_t),
      .ar_chan_t(axi_a48_d512_i4_u0_ar_chan_t),
      .r_chan_t(axi_a48_d512_i4_u0_r_chan_t),
      .req_t(axi_a48_d512_i4_u0_req_t),
      .resp_t(axi_a48_d512_i4_u0_resp_t)
  ) i_wide_cluster_out_isolate_cut (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(wide_cluster_out_isolate_req),
      .slv_resp_o(wide_cluster_out_isolate_rsp),
      .mst_req_o(wide_cluster_out_isolate_cut_req),
      .mst_resp_i(wide_cluster_out_isolate_cut_rsp)
  );


  assign quadrant_wide_out_req_o = wide_cluster_out_isolate_cut_req;
  assign wide_cluster_out_isolate_cut_rsp = quadrant_wide_out_rsp_i;

  ////////////////////
  // HBI Connection //
  ////////////////////
  axi_a48_d512_i5_u0_req_t  quadrant_hbi_out_isolate_req;
  axi_a48_d512_i5_u0_resp_t quadrant_hbi_out_isolate_rsp;

  axi_isolate #(
      .NumPending(16),
      .TerminateTransaction(0),
      .AtopSupport(0),
      .AxiIdWidth(5),
      .AxiAddrWidth(48),
      .AxiDataWidth(512),
      .AxiUserWidth(1),
      .req_t(axi_a48_d512_i5_u0_req_t),
      .resp_t(axi_a48_d512_i5_u0_resp_t)
  ) i_quadrant_hbi_out_isolate (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_xbar_quadrant_s1_out_req[WIDE_XBAR_QUADRANT_S1_OUT_HBI]),
      .slv_resp_o(wide_xbar_quadrant_s1_out_rsp[WIDE_XBAR_QUADRANT_S1_OUT_HBI]),
      .mst_req_o(quadrant_hbi_out_isolate_req),
      .mst_resp_i(quadrant_hbi_out_isolate_rsp),
      .isolate_i(isolate[4]),
      .isolated_o(isolated[4])
  );

  axi_a48_d512_i5_u0_req_t  quadrant_hbi_out_isolate_cut_req;
  axi_a48_d512_i5_u0_resp_t quadrant_hbi_out_isolate_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d512_i5_u0_aw_chan_t),
      .w_chan_t(axi_a48_d512_i5_u0_w_chan_t),
      .b_chan_t(axi_a48_d512_i5_u0_b_chan_t),
      .ar_chan_t(axi_a48_d512_i5_u0_ar_chan_t),
      .r_chan_t(axi_a48_d512_i5_u0_r_chan_t),
      .req_t(axi_a48_d512_i5_u0_req_t),
      .resp_t(axi_a48_d512_i5_u0_resp_t)
  ) i_quadrant_hbi_out_isolate_cut (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(quadrant_hbi_out_isolate_req),
      .slv_resp_o(quadrant_hbi_out_isolate_rsp),
      .mst_req_o(quadrant_hbi_out_isolate_cut_req),
      .mst_resp_i(quadrant_hbi_out_isolate_cut_rsp)
  );


  assign quadrant_hbi_out_req_o = quadrant_hbi_out_isolate_cut_req;
  assign quadrant_hbi_out_isolate_cut_rsp = quadrant_hbi_out_rsp_i;

  ////////////////////////////
  // Wide In + IW Converter //
  ////////////////////////////
  axi_a48_d512_i9_u0_req_t  wide_cluster_in_iwc_req;
  axi_a48_d512_i9_u0_resp_t wide_cluster_in_iwc_rsp;

  axi_a48_d512_i9_u0_req_t  wide_cluster_in_iwc_cut_req;
  axi_a48_d512_i9_u0_resp_t wide_cluster_in_iwc_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d512_i9_u0_aw_chan_t),
      .w_chan_t(axi_a48_d512_i9_u0_w_chan_t),
      .b_chan_t(axi_a48_d512_i9_u0_b_chan_t),
      .ar_chan_t(axi_a48_d512_i9_u0_ar_chan_t),
      .r_chan_t(axi_a48_d512_i9_u0_r_chan_t),
      .req_t(axi_a48_d512_i9_u0_req_t),
      .resp_t(axi_a48_d512_i9_u0_resp_t)
  ) i_wide_cluster_in_iwc_cut (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(wide_cluster_in_iwc_req),
      .slv_resp_o(wide_cluster_in_iwc_rsp),
      .mst_req_o(wide_cluster_in_iwc_cut_req),
      .mst_resp_i(wide_cluster_in_iwc_cut_rsp)
  );
  axi_a48_d512_i9_u0_req_t  wide_cluster_in_isolate_req;
  axi_a48_d512_i9_u0_resp_t wide_cluster_in_isolate_rsp;

  axi_isolate #(
      .NumPending(16),
      .TerminateTransaction(1),
      .AtopSupport(0),
      .AxiIdWidth(9),
      .AxiAddrWidth(48),
      .AxiDataWidth(512),
      .AxiUserWidth(1),
      .req_t(axi_a48_d512_i9_u0_req_t),
      .resp_t(axi_a48_d512_i9_u0_resp_t)
  ) i_wide_cluster_in_isolate (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(wide_cluster_in_iwc_cut_req),
      .slv_resp_o(wide_cluster_in_iwc_cut_rsp),
      .mst_req_o(wide_cluster_in_isolate_req),
      .mst_resp_i(wide_cluster_in_isolate_rsp),
      .isolate_i(isolate[2]),
      .isolated_o(isolated[2])
  );

  axi_id_remap #(
      .AxiSlvPortIdWidth(9),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d512_i9_u0_req_t),
      .slv_resp_t(axi_a48_d512_i9_u0_resp_t),
      .mst_req_t(axi_a48_d512_i2_u0_req_t),
      .mst_resp_t(axi_a48_d512_i2_u0_resp_t)
  ) i_wide_cluster_in_iwc (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_cluster_in_isolate_req),
      .slv_resp_o(wide_cluster_in_isolate_rsp),
      .mst_req_o(wide_xbar_quadrant_s1_in_req[WIDE_XBAR_QUADRANT_S1_IN_TOP]),
      .mst_resp_i(wide_xbar_quadrant_s1_in_rsp[WIDE_XBAR_QUADRANT_S1_IN_TOP])
  );

  assign wide_cluster_in_iwc_req = quadrant_wide_in_req_i;
  assign quadrant_wide_in_rsp_o  = wide_cluster_in_iwc_rsp;

  /////////////////////////
  // Quadrant Controller //
  /////////////////////////

  occamy_quadrant_s1_ctrl i_occamy_quadrant_s1_ctrl (
      .clk_i,
      .rst_ni,
      .test_mode_i,
      .clk_quadrant_o(clk_quadrant),
      .rst_quadrant_no(rst_quadrant_n),
      .isolate_o(isolate),
      .isolated_i(isolated),
      .ro_enable_o(ro_enable),
      .ro_flush_valid_o(ro_flush_valid),
      .ro_flush_ready_i(ro_flush_ready),
      .ro_start_addr_o(ro_start_addr),
      .ro_end_addr_o(ro_end_addr),
      .soc_out_req_o(quadrant_narrow_out_req_o),
      .soc_out_rsp_i(quadrant_narrow_out_rsp_i),
      .soc_in_req_i(quadrant_narrow_in_req_i),
      .soc_in_rsp_o(quadrant_narrow_in_rsp_o),
      .quadrant_out_req_o(narrow_cluster_out_isolate_cut_req),
      .quadrant_out_rsp_i(narrow_cluster_out_isolate_cut_rsp),
      .quadrant_in_req_i(narrow_cluster_in_iwc_req),
      .quadrant_in_rsp_o(narrow_cluster_in_iwc_rsp)
  );

  ///////////////
  // Cluster 0 //
  ///////////////
  axi_a48_d64_i2_u0_req_t  narrow_in_iwc_0_req;
  axi_a48_d64_i2_u0_resp_t narrow_in_iwc_0_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(7),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d64_i7_u0_req_t),
      .slv_resp_t(axi_a48_d64_i7_u0_resp_t),
      .mst_req_t(axi_a48_d64_i2_u0_req_t),
      .mst_resp_t(axi_a48_d64_i2_u0_resp_t)
  ) i_narrow_in_iwc_0 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_xbar_quadrant_s1_out_req[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_0]),
      .slv_resp_o(narrow_xbar_quadrant_s1_out_rsp[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_0]),
      .mst_req_o(narrow_in_iwc_0_req),
      .mst_resp_i(narrow_in_iwc_0_rsp)
  );
  axi_a48_d64_i4_u0_req_t   narrow_out_0_req;
  axi_a48_d64_i4_u0_resp_t  narrow_out_0_rsp;

  axi_a48_d512_i2_u0_req_t  wide_in_iwc_0_req;
  axi_a48_d512_i2_u0_resp_t wide_in_iwc_0_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(5),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d512_i5_u0_req_t),
      .slv_resp_t(axi_a48_d512_i5_u0_resp_t),
      .mst_req_t(axi_a48_d512_i2_u0_req_t),
      .mst_resp_t(axi_a48_d512_i2_u0_resp_t)
  ) i_wide_in_iwc_0 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_xbar_quadrant_s1_out_req[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_0]),
      .slv_resp_o(wide_xbar_quadrant_s1_out_rsp[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_0]),
      .mst_req_o(wide_in_iwc_0_req),
      .mst_resp_i(wide_in_iwc_0_rsp)
  );
  axi_a48_d512_i2_u0_req_t wide_out_0_req;
  axi_a48_d512_i2_u0_resp_t wide_out_0_rsp;



  logic [9:0] hart_base_id_0;
  assign hart_base_id_0 = HartIdOffset + tile_id_i * NrCoresS1Quadrant + 0 * NrCoresCluster;

  occamy_cluster_wrapper i_occamy_cluster_0 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .debug_req_i(debug_req_i[0*NrCoresCluster+:NrCoresCluster]),
      .meip_i(meip_i[0*NrCoresCluster+:NrCoresCluster]),
      .mtip_i(mtip_i[0*NrCoresCluster+:NrCoresCluster]),
      .msip_i(msip_i[0*NrCoresCluster+:NrCoresCluster]),
      .hart_base_id_i(hart_base_id_0),
      .cluster_base_addr_i(cluster_base_addr[0]),
      .clk_d2_bypass_i(1'b0),
      .narrow_in_req_i(narrow_in_iwc_0_req),
      .narrow_in_resp_o(narrow_in_iwc_0_rsp),
      .narrow_out_req_o(narrow_out_0_req),
      .narrow_out_resp_i(narrow_out_0_rsp),
      .wide_out_req_o(wide_out_0_req),
      .wide_out_resp_i(wide_out_0_rsp),
      .wide_in_req_i(wide_in_iwc_0_req),
      .wide_in_resp_o(wide_in_iwc_0_rsp),
      .sram_cfgs_i(sram_cfg_i.cluster)
  );

  ///////////////
  // Cluster 1 //
  ///////////////
  axi_a48_d64_i2_u0_req_t  narrow_in_iwc_1_req;
  axi_a48_d64_i2_u0_resp_t narrow_in_iwc_1_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(7),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d64_i7_u0_req_t),
      .slv_resp_t(axi_a48_d64_i7_u0_resp_t),
      .mst_req_t(axi_a48_d64_i2_u0_req_t),
      .mst_resp_t(axi_a48_d64_i2_u0_resp_t)
  ) i_narrow_in_iwc_1 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_xbar_quadrant_s1_out_req[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_1]),
      .slv_resp_o(narrow_xbar_quadrant_s1_out_rsp[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_1]),
      .mst_req_o(narrow_in_iwc_1_req),
      .mst_resp_i(narrow_in_iwc_1_rsp)
  );
  axi_a48_d64_i4_u0_req_t   narrow_out_1_req;
  axi_a48_d64_i4_u0_resp_t  narrow_out_1_rsp;

  axi_a48_d512_i2_u0_req_t  wide_in_iwc_1_req;
  axi_a48_d512_i2_u0_resp_t wide_in_iwc_1_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(5),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d512_i5_u0_req_t),
      .slv_resp_t(axi_a48_d512_i5_u0_resp_t),
      .mst_req_t(axi_a48_d512_i2_u0_req_t),
      .mst_resp_t(axi_a48_d512_i2_u0_resp_t)
  ) i_wide_in_iwc_1 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_xbar_quadrant_s1_out_req[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_1]),
      .slv_resp_o(wide_xbar_quadrant_s1_out_rsp[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_1]),
      .mst_req_o(wide_in_iwc_1_req),
      .mst_resp_i(wide_in_iwc_1_rsp)
  );
  axi_a48_d512_i2_u0_req_t wide_out_1_req;
  axi_a48_d512_i2_u0_resp_t wide_out_1_rsp;



  logic [9:0] hart_base_id_1;
  assign hart_base_id_1 = HartIdOffset + tile_id_i * NrCoresS1Quadrant + 1 * NrCoresCluster;

  occamy_cluster_wrapper i_occamy_cluster_1 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .debug_req_i(debug_req_i[1*NrCoresCluster+:NrCoresCluster]),
      .meip_i(meip_i[1*NrCoresCluster+:NrCoresCluster]),
      .mtip_i(mtip_i[1*NrCoresCluster+:NrCoresCluster]),
      .msip_i(msip_i[1*NrCoresCluster+:NrCoresCluster]),
      .hart_base_id_i(hart_base_id_1),
      .cluster_base_addr_i(cluster_base_addr[1]),
      .clk_d2_bypass_i(1'b0),
      .narrow_in_req_i(narrow_in_iwc_1_req),
      .narrow_in_resp_o(narrow_in_iwc_1_rsp),
      .narrow_out_req_o(narrow_out_1_req),
      .narrow_out_resp_i(narrow_out_1_rsp),
      .wide_out_req_o(wide_out_1_req),
      .wide_out_resp_i(wide_out_1_rsp),
      .wide_in_req_i(wide_in_iwc_1_req),
      .wide_in_resp_o(wide_in_iwc_1_rsp),
      .sram_cfgs_i(sram_cfg_i.cluster)
  );

  ///////////////
  // Cluster 2 //
  ///////////////
  axi_a48_d64_i2_u0_req_t  narrow_in_iwc_2_req;
  axi_a48_d64_i2_u0_resp_t narrow_in_iwc_2_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(7),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d64_i7_u0_req_t),
      .slv_resp_t(axi_a48_d64_i7_u0_resp_t),
      .mst_req_t(axi_a48_d64_i2_u0_req_t),
      .mst_resp_t(axi_a48_d64_i2_u0_resp_t)
  ) i_narrow_in_iwc_2 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_xbar_quadrant_s1_out_req[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_2]),
      .slv_resp_o(narrow_xbar_quadrant_s1_out_rsp[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_2]),
      .mst_req_o(narrow_in_iwc_2_req),
      .mst_resp_i(narrow_in_iwc_2_rsp)
  );
  axi_a48_d64_i4_u0_req_t   narrow_out_2_req;
  axi_a48_d64_i4_u0_resp_t  narrow_out_2_rsp;

  axi_a48_d512_i2_u0_req_t  wide_in_iwc_2_req;
  axi_a48_d512_i2_u0_resp_t wide_in_iwc_2_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(5),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d512_i5_u0_req_t),
      .slv_resp_t(axi_a48_d512_i5_u0_resp_t),
      .mst_req_t(axi_a48_d512_i2_u0_req_t),
      .mst_resp_t(axi_a48_d512_i2_u0_resp_t)
  ) i_wide_in_iwc_2 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_xbar_quadrant_s1_out_req[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_2]),
      .slv_resp_o(wide_xbar_quadrant_s1_out_rsp[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_2]),
      .mst_req_o(wide_in_iwc_2_req),
      .mst_resp_i(wide_in_iwc_2_rsp)
  );
  axi_a48_d512_i2_u0_req_t wide_out_2_req;
  axi_a48_d512_i2_u0_resp_t wide_out_2_rsp;



  logic [9:0] hart_base_id_2;
  assign hart_base_id_2 = HartIdOffset + tile_id_i * NrCoresS1Quadrant + 2 * NrCoresCluster;

  occamy_cluster_wrapper i_occamy_cluster_2 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .debug_req_i(debug_req_i[2*NrCoresCluster+:NrCoresCluster]),
      .meip_i(meip_i[2*NrCoresCluster+:NrCoresCluster]),
      .mtip_i(mtip_i[2*NrCoresCluster+:NrCoresCluster]),
      .msip_i(msip_i[2*NrCoresCluster+:NrCoresCluster]),
      .hart_base_id_i(hart_base_id_2),
      .cluster_base_addr_i(cluster_base_addr[2]),
      .clk_d2_bypass_i(1'b0),
      .narrow_in_req_i(narrow_in_iwc_2_req),
      .narrow_in_resp_o(narrow_in_iwc_2_rsp),
      .narrow_out_req_o(narrow_out_2_req),
      .narrow_out_resp_i(narrow_out_2_rsp),
      .wide_out_req_o(wide_out_2_req),
      .wide_out_resp_i(wide_out_2_rsp),
      .wide_in_req_i(wide_in_iwc_2_req),
      .wide_in_resp_o(wide_in_iwc_2_rsp),
      .sram_cfgs_i(sram_cfg_i.cluster)
  );

  ///////////////
  // Cluster 3 //
  ///////////////
  axi_a48_d64_i2_u0_req_t  narrow_in_iwc_3_req;
  axi_a48_d64_i2_u0_resp_t narrow_in_iwc_3_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(7),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d64_i7_u0_req_t),
      .slv_resp_t(axi_a48_d64_i7_u0_resp_t),
      .mst_req_t(axi_a48_d64_i2_u0_req_t),
      .mst_resp_t(axi_a48_d64_i2_u0_resp_t)
  ) i_narrow_in_iwc_3 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_xbar_quadrant_s1_out_req[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_3]),
      .slv_resp_o(narrow_xbar_quadrant_s1_out_rsp[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_3]),
      .mst_req_o(narrow_in_iwc_3_req),
      .mst_resp_i(narrow_in_iwc_3_rsp)
  );
  axi_a48_d64_i4_u0_req_t   narrow_out_3_req;
  axi_a48_d64_i4_u0_resp_t  narrow_out_3_rsp;

  axi_a48_d512_i2_u0_req_t  wide_in_iwc_3_req;
  axi_a48_d512_i2_u0_resp_t wide_in_iwc_3_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(5),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d512_i5_u0_req_t),
      .slv_resp_t(axi_a48_d512_i5_u0_resp_t),
      .mst_req_t(axi_a48_d512_i2_u0_req_t),
      .mst_resp_t(axi_a48_d512_i2_u0_resp_t)
  ) i_wide_in_iwc_3 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_xbar_quadrant_s1_out_req[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_3]),
      .slv_resp_o(wide_xbar_quadrant_s1_out_rsp[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_3]),
      .mst_req_o(wide_in_iwc_3_req),
      .mst_resp_i(wide_in_iwc_3_rsp)
  );
  axi_a48_d512_i2_u0_req_t wide_out_3_req;
  axi_a48_d512_i2_u0_resp_t wide_out_3_rsp;



  logic [9:0] hart_base_id_3;
  assign hart_base_id_3 = HartIdOffset + tile_id_i * NrCoresS1Quadrant + 3 * NrCoresCluster;

  occamy_cluster_wrapper i_occamy_cluster_3 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .debug_req_i(debug_req_i[3*NrCoresCluster+:NrCoresCluster]),
      .meip_i(meip_i[3*NrCoresCluster+:NrCoresCluster]),
      .mtip_i(mtip_i[3*NrCoresCluster+:NrCoresCluster]),
      .msip_i(msip_i[3*NrCoresCluster+:NrCoresCluster]),
      .hart_base_id_i(hart_base_id_3),
      .cluster_base_addr_i(cluster_base_addr[3]),
      .clk_d2_bypass_i(1'b0),
      .narrow_in_req_i(narrow_in_iwc_3_req),
      .narrow_in_resp_o(narrow_in_iwc_3_rsp),
      .narrow_out_req_o(narrow_out_3_req),
      .narrow_out_resp_i(narrow_out_3_rsp),
      .wide_out_req_o(wide_out_3_req),
      .wide_out_resp_i(wide_out_3_rsp),
      .wide_in_req_i(wide_in_iwc_3_req),
      .wide_in_resp_o(wide_in_iwc_3_rsp),
      .sram_cfgs_i(sram_cfg_i.cluster)
  );

endmodule
