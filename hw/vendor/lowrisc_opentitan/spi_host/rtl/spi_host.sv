// Copyright lowRISC contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Serial Peripheral Interface (SPI) Host module.
//
//

`include "common_cells/assertions.svh"

module spi_host
  import spi_host_reg_pkg::*;
#(
  parameter type reg_req_t = logic,
  parameter type reg_rsp_t = logic
) (
  input              clk_i,
  input              rst_ni,
  input              clk_core_i,
  input              rst_core_ni,

  // Register interface
  input  reg_req_t reg_req_i,
  output reg_rsp_t reg_rsp_o,

  // SPI Interface
  output logic             cio_sck_o,
  output logic             cio_sck_en_o,
  output logic [MaxCS-1:0] cio_csb_o,
  output logic [MaxCS-1:0] cio_csb_en_o,
  output logic [3:0]       cio_sd_o,
  output logic [3:0]       cio_sd_en_o,
  input        [3:0]       cio_sd_i,

  output logic             intr_error_o,
  output logic             intr_spi_event_o
);

  wire event_spi_event = 1'b0;
  wire event_error = 1'b0;

  spi_host_reg2hw_t reg2hw;
  spi_host_hw2reg_t hw2reg;

  // Register module
  spi_host_reg_top #(
    .reg_req_t (reg_req_t),
    .reg_rsp_t (reg_rsp_t)
  ) u_reg (
    .clk_i,
    .rst_ni,

    .reg_req_i,
    .reg_rsp_o,

    .reg_req_win_o (),
    .reg_rsp_win_i ('0),

    .reg2hw,
    .hw2reg,

    .devmode_i  (1'b1)
  );

  // Some dummy connections
  assign cio_sck_o    = '0;
  assign cio_sck_en_o = '1;
  assign cio_csb_o    = {MaxCS{'0}};
  assign cio_csb_en_o = {MaxCS{'1}};
  assign cio_sd_o     = 4'h0;
  assign cio_sd_en_o  = 4'h0;

  assign hw2reg.status.txqd.d = 9'h0;
  assign hw2reg.status.txqd.de = 1'b0;
  assign hw2reg.status.rxqd.d = 9'h0;
  assign hw2reg.status.rxqd.de = 1'b0;
  assign hw2reg.status.rxwm.d = 1'b0;
  assign hw2reg.status.rxwm.de = 1'b0;
  assign hw2reg.status.byteorder.d = 1'b0;
  assign hw2reg.status.byteorder.de = 1'b0;
  assign hw2reg.status.rxstall.d = 1'b0;
  assign hw2reg.status.rxstall.de = 1'b0;
  assign hw2reg.status.rxempty.d = 1'b0;
  assign hw2reg.status.rxempty.de = 1'b0;
  assign hw2reg.status.rxfull.d = 1'b0;
  assign hw2reg.status.rxfull.de = 1'b0;
  assign hw2reg.status.txwm.d = 1'b0;
  assign hw2reg.status.txwm.de = 1'b0;
  assign hw2reg.status.txstall.d = 1'b0;
  assign hw2reg.status.txstall.de = 1'b0;
  assign hw2reg.status.txempty.d = 1'b0;
  assign hw2reg.status.txempty.de = 1'b0;
  assign hw2reg.status.txfull.d = 1'b0;
  assign hw2reg.status.txfull.de = 1'b0;
  assign hw2reg.status.active.d = 1'b0;
  assign hw2reg.status.active.de = 1'b0;
  assign hw2reg.status.ready.d = 1'b0;
  assign hw2reg.status.ready.de = 1'b0;
  for(genvar ii = 0; ii < MaxCS; ii++) begin : gen_go_bit_tie_offs
    assign hw2reg.command[ii].go.d = 1'b0;
    assign hw2reg.command[ii].go.de = 1'b0;
  end
  assign hw2reg.rxdata.d = 32'h0;
  assign hw2reg.error_status.cmderr.d = 1'b0;
  assign hw2reg.error_status.cmderr.de = 1'b0;
  assign hw2reg.error_status.overflow.d = 1'b0;
  assign hw2reg.error_status.overflow.de = 1'b0;
  assign hw2reg.error_status.underflow.d = 1'b0;
  assign hw2reg.error_status.underflow.de = 1'b0;

  prim_intr_hw #(.Width(1)) intr_hw_spi_event (
    .clk_i,
    .rst_ni,
    .event_intr_i           (event_spi_event),
    .reg2hw_intr_enable_q_i (reg2hw.intr_enable.spi_event.q),
    .reg2hw_intr_test_q_i   (reg2hw.intr_test.spi_event.q),
    .reg2hw_intr_test_qe_i  (reg2hw.intr_test.spi_event.qe),
    .reg2hw_intr_state_q_i  (reg2hw.intr_state.spi_event.q),
    .hw2reg_intr_state_de_o (hw2reg.intr_state.spi_event.de),
    .hw2reg_intr_state_d_o  (hw2reg.intr_state.spi_event.d),
    .intr_o                 (intr_spi_event_o)
  );

  prim_intr_hw #(.Width(1)) intr_hw_error (
    .clk_i,
    .rst_ni,
    .event_intr_i           (event_error),
    .reg2hw_intr_enable_q_i (reg2hw.intr_enable.error.q),
    .reg2hw_intr_test_q_i   (reg2hw.intr_test.error.q),
    .reg2hw_intr_test_qe_i  (reg2hw.intr_test.error.qe),
    .reg2hw_intr_state_q_i  (reg2hw.intr_state.error.q),
    .hw2reg_intr_state_de_o (hw2reg.intr_state.error.de),
    .hw2reg_intr_state_d_o  (hw2reg.intr_state.error.d),
    .intr_o                 (intr_error_o)
  );

  `ASSERT_KNOWN(CioSckKnownO_A, cio_sck_o)
  `ASSERT_KNOWN(CioSckEnKnownO_A, cio_sck_en_o)
  `ASSERT_KNOWN(CioCsbKnownO_A, cio_csb_o)
  `ASSERT_KNOWN(CioCsbEnKnownO_A, cio_csb_en_o)
  `ASSERT_KNOWN(CioSdKnownO_A, cio_sd_o)
  `ASSERT_KNOWN(CioSdEnKnownO_A, cio_sd_en_o)
  `ASSERT_KNOWN(IntrSpiEventKnownO_A, intr_spi_event_o)
  `ASSERT_KNOWN(IntrErrorKnownO_A, intr_error_o)

endmodule : spi_host
