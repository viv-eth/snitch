#!/usr/bin/env python3
# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Alessandro Ottaviano <aottaviano@iis.ee.ethz.ch>

from string import Template
import argparse
import os.path


parser = argparse.ArgumentParser(description='Convert binary file to verilog rom')
parser.add_argument(
    '--vmem',
    '-m',
    help='Filename of input vmem for $readmemh calls. To be used with flatten==0')
parser.add_argument(
    '--out',
    '-o',
    help='Filename of output System Verilog bootrom')

args = parser.parse_args()

license = """\
//-----------------------------------------------------------------------------
// Title         : Bootrom for Occamy
//-----------------------------------------------------------------------------
// File          : occamy_bootrom.sv
//-----------------------------------------------------------------------------
// Description :
// Auto-generated bootrom from gen_rom_sv.py
//-----------------------------------------------------------------------------
// Copyright (C) 2013-2019 ETH Zurich, University of Bologna
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//-----------------------------------------------------------------------------

// Auto-generated code
"""

module = """\
module $module_name
  #(
    parameter  Depth = 2048,
    localparam Aw    = $$clog2(Depth),
    parameter  Dw    = 32
    )
  (
   input logic                   clk_i,
   input logic                   req_i,
   input logic  [Aw-1:0]         addr_i,
   output logic [Dw-1:0]         rdata_o
  );

    (* rom_style = "block" *) logic [Dw-1:0] mem [Depth];

    localparam MEM_FILE = "$content";
    initial
    begin
       $$display("Initializing ROM from %s", MEM_FILE);
       $$readmemh(MEM_FILE, mem);
    end

    always_ff @(posedge clk_i)
    begin
      if (req_i) begin
        rdata_o <= mem[addr_i];
      end
    end

endmodule
"""


""" Generate SystemVerilog bootcode for FPGA
"""

with open(args.out + ".sv", "w") as f:
    vmem_path = os.path.join("../../../../../../../bootrom/", args.vmem)
    f.write(license)
    s = Template(module)
    f.write(s.substitute(content=vmem_path, module_name=args.out))
