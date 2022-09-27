#OpenFile vsim.wlf

onerror {resume}

# add the waves of cluster 0
view -new wave -title {Occamy MNIST}
add wave -position insertpoint {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_0/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/*}
# add the waves of cluster 1
# add wave -position insertpoint {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_1/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/*}
# add a new divider called DEBUG to the same window
add wave -height 200 -divider "DEBUG"
# add debug signal groups
add wave -group "STALL_CL0" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_0/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/stall}
add wave -group "STALL_CL0" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_0/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/lsu_stall}
add wave -group "STALL_CL0" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_0/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/acc_stall}
add wave -group "STALL_CL0" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_0/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/valid_instr}

add wave -group "PC_CL0" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_0/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/next_pc}

# add wave -group "STALL_CL1" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_1/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/stall}
# add wave -group "STALL_CL1" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_1/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/lsu_stall}
# add wave -group "STALL_CL1" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_1/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/acc_stall}
# add wave -group "STALL_CL1" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_1/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/valid_instr}

# add wave -group "PC_CL1" {tb_bin/i_dut/i_occamy/i_occamy_soc/i_occamy_quadrant_s1_0/i_occamy_cluster_1/i_cluster/gen_core[0]/i_snitch_cc/i_snitch/next_pc}

# toggle signal names to leaves only 
config wave -signalnamewidth 1