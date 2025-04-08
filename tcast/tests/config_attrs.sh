# !/bin/bash
# tcast/config_attrs.sh: example bash script to set lp_config via env vars
# SPDX-License-Identifier: MIT
# TensorCast: Specification, conversion and compression of arbitrary datatypes.

export TC_BLOCK_SIZE="32, 32"
export TC_SCALE_DTYPE="none"
export TC_Q_DTYPE="float8_e4m3fnuz"
export TC_K_DTYPE="float8_e4m3fnuz"
export TC_V_DTYPE="float8_e4m3fnuz"
export TC_P_DTYPE="float8_e5m2fnuz"
export TC_DS_DTYPE="float8_e5m2fnuz"
export TC_DO_DTYPE="float8_e5m2fnuz"
export TC_ICP_QK="true"
export TC_ICP_PV="false"
export TC_ICP_FP32="true"
export TC_ROUNDMODE="even"
export TC_SCALEMODE="floor"
export TC_CASTMODE="virtual"
