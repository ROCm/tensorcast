# !/bin/bash
# tcast/config_attrs.sh: example bash script to set lp_config via env vars
# SPDX-License-Identifier: MIT
# TensorCast: Specification, conversion and compression of arbitrary datatypes.

export LP_BLOCK_SIZE="32, 32"
export LP_BLOCK_AXES="0, 1"
export LP_SCALE_DTYPE="none"
export LP_Q_DTYPE="float8_e4m3fnuz"
export LP_K_DTYPE="float8_e4m3fnuz"
export LP_V_DTYPE="float8_e4m3fnuz"
export LP_P_DTYPE="float8_e5m2fnuz"
export LP_DS_DTYPE="float8_e5m2fnuz"
export LP_DO_DTYPE="float8_e5m2fnuz"
export LP_ICP_QK="true"
export LP_ICP_PV="false"
export LP_ICP_FP32="true"
export LP_ROUNDMODE="nearest"
export LP_SCALEMODE="floor"
export LP_CASTMODE="virtual"
