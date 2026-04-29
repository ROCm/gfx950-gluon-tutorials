#! /bin/bash
##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################


## $1: input script that contains one kernel

rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"/

export MLIR_ENABLE_DUMP=1
export AMDGCN_ENABLE_DUMP=1
## Assume CDNA arch
SIMD=4
LDS_SIZE=163840
TOTAL_VGPR=512

get_occ_per_CU() {
    ## $1: vgpr count
    vgpr=$1
    occPerEU=$((TOTAL_VGPR/vgpr))
    if [[ $vgpr -gt 256 ]]; then
        occPerEU=1
    elif [[ $vgpr -gt 168 ]]; then
        occPerEU=2
    elif [[ $vgpr -gt 128 ]]; then
        occPerEU=3
    elif [[ $vgpr -gt 96 ]]; then
        occPerEU=4
    elif [[ $vgpr -gt 80 ]]; then
        occPerEU=5
    elif [[ $vgpr -gt 72 ]]; then
        occPerEU=6
    elif [[ $vgpr -gt 64 ]]; then
        occPerEU=7
    else
        occPerEU=8
    fi

    occPerCU=$((occPerEU*SIMD/num_warps))
    echo $occPerCU
}

$1 > output.mlir 2>&1

LDS_line=$(sed -n '/ttg\.shared\ /p' output.mlir | tail -n 1 | grep -o 'ttg.shared = [0-9]*')
numWarps_line=$(sed -n '/ttg\.num-warps/p' output.mlir | tail -n 1 | grep -o 'ttg.num-warps. = [0-9]*')

LDS=${LDS_line##*=}
num_warps=${numWarps_line##*=}
echo "LDS: $LDS, num_warps: $num_warps"

VGPRs=$(sed -n '/vgpr_count/p' output.mlir | tail -n 1 | awk '{print $2}')
SPILLs=$(sed -n '/vgpr_spill/p' output.mlir | tail -n 1 | awk '{print $2}')

echo "VGPRS: $VGPRs (spill: $SPILLs)"

occLDSPerCU=$((LDS_SIZE/LDS))
occVgprPerCU=$(get_occ_per_CU $VGPRs)
occPerCU=$occVgprPerCU
if [ $occLDSPerCU -lt $occVgprPerCU ];then
    occPerCU=$occLDSPerCU
fi
occPerEU=$((occPerCU*num_warps/SIMD))
echo "occupancy: $occPerEU waves/SIMD or $occPerCU workgroups/CU (occLDSPerCU: $occLDSPerCU, occVgprPerCU: $occVgprPerCU)"

perf=$(tail -n 2 output.mlir)
echo "$perf"

## remove distracting info from the assembly
sed -i '/local_/! {/\.loc/d}' output.mlir
sed -i '/\.Ltmp.*:/d' output.mlir
sed -i '/AMD clang version/d' output.mlir

sed -n '/AMDGCN/, $p' output.mlir > output.amdgcn
