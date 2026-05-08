#!/usr/bin/env bash
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
#
# install_att_decoder.sh — fetch and install the rocprof-trace-decoder library
# that `rocprofv3 --att` needs at trace-decode time. `run_att.py` (and anything
# else that runs `rocprofv3 --att`) will load this .so via the
# ROCPROF_ATT_LIBRARY_PATH environment variable.
#
# Default install drops the .so into ROCm's standard library directory
# (/opt/rocm/lib), which `run_att.py` already searches by default — no env-var
# tweaking required afterwards. Override with --prefix DIR to install elsewhere
# (in which case you'll need to export ROCPROF_ATT_LIBRARY_PATH=DIR/lib).
#
# Usage:
#   scripts/install_att_decoder.sh                 # install to /opt/rocm/lib
#   scripts/install_att_decoder.sh --prefix /opt/foo
#   VERSION=0.1.6 scripts/install_att_decoder.sh   # pin a different release
#
# Requires curl and tar in PATH and write access to the install prefix.
##############################################################################

set -euo pipefail

VERSION="${VERSION:-0.1.6}"
PREFIX="/opt/rocm"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --prefix=*)
            PREFIX="${1#--prefix=}"
            shift
            ;;
        -h|--help)
            sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

ASSET="rocprof-trace-decoder-manylinux-2.28-${VERSION}-Linux.tar.gz"
URL="https://github.com/ROCm/rocprof-trace-decoder/releases/download/${VERSION}/${ASSET}"
LIB_DIR="${PREFIX}/lib"
LIB_NAME="librocprof-trace-decoder.so"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Downloading ${ASSET} from ${URL}"
curl -fL --retry 3 --output "${WORK}/${ASSET}" "$URL"

echo "Extracting"
tar -xzf "${WORK}/${ASSET}" -C "$WORK"

# The archive lays out the file at:
#   <archive-name>/opt/rocm/lib/librocprof-trace-decoder.so
SRC=$(find "$WORK" -name "$LIB_NAME" -type f | head -1)
if [[ -z "$SRC" ]]; then
    echo "error: ${LIB_NAME} not found in archive contents" >&2
    exit 1
fi

mkdir -p "$LIB_DIR"
install -m 0644 "$SRC" "${LIB_DIR}/${LIB_NAME}"

echo
echo "Installed: ${LIB_DIR}/${LIB_NAME}"
if [[ "$PREFIX" != "/opt/rocm" ]]; then
    echo
    echo "Non-default prefix used. Export the path before running rocprofv3 --att:"
    echo "  export ROCPROF_ATT_LIBRARY_PATH=${LIB_DIR}/"
fi
