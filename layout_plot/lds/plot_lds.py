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

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LDSConfig:
    banks: int
    ldsLayout: str
    ldsAccess: str
    mnContig: bool
    mfmaTransLD: bool
    swizzleVec: int
    accessVec: int
    kWidth: int
    padInterval: int
    padAmount: int
    waveSize: int = 64

    def __init__(
        self,
        banks,
        ldsLayout,
        ldsAccess,
        mnContig,
        mfmaTransLD,
        swizzleVec,
        accessVec,
        kWidth,
        padInterval,
        padAmount,
        waveSize=64,
    ):
        self.banks = banks
        self.ldsLayout = ldsLayout
        self.ldsAccess = ldsAccess
        self.mnContig = mnContig
        self.mfmaTransLD = mfmaTransLD
        self.swizzleVec = swizzleVec
        self.accessVec = accessVec
        self.kWidth = kWidth
        self.padInterval = padInterval
        self.padAmount = padAmount
        self.waveSize = waveSize

    def print(self):
        print(
            f"{self.banks=} {self.ldsLayout=} {self.ldsAccess=} {self.mnContig=} {self.mfmaTransLD=} {self.swizzleVec=} {self.accessVec=} {self.kWidth=} {self.padInterval} {self.padAmount} {self.waveSize=}"
        )


def _parse_shared_layout(shared_layout: str):
    if not shared_layout:
        return None
    try:
        parsed = literal_eval(f"[{shared_layout}]")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid --sharedLayout format: {shared_layout}") from exc
    if (not isinstance(parsed, list)) or len(parsed) != 2:
        raise ValueError("--sharedLayout must contain two lists: padding and swizzle basis")
    pads, basis = parsed
    if not isinstance(pads, list) or not isinstance(basis, list):
        raise ValueError("--sharedLayout must contain two list values")
    for item in pads:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each padding entry must be [padInterval, padAmount] in elements")
    for item in basis:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each swizzle basis entry must be [row, col]")
    return [[int(p[0]), int(p[1])] for p in pads], [[int(b[0]), int(b[1])] for b in basis]


def _build_shared_layout_lookup(
    dim0, dim1, vec, swizzle_vec, elem_type_in_bytes, banks, pads, basis
):
    coord_to_off = {}
    total = dim0 * dim1
    for off in range(total):
        row = col = 0
        bit = 0
        curr = off
        while curr:
            if curr & 1:
                if bit >= len(basis):
                    raise ValueError("Not enough swizzle basis entries for LDS offsets")
                row += basis[bit][0]
                col += basis[bit][1]
            bit += 1
            curr >>= 1
        if 0 <= row < dim0 and 0 <= col < dim1:
            coord_to_off[(row, col)] = off

    for interval, amount in pads:
        if interval % swizzle_vec != 0 or amount % swizzle_vec != 0:
            raise ValueError(
                f"sharedLayout padInterval/padAmount must be multiples of swizzleVec ({swizzle_vec}); got [{interval}, {amount}]"
            )

    vecs_per_row = dim1 // vec
    max_off_vec = dim0 * vecs_per_row
    lookup_rows = []
    lookup_vecs = []
    lds_row_bytes = int(banks * 4)
    vec_in_bytes = int(vec * elem_type_in_bytes)
    elems_per_lds_row = int(lds_row_bytes / elem_type_in_bytes)

    vecs_per_lds_row = int(lds_row_bytes / vec_in_bytes)

    # row_start_offsets is indexed by compact_row, which is an LDS-row index
    # (see compact_row below), not a tensor-row index: a single wide tensor row
    # can span several LDS rows, so the number of LDS rows is generally larger
    # than dim0. Size the list to the largest compact_row the loop can produce
    # (sizing it to dim0 under-allocated for wide tensors and raised
    # IndexError, e.g. a 64x512 bf16 tile with vec=8 needs 256 LDS rows).
    if max_off_vec > 0:
        num_lds_rows = (
            max(
                (max_off_vec - 1) // vecs_per_lds_row,
                (max_off_vec - 1) // vecs_per_row,
            )
            + 1
        )
    else:
        num_lds_rows = 0
    row_start_offsets = [0 for _ in range(num_lds_rows)]

    for off_vec in range(max_off_vec):
        tensor_row = off_vec // vecs_per_row
        gp = off_vec % vecs_per_row
        col_start = gp * vec
        tensor_off = coord_to_off.get((tensor_row, col_start))
        if tensor_off is None:
            raise ValueError(
                f"Cannot map tensor coord ({tensor_row}, {col_start}) with provided shared layout"
            )
        padded_elem_off = tensor_off + sum(
            (tensor_off // interval) * amount for interval, amount in pads if interval > 0
        )
        padded_byte_off = int(padded_elem_off * elem_type_in_bytes)
        abs_row = padded_byte_off // lds_row_bytes
        compact_row = max(int(off_vec // vecs_per_lds_row), int(off_vec // vecs_per_row))
        if gp == 0:
            row_start_offsets[compact_row] = int(abs_row * lds_row_bytes)
        lookup_rows.append(compact_row)
        # Store the exact element offset within the LDS row; the tex divides by
        # vec to get the (possibly fractional) horizontal vec position. Storing
        # the raw offset is exact for every vec/swizzleVec relationship:
        #   - // swizzle_vec collapsed vecs onto one column when vec < swizzle_vec
        #     (mfma-transpose-load, vec=4 < swizzleVec=16) -- they overlapped.
        #   - // vec truncated sub-vec padding when vec > swizzle_vec (a8w8
        #     kWidth=32 with a 16-element pad lands a vec at byte 16 = pos 0.5).
        lookup_vecs.append(padded_elem_off % elems_per_lds_row)

    return lookup_rows, lookup_vecs, row_start_offsets


def typeToBytes(dtype):
    if dtype == "bf16" or dtype == "fp16":
        return 2
    if dtype == "bf8" or dtype == "fp8" or dtype == "i8":
        return 1
    if dtype == "f4":
        return 0.5
    if dtype == "fp6" or dtype == "bf6":
        return 0.75


def maxKDimInBytes(dtype, mfmaNonKDim, kWidth, waveSize=64):
    groups = waveSize / mfmaNonKDim
    if (dtype == "bf8" or dtype == "fp8") and kWidth == 16:
        groups *= 2
    return groups * kWidth * typeToBytes(dtype)


def calcPerPhase(banks, dtype, K):
    bytesPerBank = 4
    return max(banks * bytesPerBank / (K * typeToBytes(dtype)), 1)


def draw_lds_access_cmd(dim0, dim1, dtype, mfmaNonKDim, ldsConfig, sharedLayout):
    if sharedLayout is not None:
        hasSwizzle = 3
    elif ldsConfig.ldsLayout == "swizzle":
        hasSwizzle = 1
    elif ldsConfig.ldsLayout == "padding":
        hasSwizzle = 2
    else:
        hasSwizzle = 0

    if ldsConfig.ldsAccess == "read":
        accessMode = 1
    elif ldsConfig.ldsAccess == "write":
        accessMode = 2
    else:
        accessMode = 0

    trans = 1 if ldsConfig.mnContig else 0
    useMfmaTransLD = 1 if ldsConfig.mfmaTransLD else 0
    banks = ldsConfig.banks
    padInterval = ldsConfig.padInterval
    padAmount = ldsConfig.padAmount
    waveSize = ldsConfig.waveSize

    if trans:
        dim0Name = "k"
        dim1Name = "n"
    else:
        dim0Name = "m"
        dim1Name = "k"
    dim0Size = dim0
    dim1Size = dim1
    """
    Definitions of different vector size

    swizzleVec: Number of elements that are grouped together when swizzling is enabled.
                Note that this is all about LDS layout without considering LDS read
                or write patterns. And this is un-related to K- or MN-contig settings.
    accessVec:  When reading from or writing to LDS, accessVec is the number of contiguous
                elements each thread read or write as a vector.
                This is un-related to K- or MN-contig settings.
                Note that accessVec <= swizzleVec. accessVec for read and write are not
                required to be the same.
    kWidth:     Number of contiguous elements along the k dim that each thread holds
                right before invoking mfma instruction(s). kWidth can be larger than
                the required number of contiguous elements along the k dim for a single
                mfma instruction.
                Note that kWidth is un-related to swizzleVec or accessVec. kWidth should
                be set according to datatype and mfmaNonKDim.

    We need to handle the following cases of LDS layout and access patterns:

    case 1: K-contig in both HBM and LDS (default)
      In most cases, we can set swizzleVec = accessVec = kWidth according to the dtype.
      However, for mfmaNonKDim = 16, banks = 64, and kWidth = 8B, 32 threads will
      access LDS at the same cycle. In this case, we need to double swizzleVec = 16B.

      Swizzling: works as suggested above.
      Padding:   will have bank conflicts for ds_read_b128 due to non-linear thread ids
                 are accessing LDS at the same cycle

    case 2: MN-contig in both HBM and LDS without using mfma_transpose_ld instructions (-mnContig)
      In this case, ds_read can only read one element at a time (i.e. accessVec is always 1).
      Therefore, we can always choose swizzleVec = 16B. kWidth does not matter. accessVec is always 1.
      Note that in this case, only swizzling is supported and can help resolve bank conflicts.
      But the performance bottleneck is scalar ds_read rather than bank conflicts.

    case 3: MN-contig in both HBM and LDS using mfma_transpose_ld instructions (-mnContig -mfma_trans_load)
      In this case, ds_read is done in a special pattern so that the ds_read_b64_tr_bx instructions
      can be used. Each thread will read 8B data, which corresponds to kWidth = 8B/elemInBytes.
      The swizzleVec needs to be set to mfmaNonKDim.

      Swizzling: currently, it leads to bank conflicts for nonKDim = 16 and
                 if the read pattern follows the spec.
                 For nonKDim = 32, swizzling does not have bank conflicts.
      Padding:   It can help resolve bank conflicts for both nonKDim = 16 and 32.
                 However, it leads to a lot of waste of LDS space.

    case 4: MN-contig in HBM and k-Contig in LDS (-inThreadTrans)
      Not supported yet
    """

    elemTypeInBytes = typeToBytes(dtype)

    bankLabelScale = 0.8
    bsize = 0.15

    if trans == 0:
        # case 1
        swizzleVec = ldsConfig.swizzleVec
        accessVec = ldsConfig.accessVec
        vec = ldsConfig.kWidth
    elif useMfmaTransLD == 0:
        # case 2
        # int() keeps vec/swizzleVec integral: they index ranges and divide LDS
        # offsets in _build_shared_layout_lookup, where a float raises TypeError.
        swizzleVec = int(16 / elemTypeInBytes)
        accessVec = 1
        vec = swizzleVec
    else:
        # case 3
        vec = int(8 / elemTypeInBytes)
        swizzleVec = mfmaNonKDim
        accessVec = ldsConfig.accessVec

    kWidth = ldsConfig.kWidth
    vecInBytes = vec * elemTypeInBytes

    sharedLayoutRows = ""
    sharedLayoutVecs = ""
    sharedLayoutRowStarts = ""
    if sharedLayout is not None:
        pads, basis = sharedLayout
        lookupRows, lookupVecs, rowStartOffsets = _build_shared_layout_lookup(
            dim0, dim1, vec, swizzleVec, elemTypeInBytes, banks, pads, basis
        )
        sharedLayoutRows = "\n".join(
            [
                f"\\expandafter\\def\\csname sharedrow{i}\\endcsname{{{v}}}"
                for i, v in enumerate(lookupRows)
            ]
        )
        sharedLayoutVecs = "\n".join(
            [
                f"\\expandafter\\def\\csname sharedvec{i}\\endcsname{{{v}}}"
                for i, v in enumerate(lookupVecs)
            ]
        )
        sharedLayoutRowStarts = "\n".join(
            [
                f"\\expandafter\\def\\csname sharedrowstart{i}\\endcsname{{{v}}}"
                for i, v in enumerate(rowStartOffsets)
            ]
        )

    return rf"""\begin{{document}}
               \begin{{tikzpicture}}
               \def\scale{{1}}
               \def\M{{{dim0}}}
               \def\K{{{dim1}}}
               \def\mfmaKWidth{{{kWidth}}}
               \def\vec{{{vec}}}
               \def\swizzleVec{{{swizzleVec}}}
               \def\accessVec{{{accessVec}}}
               \def\vecInBytes{{{vecInBytes}}}
               \def\bytesPerElem{{{elemTypeInBytes}}}
               \def\hasSwizzle{{{hasSwizzle}}}
               \def\accessMode{{{accessMode}}}
               \def\mfmaNonKDim{{{mfmaNonKDim}}}
               \def\dtype{{{dtype}}}
               \def\trans{{{trans}}}
               \def\useMfmaTransLD{{{useMfmaTransLD}}}
               \def\padInterval{{{padInterval}}}
               \def\padAmount{{{padAmount}}}
               \def\waveSize{{{waveSize}}}
               {sharedLayoutRows}
               {sharedLayoutVecs}
               {sharedLayoutRowStarts}

               \def\elemH{{0.18}}
               \def\elem{{0.18}}
               \def\bsize{{{bsize}}}
               \def\bankLabelScale{{{bankLabelScale}}}
               \coordinate (tile TL) at (0,0);
               \coordinate (TL) at (tile TL);
               \drawTensorLayoutGlobalMem{{{dim0Name}}}{{{dim1Name}}}{{{dim0Size}}}{{{dim1Size}}}
               \coordinate (TL) at ($(TL)+(0, -\drawRow-8*\elemH)$);
               \drawLDSLayoutAndAccess{{\hasSwizzle}}{{\accessMode}}{{{banks}}}{{{dim0Name}}}{{{dim1Name}}}{{{dim1Size}}}
               \end{{tikzpicture}}
               \end{{document}}"""


def generate_lds_tex(args):
    assert (
        args.plot_type == "lds"
    ), f"parsing the wrong arguments. Want lds but have {args.plot_type}"
    # preprocess the args
    tShape = args.tensorShape
    dim0 = tShape[0]
    dim1 = tShape[1]
    accessVec = kWidth = args.kWidth
    dtype = args.dtype
    mfmaNonKDim = args.nonKDim
    ldsLayout = args.layout
    ldsAccess = args.access
    banks = args.banks
    mnContig = args.mnContig
    mfmaTransLD = args.mfma_trans_load
    swizzleVec = args.swizzleVec
    padInterval = args.padInterval
    padAmount = args.padAmount
    waveSize = args.waveSize
    sharedLayout = _parse_shared_layout(args.sharedLayout)

    ldsConfig = LDSConfig(
        banks,
        ldsLayout,
        ldsAccess,
        mnContig,
        mfmaTransLD,
        swizzleVec,
        accessVec,
        kWidth,
        padInterval,
        padAmount,
        waveSize,
    )

    # checks and logging
    print(f"Plotting LDS access for tensor {dim0}x{dim1} with vec={kWidth}")
    # write the tex file
    curr_dir = Path(__file__).resolve().parent
    with open("myplot.tex", "w") as f_plot:
        with open(curr_dir / "../utils/preamble.tex") as file:
            preamble = file.read()

        f_plot.write(preamble)
        draw_lds_str = draw_lds_access_cmd(dim0, dim1, dtype, mfmaNonKDim, ldsConfig, sharedLayout)
        func_ref = str(curr_dir / "ldsLayout")
        f_plot.write(f"\\input{{ {func_ref} }}\n")
        f_plot.write(draw_lds_str)
