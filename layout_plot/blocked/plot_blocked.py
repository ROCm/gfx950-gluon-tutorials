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

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BlockedConfig:
    sizePerThread: tuple
    threadsPerWarp: tuple
    warpsPerCTA: tuple
    order: tuple
    waveSize: int = 64


def draw_blocked_layout_cmd(dim0, dim1, dim0Name, dim1Name, blockedConfig):
    return f"""\\begin{{document}}
               \\begin{{tikzpicture}}
               \\def\\scale{{1}}
               \\def\\elem{{0.06}}
               \\def\\waveSize{{{blockedConfig.waveSize}}}
               \\coordinate (TL) at (0,0);
               \\def\\dimColName{{{dim0Name}}}
               \\def\\dimRowName{{{dim1Name}}}
               \\drawBlockedTensor{{{dim0}}}{{{dim1}}}{{{blockedConfig.sizePerThread[0]}}}{{{blockedConfig.sizePerThread[1]}}}{{{blockedConfig.threadsPerWarp[0]}}}{{{blockedConfig.warpsPerCTA[0]}}}{{{blockedConfig.warpsPerCTA[1]}}}{{{blockedConfig.order[0]}}}
               \\end{{tikzpicture}}
               \\end{{document}}"""


def generate_blocked_tex(args):
    """Generate the tex file of blocked layout and draw it out"""
    assert (
        args.plot_type == "blocked"
    ), f"parsing the wrong arguments. Want blocked but have {args.plot_type}"
    # preprocess the args
    # shortcut to plot dot operand B to save some cmd args
    if args.matrixB:
        dim0Name, dim1Name = "K", "N"
    else:
        dim0Name, dim1Name = args.rowName, args.colName
    # TODO: this can be further refactored to absorb the assertions below to make it more elegant
    sizePerThread = args.sizePerThread
    threadsPerWarp = args.threadsPerWarp
    warpsPerCTA = args.warpsPerCTA
    order = args.order
    waveSize = args.waveSize
    blockedConfig = BlockedConfig(sizePerThread, threadsPerWarp, warpsPerCTA, order, waveSize)
    CTAShape = [
        sizePerThread[0] * threadsPerWarp[0] * warpsPerCTA[0],
        sizePerThread[1] * threadsPerWarp[1] * warpsPerCTA[1],
    ]

    # checks and logging
    if args.blockShape is not None:
        dim0, dim1 = args.blockShape
    else:
        print(
            f"Since block size is not explicitly defined, it assumes block size = CTAShape = {CTAShape}"
        )
        dim0, dim1 = CTAShape
    print(
        f"Plotting a block [{dim0Name}, {dim1Name}] = [{dim0}, {dim1}] with the following blocked layout:"
    )
    print(f"{sizePerThread=}", end=", ")
    print(f"{threadsPerWarp=}", end=", ")
    print(f"{warpsPerCTA=}", end=", ")
    print(f"{order=}", end=", ")
    print(f"CTAShape={CTAShape}")
    assert (
        dim0 != 0 and CTAShape[0] <= dim0 and dim0 % CTAShape[0] == 0
    ), "CTAShape[0] should be smaller than dim of {dim0Name}={dim0} and fully spans it"
    assert (
        dim1 != 0 and CTAShape[1] <= dim1 and dim1 % CTAShape[1] == 0
    ), "CTAShape[1] should be smaller than dim of {dim1Name}={dim1} and fully spans it"

    # write the tex file
    curr_dir = Path(__file__).resolve().parent
    with open("myplot.tex", "w") as f_plot:
        with open(curr_dir / "../utils/preamble.tex") as file:
            preamble = file.read()

        f_plot.write(preamble)
        draw_blockedLayout_str = draw_blocked_layout_cmd(
            dim0, dim1, dim0Name, dim1Name, blockedConfig
        )
        func_ref = str(curr_dir / "blockedLayout")
        f_plot.write(f"\\input{{ {func_ref} }}\n")
        f_plot.write(draw_blockedLayout_str)
