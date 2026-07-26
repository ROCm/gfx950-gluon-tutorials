#!/usr/bin/env python3
##############################################################################
# MIT License
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
##############################################################################
"""LDS bank-conflict model for gfx942, and the sweep used to pick the layouts.

`gl.bank_conflicts()` asserts on AMD shared layouts in this Triton build, so
`analyze()` reconstructs the same information from `gl.to_linear_layout()`:

  * shared layout  -> offset_bases : LDS element offset bit -> (row, col)
  * distributed    -> reg/lane/warp bases

CDNA3 LDS is 64 banks x 4 B = 256 B serviced per cycle, and a `ds_*_b<W>` groups
256/W_bytes consecutive lanes into one phase; within a phase every bank should
be touched at most once.  `analyze()` reports the worst-case serialization
factor (1 == conflict free).

Running the file sweeps the swizzle parameters against both the global-load
(`local_store`) and dot-operand (`local_load`) sides.  It is what picked
`SwizzledSharedLayout(8, 2, 8)` and the consecutive-row global-load layout:
the gfx950 tutorial's layout puts lanes 0-7 on row M0 and lanes 8-15 on row
M0+16, which with a 128 B row stride collide on banks no matter the swizzle
(the swizzle only permutes chunks *within* a row), giving a fixed 2x conflict.

    python layout_check.py
"""

import contextlib
import io
import itertools
import re

from triton._filecheck import run_parser
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

TARGET = GPUTarget("hip", "gfx942", 64)
BANKS = 64
BANK_BYTES = 4
BYTES_PER_CYCLE = BANKS * BANK_BYTES  # 256


@gluon.jit
def _probe(layout: gl.constexpr, shape: gl.constexpr):
    computed: gl.constexpr = gl.to_linear_layout(layout, shape)
    gl.static_print(computed)


def linear_layout(layout, shape, num_warps):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_parser(
            _probe, args=(layout, tuple(shape)), kwargs={"num_warps": num_warps}, target=TARGET
        )
    text = buf.getvalue()
    out = {}
    for key in ("offset_bases", "reg_bases", "lane_bases", "warp_bases", "block_bases"):
        m = re.search(rf"{key}=(\[.*?\]\])", text)
        out[key] = eval(m.group(1)) if m else []
    return out


def _apply(bases, idx):
    """XOR-combine the bases selected by the bits of idx."""
    acc = None
    for bit, base in enumerate(bases):
        if idx >> bit & 1:
            acc = list(base) if acc is None else [a ^ b for a, b in zip(acc, base)]
    return acc if acc is not None else [0] * (len(bases[0]) if bases else 2)


def shared_addr_map(shared_ll, shape):
    """coord (row, col) -> LDS element offset."""
    bases = shared_ll["offset_bases"]
    table = {}
    for off in range(1 << len(bases)):
        coord = tuple(_apply(bases, off))
        table[coord] = off
    assert len(table) == 1 << len(bases), "shared layout is not a bijection"
    return table


def analyze(distr_layout, shared_layout, shape, dtype_bytes, num_warps, label=""):
    d = linear_layout(distr_layout, shape, num_warps)
    s = linear_layout(shared_layout, shape, num_warps)
    addr_of = shared_addr_map(s, shape)

    reg_b, lane_b, warp_b = d["reg_bases"], d["lane_bases"], d["warp_bases"]

    # Vectorization: leading register bases that walk contiguous LDS offsets.
    vec = 1
    for bit, base in enumerate(reg_b):
        want = 1 << bit
        if addr_of.get(tuple(base)) == want:
            vec = 1 << (bit + 1)
        else:
            break
    vec_bytes = vec * dtype_bytes
    lanes_per_phase = min(64, max(1, BYTES_PER_CYCLE // vec_bytes))

    n_vec_bits = vec.bit_length() - 1
    outer_regs = 1 << (len(reg_b) - n_vec_bits)

    worst = 1
    for warp in range(1 << len(warp_b)):
        wc = _apply(warp_b, warp)
        for r in range(outer_regs):
            rc = _apply(reg_b[n_vec_bits:], r) if len(reg_b) > n_vec_bits else [0, 0]
            for phase0 in range(0, 64, lanes_per_phase):
                hits = [0] * BANKS
                for l in range(phase0, phase0 + lanes_per_phase):
                    lc = _apply(lane_b, l)
                    coord = tuple(a ^ b ^ c for a, b, c in zip(wc, rc, lc))
                    byte = addr_of[coord] * dtype_bytes
                    for k in range(vec_bytes // BANK_BYTES):
                        hits[((byte + k * BANK_BYTES) // BANK_BYTES) % BANKS] += 1
                worst = max(worst, max(hits))
    return {"label": label, "vec_bytes": vec_bytes, "conflict": worst}


BM, BN, BK = 256, 256, 64
A_SHAPE = [BM // 2, BK]  # [128, 64]
B_SHAPE = [BK, BN // 2]  # [64, 128]

# --- candidate global-load layouts ---------------------------------------
# "tut"  : gfx950 tutorial layout   (lanes 0-7 -> row M0, lanes 8-15 -> row M0+16)
# "cons" : consecutive-row variant  (lanes 0-7 -> row 0, lanes 8-15 -> row 1)
A_GLB = {
    "gA.tut": gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=A_SHAPE,
    ),
    "gA.cons": gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [16, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[],
        shape=A_SHAPE,
    ),
    "gA.cons8w": gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[16, 0], [32, 0], [64, 0]],
        block_bases=[],
        shape=A_SHAPE,
    ),
}
B_GLB = {
    "gB.tut": gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=B_SHAPE,
    ),
    "gB.cons": gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 16]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 1], [0, 2], [0, 4]],
        warp_bases=[[0, 32], [0, 64]],
        block_bases=[],
        shape=B_SHAPE,
    ),
    "gB.cons8w": gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 1], [0, 2], [0, 4]],
        warp_bases=[[0, 16], [0, 32], [0, 64]],
        block_bases=[],
        shape=B_SHAPE,
    ),
}


def run(name, shape, order, cols, nw):
    shared = {
        f"swz({v},{p},{m})": gl.SwizzledSharedLayout(v, p, m, order=order)
        for v, p, m in itertools.product([8], [1, 2, 4], [4, 8])
    }
    rows = []
    for sname, slay in shared.items():
        rec = {"shared": sname}
        for cname, clay in cols.items():
            try:
                r = analyze(clay, slay, shape, 2, nw)
                rec[cname] = f"{r['conflict']}x/{r['vec_bytes']}B"
            except Exception:
                rec[cname] = "ERR"
        rows.append(rec)
    hdr = ["shared"] + list(cols)
    w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in hdr}
    print(f"\n===== {name}  shape={shape} num_warps={nw} =====")
    print("  ".join(c.ljust(w[c]) for c in hdr))
    for r in rows:
        clean = all(str(r.get(c, "")).startswith("1x") for c in cols)
        print(
            "  ".join(str(r.get(c, "")).ljust(w[c]) for c in hdr)
            + ("   <== CLEAN" if clean else "")
        )


for nw, wm, wn in ((4, 2, 2), (8, 2, 4)):
    mfma = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[wm, wn]
    )
    acols = dict(A_GLB)
    acols["r.dotA.kw8"] = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    acols["r.dotA.kw4"] = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=4)
    run("A half-tile", A_SHAPE, [1, 0], acols, nw)
    bcols = dict(B_GLB)
    bcols["r.dotB.kw8"] = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    bcols["r.dotB.kw4"] = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=4)
    run("B half-tile", B_SHAPE, [0, 1], bcols, nw)
