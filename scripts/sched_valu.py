#!/usr/bin/env python3
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

"""Re-balance in-loop VALU across the MFMA shadows without adding or removing any.

The companion `ablate_valu.py` sweeps in-loop MFMA efficiency by *deleting* VALU, which
confounds the throughput reading with a power reading: a kernel that issues 60% fewer VALU
per second draws less power and is granted more clock. This module sweeps the same
efficiency axis by *reordering* the identical instruction stream, so the instruction mix,
the register traffic and the arithmetic are unchanged and the only variable is placement.

The knob is `FA_SCHED_VALU=<f>`, -1.0 to 1.0. At 0.0 the assembly is untouched. Positive
values spread each MFMA-bearing sub-region's VALU across that region's MFMA shadows, 1.0
being as even as its dependences allow. Negative values do the opposite: they crowd the
VALU into the earliest shadow each can reach, leaving the later MFMAs bare. Magnitudes in
between interpolate from where the compiler put each VALU toward that goal.

Spreading is the direction with no range on this kernel -- it is a dual-wave ping-pong, so
one wave's MFMA shadows are filled largely by the *other* wave's VALU, and evening out the
intra-wave balance moves in-loop MFMA efficiency by 0.2 points. Clumping is the direction
that works: it walks efficiency down over a wide span with the instruction stream fixed.

The reorder is dependency-checked, so **the kernel still computes correct attention** at
every point on the sweep -- `bench.py`'s reference comparison is the control. A pass that
merely shuffled text would produce NaN and reintroduce a data-dependent power term, which
is the confound this module exists to remove.

    FA_SCHED_VALU=0.5 FA_MODULE=fav4 DISABLE_LLVM_OPT=disable-machine-sink \
      AMDGCN_SCALARIZE_PACKED_FOPS=1 python bench.py --seqlen 8192

`FA_SCHED_DUMP=<prefix>` writes the before/after assembly. `--report` on the command line
prints the movability analysis for one file instead of rewriting it.
"""

import hashlib
import os
import re
import sys

_FRAC = os.environ.get("FA_SCHED_VALU", "")

# v_mfma_f32_32x32x16_bf16 issues one pass every 4 cycles for 8 passes; VALU may issue from
# cycle 8, so 24 cycles == 6 single-cycle-issue VALU fit in one shadow. Anything above that
# is over capacity and spills into the next MFMA's issue slot.
SHADOW_SLOTS = 6

# hazard-repair s_nop counter, reported by the hook
_NOPS = [0]


def opcode(line):
    t = line.split(";")[0].strip()
    return t.split()[0] if t and not t.startswith((".", "/")) else ""


def _operands(line):
    t = line.split(";")[0].strip()
    parts = t.split(None, 1)
    if len(parts) < 2:
        return []
    # drop trailing instruction modifiers, which are `key:value` and never registers
    body = re.sub(r"\b\w+:\[[^\]]*\]", "", parts[1])
    body = re.sub(r"\b(?:offset|offset0|offset1|op_sel\w*|blgp|cbsz|abid|neg\w*|clamp|"
                  r"mul|div|bound_ctrl|row_mask|bank_mask|dpp\w*|sdwa|glc|slc|nt|sc0|sc1)"
                  r"(?::-?\w+)?", "", body)
    return [o.strip() for o in body.split(",") if o.strip()]


def _regs(operand):
    """Register tokens an operand touches, as ('v'|'s', index) pairs."""
    out = set()
    for m in re.finditer(r"\b([vs])\[(\d+):(\d+)\]", operand):
        out.update((m.group(1), i) for i in range(int(m.group(2)), int(m.group(3)) + 1))
    for m in re.finditer(r"(?<![\w\[:])([vs])(\d+)\b", operand):
        out.add((m.group(1), int(m.group(2))))
    if "vcc" in operand:
        out.add(("special", "vcc"))
    if "exec" in operand:
        out.add(("special", "exec"))
    if re.search(r"\bscc\b", operand):
        out.add(("special", "scc"))
    if re.search(r"\bm0\b", operand):
        out.add(("special", "m0"))
    return out


# Opcodes whose first operand is the sole destination. Anything else beginning with `v_`
# is treated as reading *and* writing every register it names -- see `defs_uses`. Keeping
# this list explicit is what makes an unfamiliar instruction fail safe rather than silently
# lose a dependence: `v_permlane32_swap_b32 v96, v97` exchanges the two registers and so
# writes both, and reading it as "writes v96, reads v97" let the scheduler hoist a consumer
# of v97 above it. That produced a kernel that ran 15% faster and returned NaN.
_SIMPLE_DEST = (
    "v_mfma", "v_mov_b32", "v_mov_b64", "v_add_f", "v_sub_f", "v_subrev_f", "v_mul_f",
    "v_fma_f", "v_fmaak", "v_fmamk", "v_max_f", "v_min_f", "v_maximum", "v_minimum",
    "v_max3", "v_min3", "v_med3", "v_cmp_", "v_cndmask", "v_cvt_", "v_exp_f", "v_log_f",
    "v_rcp_", "v_rsq_", "v_sqrt_", "v_trunc_", "v_floor_", "v_ceil_", "v_rndne_",
    "v_ldexp_", "v_and_b", "v_or_b", "v_xor_b", "v_not_b", "v_bfe_", "v_bfi_", "v_lshl",
    "v_lshr", "v_ashr", "v_alignbit", "v_alignbyte", "v_perm_b32", "v_sad_", "v_pack_",
    "v_pk_add", "v_pk_sub", "v_pk_mul", "v_pk_fma", "v_pk_max", "v_pk_min",
    "v_readfirstlane", "v_readlane", "v_bcnt_", "v_ffbl_", "v_ffbh_", "v_cvt_pk",
)
# Accumulate-in-place forms read their destination as well.
_ACC_DEST = ("v_fmac", "v_mac_", "v_pk_fmac", "v_dot", "v_mad_mix", "v_fma_mix")

# Transcendentals. Their result is the second software-enforced hazard class here, next to
# MFMA: a consumer issued too soon after one reads a register that is not written yet, and
# the compiler covers it with whatever independent work already sat in the gap rather than
# an `s_nop`. Drain that gap and the kernel returns NaN while staying correctly ordered.
_TRANS = ("v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_",
          "v_tanh_", "v_rcp_iflag")

# Cross-lane ops read the whole wave's copy of their source, and a VALU write feeding one
# needs a wait state that nothing in the dataflow expresses. FlyDSL pairs every
# `v_mov / v_permlane32_swap_b32` with an `s_nop 1`; ours does the same.
_XLANE = ("v_permlane", "v_ds_permute", "v_ds_bpermute", "v_mov_b32_dpp")
_XLANE_WS = 2   # GCNHazardRecognizer::checkPermlaneHazards, VALUWritesVDstWaitStates
_M0_WS = 1      # s_mov to m0 -> an instruction that reads m0


def defs_uses(line):
    """(written, read) register sets. Deliberately over-approximate: an operand we cannot
    classify is counted as both, which can only make the scheduler more conservative."""
    op = opcode(line)
    ops = _operands(line)
    if not op or not ops:
        return set(), set()
    w, r = set(), set()
    if op.startswith("v_") and not op.startswith(_SIMPLE_DEST) and not op.startswith(_ACC_DEST):
        # Unknown vector op: assume every register it names is both read and written. A
        # cross-lane swap or a multi-destination form then cannot be reordered past anything
        # it touches, which is the safe reading.
        for o in ops:
            w |= _regs(o)
            r |= _regs(o)
        return w, r
    if op.startswith(("v_", "ds_read", "buffer_load", "global_load", "flat_load", "s_load",
                      "v_readfirstlane", "s_mov", "s_add", "s_sub", "s_mul", "s_and",
                      "s_or", "s_xor", "s_lshl", "s_lshr", "s_ashr", "s_min", "s_max",
                      "s_cselect", "s_bfe", "s_not", "s_pack", "s_cvt")):
        w |= _regs(ops[0])
        for o in ops[1:]:
            r |= _regs(o)
        # read-modify-write forms: MFMA srcC, and any *_co_ carry operand
        if op.startswith("v_mfma") and len(ops) >= 4:
            r |= _regs(ops[3])
        if op.startswith(_ACC_DEST):
            r |= _regs(ops[0])
        # An `lds` modifier makes a buffer load a global->LDS DMA, whose LDS base comes
        # from m0. Nothing names m0 in the operand list, so add it by hand -- without this
        # the `s_mov_b32 m0, sN` before it looks independent, and the `s_nop` guarding the
        # m0-write hazard looks removable. It is not: removing those six nops in FlyDSL's
        # loop leaves the DMA writing to a stale LDS offset (max_err 2.2e-02).
        if re.search(r"\blds\b", line.split(";")[0]):
            r.add(("special", "m0"))
        if "_co_" in op or op.startswith(("v_add_c", "v_sub_c")):
            w.add(("special", "vcc"))
            r.add(("special", "vcc"))
    else:
        # stores, barriers, branches, waitcnt, exec manipulation: read everything named
        for o in ops:
            r |= _regs(o)
        if op.startswith(("s_and_saveexec", "s_or_b", "s_mov_b64", "s_mov_b32",
                          "s_andn2", "s_cbranch", "s_cmp")):
            w |= _regs(ops[0]) if ops else set()
    if op.startswith(("s_cmp", "v_cmp")):
        w.add(("special", "scc" if op.startswith("s_") else "vcc"))
    return w, r


def _is_valu(op):
    return op.startswith("v_") and not op.startswith("v_mfma")


def _regions(lines, lo, hi):
    """Barrier-delimited sub-regions of the loop body, as (start, end) index pairs."""
    out, start = [], lo
    for i in range(lo, hi):
        if opcode(lines[i]) == "s_barrier":
            out.append((start, i + 1))
            start = i + 1
    if start < hi:
        out.append((start, hi))
    return out


def _pinned(lines, a, b):
    """Indices that must not move: anything inside an exec-modified span, and the
    hazard `s_nop`s, whose spacing is the only thing standing between a moved VALU and an
    MFMA result that is not yet architecturally readable."""
    pin, depth = set(), 0
    for i in range(a, b):
        op = opcode(lines[i])
        if op.startswith("s_and_saveexec"):
            depth += 1
        if depth:
            pin.add(i)
        if op.startswith(("s_or_b64", "s_or_b32")) and "exec" in lines[i]:
            depth = max(0, depth - 1)
            pin.add(i)
    return pin


def analyse(lines, lo, hi, verbose=False):
    """Per-region movability of the VALU, and the even-spread target."""
    rep = []
    for a, b in _regions(lines, lo, hi):
        idx = list(range(a, b))
        mfma = [i for i in idx if opcode(lines[i]).startswith("v_mfma")]
        valu = [i for i in idx if _is_valu(opcode(lines[i]))]
        if not mfma:
            continue
        pin = _pinned(lines, a, b)
        # A VALU is movable if it is not pinned and does not depend on -- or feed -- an
        # MFMA of this region. Those are the ones whose placement is free.
        mw = set()
        for i in mfma:
            mw |= defs_uses(lines[i])[0]
        free = []
        for i in valu:
            if i in pin:
                continue
            w, r = defs_uses(lines[i])
            if (r & mw) or (w & mw):
                continue
            free.append(i)
        rep.append(dict(a=a, b=b, mfma=len(mfma), valu=len(valu), free=len(free),
                        per=len(valu) / len(mfma), pinned=len(set(valu) & pin)))
        if verbose:
            r = rep[-1]
            print(f"  region {a}..{b}: {r['mfma']:3d} mfma  {r['valu']:3d} valu "
                  f"({r['per']:.2f}/shadow, capacity {SHADOW_SLOTS})  "
                  f"movable {r['free']:3d}  pinned-by-exec {r['pinned']}")
    return rep


def _items(reg):
    """Region lines grouped into instructions, each carrying the comment/blank lines that
    preceded it so a moved instruction takes its annotation along."""
    items, pending = [], []
    for line in reg:
        if opcode(line):
            items.append([pending, line])
            pending = []
        else:
            pending.append(line)
    return items, pending


def _movable(items, pin):
    """Which items may be re-placed.

    A VALU is movable only if it shares **no register** with any MFMA in the region --
    neither one the MFMA writes nor one it reads. That is a stronger test than dataflow
    correctness needs, and deliberately so: MFMA operand and accumulator hazards are
    enforced in software on this part, by wait states the compiler counted out for the
    original spacing. A VALU with no register in common with any MFMA cannot be subject to
    one, so its placement is free with no re-counting. VALU-to-VALU dependences, by
    contrast, are fully interlocked in hardware, so the scheduler only has to order them.
    """
    mregs = set()
    for k, (_, line) in enumerate(items):
        if opcode(line).startswith("v_mfma"):
            w, r = defs_uses(line)
            mregs |= w | r
    out = []
    for k, (_, line) in enumerate(items):
        if k in pin or not _is_valu(opcode(line)):
            continue
        w, r = defs_uses(line)
        if (w | r) & mregs:
            continue
        out.append(k)
    return set(out)


def _deps(items, movable):
    """succ/pred adjacency over items: register RAW/WAR/WAW, the original order of every
    fixed item, and `s_nop`/`s_waitcnt`/`s_barrier` as walls no VALU may cross."""
    n = len(items)
    du = [defs_uses(l) for _, l in items]
    edges = set()
    for i in range(n):
        wi, ri = du[i]
        for j in range(i + 1, n):
            wj, rj = du[j]
            if (rj & wi) or (wj & wi) or (wj & ri):
                edges.add((i, j))
    fixed = [i for i in range(n) if i not in movable]
    for a, b in zip(fixed, fixed[1:]):
        edges.add((a, b))
    walls = [i for i, (_, l) in enumerate(items)
             if opcode(l).startswith(("s_nop", "s_waitcnt", "s_barrier"))]
    for w in walls:
        for m in movable:
            edges.add((m, w) if m < w else (w, m))
    pred = [set() for _ in range(n)]
    succ = [set() for _ in range(n)]
    for a, b in edges:
        succ[a].add(b)
        pred[b].add(a)
    return pred, succ


# Hazard distances are counted in **wait states**, one per instruction, exactly as
# `SIInstrInfo::getNumWaitStates` does -- not in cycles. Getting this wrong the other way
# (MFMA = 32, VALU = 4) made the repair below preserve distances of several hundred, which
# it padded with `s_nop` chains that cost more than the whole reorder gained: in-loop MFMA
# efficiency fell from 80.8% to 32.8%.
_MAX_MFMA_WS = 19   # GCNHazardRecognizer MaxWaitStates, SMFMA 32x32 on gfx950
_TRANS_WS = 1       # GCNSubtarget::hasTransForwardingHazard(), gfx940 and later


def _ws(line):
    """Wait states an instruction contributes."""
    op = opcode(line)
    if op.startswith("s_nop"):
        m = re.search(r"s_nop\s+(\d+)", line)
        return (int(m.group(1)) + 1) if m else 1
    return 1 if op else 0


def _hazard_pairs(items):
    """(producer, consumer, original cycle distance) for every register dependence whose
    spacing the hardware does not enforce for us -- one involving an MFMA, or one out of a
    transcendental.

    MFMA operand/accumulator hazards and the gfx940+ transcendental forwarding hazard are
    software-enforced, and the compiler satisfied them partly with `s_nop` and partly by
    letting independent VALU sit in the gap. Moving those VALU away shortens the gap, which
    no dataflow check can see: the schedule is still correctly ordered, it is just illegal.

    The requirement is the architectural one, capped -- not the original spacing, which is
    often far larger than any hazard needs and whose preservation is ruinously expensive.
    """
    n = len(items)
    du = [defs_uses(l) for _, l in items]
    ismfma = [opcode(l).startswith("v_mfma") for _, l in items]
    istrans = [opcode(l).startswith(_TRANS) for _, l in items]
    isvalu = [_is_valu(opcode(l)) for _, l in items]
    isxlane = [opcode(l).startswith(_XLANE) for _, l in items]
    cyc = [_ws(l) for _, l in items]
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + cyc[i]
    out = []
    for i in range(n):
        wi, ri = du[i]
        for j in range(i + 1, n):
            wj, rj = du[j]
            dep = (rj & wi) or (wj & wi) or (wj & ri)
            if not dep:
                continue
            cap = 0
            if ("special", "m0") in (rj & wi):
                cap = _M0_WS
            elif isxlane[j] and not isxlane[i]:
                cap = _XLANE_WS
            elif ismfma[i] or ismfma[j]:
                cap = _MAX_MFMA_WS
            elif istrans[i] and isvalu[j] and not istrans[j]:
                cap = _TRANS_WS
            if not cap:
                continue
            out.append((i, j, min(pref[j] - pref[i + 1], cap)))
    return out


def schedule(lines, lo, hi, frac):
    """Re-emit each MFMA-bearing region with its VALU spread `frac` of the way to even.

    A greedy list schedule over the dependence DAG: repeatedly emit whichever ready item
    has the earliest target position. Because the order is always topological the region
    computes exactly what it did before; because the priority is the target position, the
    VALU end up as close to an even spread across the MFMA shadows as their dependences
    allow.
    """
    moved = 0
    only = os.environ.get("FA_SCHED_ONLY", "")
    only = {int(x) for x in only.split(",") if x.strip()} if only else None
    dot_seen = -1
    out = lines[:lo]
    prev_end = lo
    for a, b in _regions(lines, lo, hi):
        out += lines[prev_end:a]
        prev_end = b
        reg = lines[a:b]
        items, tail = _items(reg)
        mfma = [i for i, (_, l) in enumerate(items) if opcode(l).startswith("v_mfma")]
        if not mfma:
            out += reg
            continue
        dot_seen += 1
        if only is not None and dot_seen not in only:
            out += reg
            continue
        pin = _pin_items(lines, a, b, items)
        movable = _movable(items, pin)
        if not movable:
            out += reg
            continue

        # Which shadow each movable VALU sits in now, and which one an even spread wants.
        shadow, cur = {}, -1
        for k, (_, l) in enumerate(items):
            if k in mfma:
                cur += 1
            if k in movable:
                shadow[k] = max(0, cur)
        k_sh = len(mfma)
        order = sorted(movable)
        target = {}
        for rank, i in enumerate(order):
            if frac >= 0:
                # spread: rank r of n belongs in shadow r*k/n
                goal = min(k_sh - 1, int(rank * k_sh / len(order)))
            else:
                # clump: everything crowds into the earliest shadow its dependences allow,
                # which leaves the later MFMAs bare. The opposite extreme from spreading,
                # and the direction with real range on a kernel whose compiler-chosen
                # schedule is already reasonably balanced.
                goal = 0
            want = shadow[i] + abs(frac) * (goal - shadow[i])
            target[i] = max(0, min(k_sh - 1, int(round(want))))

        # Target position in item units: just after the MFMA that opens the shadow, with a
        # tie-break that preserves the original relative order inside a shadow.
        prio = {}
        for i, (_, l) in enumerate(items):
            prio[i] = float(i)
        for rank, i in enumerate(order):
            prio[i] = mfma[target[i]] + 0.5 + 1e-6 * rank

        pred, succ = _deps(items, movable)
        indeg = [len(pred[i]) for i in range(len(items))]
        ready = [i for i in range(len(items)) if indeg[i] == 0]
        seq = []
        while ready:
            ready.sort(key=lambda i: (prio[i], i))
            i = ready.pop(0)
            seq.append(i)
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
        assert len(seq) == len(items), f"cycle in region {a}: {len(seq)}/{len(items)}"
        moved += sum(1 for r, i in enumerate(seq) if i != r)

        # Emit, restoring any MFMA hazard distance the reorder shortened.
        need = {}
        for i, j, d in _hazard_pairs(items):
            need.setdefault(j, []).append((i, d))
        indent = re.match(r"\s*", items[seq[0]][1]).group(0) or "\t"
        res, cum, end_at = [], 0, {}
        for i in seq:
            deficit = 0
            for p, d in need.get(i, ()):
                if p in end_at:
                    deficit = max(deficit, d - (cum - end_at[p]))
            while deficit > 0:
                k = min(15, deficit - 1)
                res.append(f"{indent}s_nop {k}")
                cum += k + 1
                deficit -= k + 1
                _NOPS[0] += 1
            res += items[i][0]
            res.append(items[i][1])
            cum += _ws(items[i][1])
            end_at[i] = cum
        res += tail
        out += res
    out += lines[prev_end:]
    return out, moved


def relax_vmcnt(lines, lo, hi):
    """Replace a blanket `s_waitcnt vmcnt(0)` with counted waits placed at each consumer.

    A `vmcnt(0)` drains every outstanding VMEM load before anything may proceed. When the
    loads are all the same type their returns are in order, so `vmcnt(N)` guarantees the
    first (issued - N) have landed, and the consumers of the earlier loads can start while
    the later ones are still in flight. This walks forward from the wait, tracks which load
    each live register came from, and emits the loosest legal count in front of each
    consumer -- then restores a full drain at the end of the region, so nothing that used to
    be covered by the blanket wait is left uncovered.

    Used on the prologue, where `fav4` spends 3512 cycles per wave on one such wait: the Q
    tile, which `SCALE_ON_Q` must have in registers before it can be scaled and stored to
    LDS.
    """
    ins = [i for i in range(lo, hi) if opcode(lines[i])]
    loads = [i for i in ins
             if opcode(lines[i]).startswith(("global_load", "buffer_load", "flat_load"))
             and "lds" not in lines[i].split(";")[0]]
    waits = [i for i in ins if opcode(lines[i]) == "s_waitcnt" and "vmcnt(0)" in lines[i]]
    if not loads or not waits:
        return lines, (0, 0)
    pre = {}                                   # line -> text to insert before it
    moved = 0
    for w in waits:
        before = [k for k in loads if k < w]
        if len(before) < 2:
            continue
        owner = {}
        for n, k in enumerate(before):
            for r in defs_uses(lines[k])[0]:
                owner[r] = n
        ind = re.match(r"\s*", lines[w]).group(0) or "\t"
        waited, last = -1, len(before) - 1
        for i in ins:
            if i <= w:
                continue
            wr, rd = defs_uses(lines[i])
            need = max((owner[r] for r in rd if r in owner), default=-1)
            if need > waited:
                waited = need
                pre.setdefault(i, []).append(f"{ind}s_waitcnt vmcnt({last - need})")
            for r in wr:
                owner.pop(r, None)
            if waited >= last:
                break
        if waited < 0:                          # nothing consumed: leave it alone
            continue
        if waited < last:
            # some of the loads are not read in this region; the blanket wait covered them,
            # so put the drain back at the end rather than dropping it
            pre.setdefault(ins[-1], []).append(f"{ind}s_waitcnt vmcnt(0)")
        lines = list(lines)
        lines[w] = None
        moved += 1
    out, added = [], 0
    for i, l in enumerate(lines):
        for extra in pre.get(i, ()):
            out.append(extra)
            added += 1
        if l is not None:
            out.append(l)
    return out, (moved, added)


def strip_pacing(lines, lo, hi, drop=("s_nop",)):
    """Delete scalar pacing instructions from the loop, re-inserting only the `s_nop` the
    architecture actually requires.

    Neither `s_nop` nor `s_setprio` moves data, so removing them cannot change what the loop
    computes -- but `s_nop` can be load-bearing for the MFMA and transcendental hazards. So
    the requirements are computed on the *original* stream (capped at the architectural
    figure, as `_hazard_pairs` does) and re-satisfied on the stripped one, which leaves the
    minimum legal spacing rather than whatever the producer chose.

    This is the lever for a loop whose VALU already fits in its MFMA shadows: there the
    cycles above the ideal are pacing and arbitration, and no amount of VALU re-placement
    touches them.
    """
    out, prev, removed, readded = lines[:lo], lo, 0, 0
    for a, b in _regions(lines, lo, hi):
        out += lines[prev:a]
        prev = b
        items, tail = _items(lines[a:b])
        if not items:
            out += lines[a:b]
            continue
        need = {}
        for i, j, d in _hazard_pairs(items):
            need.setdefault(j, []).append((i, d))
        indent = re.match(r"\s*", items[0][1]).group(0) or "\t"
        res, cum, end_at = [], 0, {}
        for k, (pre, line) in enumerate(items):
            if opcode(line).startswith(drop):
                # keep the attached lines -- they include the loop label and the inline-asm
                # markers, and dropping the label with the instruction breaks the back edge
                res += pre
                removed += 1
                continue
            deficit = 0
            for pk, d in need.get(k, ()):
                if pk in end_at:
                    deficit = max(deficit, d - (cum - end_at[pk]))
            while deficit > 0:
                q = min(15, deficit - 1)
                res.append(f"{indent}s_nop {q}")
                cum += q + 1
                deficit -= q + 1
                readded += 1
            res += pre
            res.append(line)
            cum += _ws(line)
            end_at[k] = cum
        res += tail
        out += res
    out += lines[prev:]
    return out, (removed, readded)


def _pin_items(lines, a, b, items):
    """Item indices inside an exec-modified span."""
    pinned_lines = _pinned(lines, a, b)
    pin, k = set(), 0
    for i in range(a, b):
        if opcode(lines[i]):
            if i in pinned_lines:
                pin.add(k)
            k += 1
    return pin


def _loop_bounds(lines):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ablate_valu import _loop_bounds as lb
    return lb(lines)


def rewrite(text, frac):
    lines = text.split("\n")
    b = _loop_bounds(lines)
    if b is None:
        return text, (0, 0)
    relaxed = 0
    if os.environ.get("FA_RELAX_VMCNT"):
        lines, (nw, relaxed) = relax_vmcnt(lines, 0, b[0])
        b = _loop_bounds(lines)
    if frac == 0:
        return "\n".join(lines), (0, relaxed)
    _NOPS[0] = 0
    out, moved = schedule(lines, b[0], b[1], frac)
    return "\n".join(out), (moved, _NOPS[0] + relaxed)


def get_key():
    return f"sched_valu:{_FRAC}:only{os.environ.get('FA_SCHED_ONLY', '')}"


def get_hash():
    return hashlib.sha256(get_key().encode("utf-8")).hexdigest()


def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(a is None for a in (stages, options, language, capability)):
        return get_key(), get_hash()
    if stages is None or "amdgcn" not in stages:
        return get_key(), get_hash()
    orig = self.make_amdgcn

    def wrapper(src, metadata):
        asm = orig(src, metadata, options)
        out, (moved, nops) = rewrite(asm, float(_FRAC or 0))
        dump = os.environ.get("FA_SCHED_DUMP")
        if dump:
            open(dump + ".orig.amdgcn", "w").write(asm)
            open(dump + ".sched.amdgcn", "w").write(out)
        print(f"[sched_valu] frac={_FRAC or 0} moved {moved} VALU between MFMA shadows"
              + (f", {nops} s_nop inserted to restore hazard distance" if nops else ""),
              file=sys.stderr, flush=True)
        return out

    stages["amdgcn"] = wrapper
    return get_key(), get_hash()


if __name__ == "__main__":
    src = open(sys.argv[1]).read()
    lines = src.split("\n")
    b = _loop_bounds(lines)
    if "--report" in sys.argv:
        print(f"loop body {b[0]}..{b[1]}")
        analyse(lines, b[0], b[1], verbose=True)
    else:
        out, (moved, nops) = rewrite(src, float(sys.argv[2]))
        sys.stderr.write(f"moved {moved} VALU, inserted {nops} s_nop\n")
        sys.stdout.write(out)
