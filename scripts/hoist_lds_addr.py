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

"""Hoist a loop-invariant LDS base address out of the loop into a dedicated VGPR.

`FA_Q_DIRECT_LDS=1` leaves one `v_add_u32_e32 vX, <imm>, vY` in the loop, forming the LDS
base for a run of `ds_read`s whose offsets no longer fit the 16-bit `offset:` field (note.md
3.29.4). `vY` is never written in the loop, so the add is loop-invariant -- the compiler is
**rematerialising** it rather than keeping the base live, because `vX` is otherwise the MFMA
accumulator and a softmax temporary for the rest of the loop. Spending one more VGPR would
let it be computed once.

This does that, on the assembly: the add moves to the loop preheader and writes a register
nothing else uses, the `ds_read`s that took `vX` as their address take the new one, and
`.amdhsa_next_free_vgpr` grows by one. `FA_HOIST_LDS_ADDR=1`.

The point is to price it, not to ship it -- one VALU per iteration is ~4 cycles of ~4460, so
the direct saving is under 0.1%. What makes it worth measuring is *where* it sits: at the head
of a 16-instruction LDS burst, which is the position the `MEMNOP` sweeps showed the kernel is
sensitive to.
"""

import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_ON = os.environ.get("FA_HOIST_LDS_ADDR", "")


def hoist(text):
    import sched_valu as S

    def opcode_of(l):
        return S.opcode(l)

    lines = text.split("\n")
    b = S._loop_bounds(lines)
    if b is None:
        return text, (0, None)
    lo, hi = b
    body = [i for i in range(lo, hi) if S.opcode(lines[i])]

    def vregs(s):
        return {n for k, n in s if k == "v"}

    written = set()
    for i in body:
        written |= vregs(S.defs_uses(lines[i])[0])

    # the candidate: an integer add whose source is not written in the loop
    cand = None
    for i in body:
        m = re.match(r"^\s*v_add_u32_e32\s+v(\d+),\s+(\S+),\s+v(\d+)\s*$",
                     lines[i].split(";")[0].rstrip())
        if not m:
            continue
        dst, imm, src = int(m.group(1)), m.group(2), int(m.group(3))
        if src in written:
            continue                      # not invariant
        cand = (i, dst, imm, src)
        break
    if cand is None:
        return text, (0, None)
    at, dst, imm, src = cand

    # the uses to redirect: ds_read/ds_write taking vdst as address, before vdst is redefined
    uses = []
    for i in body:
        if i <= at:
            continue
        if dst in vregs(S.defs_uses(lines[i])[0]):
            break
        if re.match(r"^\s*ds_(read|write)\S*\s+[^,]+,\s+v%d\b" % dst, lines[i]):
            uses.append(i)
    if not uses:
        return text, (0, None)

    # a register nothing uses: one past the high-water mark the descriptor declares
    nfv = None
    for i, l in enumerate(lines):
        m = re.match(r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", l)
        if m:
            nfv = (i, int(m.group(1)))
            break
    if nfv is None:
        return text, (0, None)
    new = nfv[1]                          # v0..v(nfv-1) are in use, so v(nfv) is free

    out = list(lines)
    ind = re.match(r"\s*", lines[at]).group(0) or "\t"
    for i in uses:
        out[i] = re.sub(r"(ds_\S+\s+[^,]+,\s+)v%d\b" % dst, r"\g<1>v%d" % new, out[i])
    out[at] = None                        # drop the in-loop add
    # Place it immediately after its source is defined, not "just before the loop label".
    # The label is not reached by fall-through: a forward `s_branch` jumps to it over the
    # preceding lines, so anything put there is skipped and the base register stays
    # uninitialised (max_err 2.88 with a small mean -- part of the output reading the wrong
    # LDS). The def of the source dominates the loop by construction, since the loop reads it.
    defs = [i for i in range(0, lo)
            if ("v", src) in S.defs_uses(lines[i])[0] and opcode_of(lines[i])]
    if not defs:
        return text, (0, None)
    site = defs[-1]
    out[site] = out[site] + f"\n{ind}v_add_u32_e32 v{new}, {imm}, v{src}"
    # The descriptor has to stay self-consistent. `accum_offset` is where the accumulation
    # registers begin and is a multiple of 4, so `next_free_vgpr` must cover it: with 246 used
    # registers the assembler emits accum_offset 248 = align4(246), and declaring only 247
    # total leaves v246 unbacked -- writes to it go nowhere and the kernel returns garbage
    # (max_err 2.88 with a tiny mean, i.e. a few wrong elements). Round the total up to 4.
    total = -(-(new + 1) // 4) * 4
    out[nfv[0]] = re.sub(r"(\.amdhsa_next_free_vgpr\s+)\d+", r"\g<1>%d" % total, out[nfv[0]])
    for i, l in enumerate(out):
        if l is None:
            continue
        m = re.match(r"^\s*\.amdhsa_accum_offset\s+(\d+)", l)
        if m and int(m.group(1)) < total:
            out[i] = re.sub(r"(\.amdhsa_accum_offset\s+)\d+", r"\g<1>%d" % total, l)

    res = []
    for l in out:
        if l is None:
            continue
        res.extend(l.split("\n")) if "\n" in l else res.append(l)
    return "\n".join(res), (len(uses), (dst, new, imm, src, nfv[1], total))


def get_key():
    return f"hoist_lds_addr:{_ON}"


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
        if not _ON:
            return asm
        out, (n, info) = hoist(asm)
        if info:
            dst, new, imm, s, old_nfv, nfv = info
            print(f"[hoist_lds_addr] v{dst} = {imm} + v{s} hoisted to v{new} "
                  f"({n} ds_read redirected, VGPRs {old_nfv} -> {nfv})",
                  file=sys.stderr, flush=True)
        else:
            print("[hoist_lds_addr] no loop-invariant LDS base add found",
                  file=sys.stderr, flush=True)
        return out

    stages["amdgcn"] = wrapper
    return get_key(), get_hash()


if __name__ == "__main__":
    out, (n, info) = hoist(open(sys.argv[1]).read())
    sys.stderr.write(f"redirected {n} uses; {info}\n")
    sys.stdout.write(out)
