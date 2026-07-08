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

from collections import defaultdict
from typing import Set, Tuple, Optional
import re
import logging
import argparse

NO_DEF_OPS = {
    's_waitcnt',
    's_nop',
    's_branch',
    's_cbranch_scc0',
    's_cbranch_scc1',
}


## TODO(lixun)
## Only buffer_load lds should be included in ALL_USERS set
ALL_USERS = ('s_cmp', 'v_permlane', 'buffer_store', 'ds_write', 'ds_store')
ALL_DEFS_USES = ('v_permlane', )
COPY_DATA = ('v_accvgpr_read', 'v_accvgpr_write', 'v_accvgpr_mov', 'v_mov', 'scratch_load', 'scratch_store')

# The full architected register file (gfx950): v0-v255, a0-a255, s0-s101. Used as
# the universe when computing each block's free registers.
ALL_REGS = ({('v', i) for i in range(256)} | {('a', i) for i in range(256)} | {('s', i) for i in range(102)})


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def dbg(msg, indent=0):
    logging.debug(" " * indent + msg)


class Register:

    def __init__(self, kind, ids):
        self.kind = kind  # 's', 'v', 'a', 'm', 'l'
        self.ids = ids  # list of ints

        if len(self.ids) > 1:
            expected = list(range(self.ids[0], self.ids[0] + len(self.ids)))
            if self.ids != expected:
                raise ValueError(f"Non-contiguous register ids: {self.ids}")

    @property
    def start(self):
        return min(self.ids)

    @property
    def size(self):
        return len(self.ids)

    @property
    def end(self):
        return max(self.ids)

    def __hash__(self):
        return hash((self.kind, tuple(sorted(self.ids))))

    def __eq__(self, other):
        return self.kind == other.kind and self.ids == other.ids

    def emit(self):
        return self.__repr__()

    def __repr__(self):
        if len(self.ids) == 1:
            return f"{self.kind}{next(iter(self.ids))}"
        return f"{self.kind}[{min(self.ids)}:{max(self.ids)}]"


class Instruction:

    def __init__(self, opcode, operands, regs_by_operand, loc, raw_line, bb):
        self.opcode = opcode
        self.operands = operands
        self.regs_by_operand = regs_by_operand  # List[List[Register]]
        self.loc = loc
        self.raw_line = raw_line

        self.defs = set()  # set[(kind, id)]
        self.uses = set()

        self.users = set()  # instructions that read registers produced by this instruction

        self.index = None  # position inside BB

        self.mark_dead: bool = False

    def emit(self):
        if self.opcode.startswith("."):
            return self.raw_line
        if not self.operands:
            return self.opcode
        return f"{self.opcode} " + ", ".join(self.operands)

    def get_dst_regs(self) -> Register:
        if (not self.operands) or (not self.regs_by_operand):
            return None
        return parse_register(self.operands[0])

    def get_src_regs(self) -> list[Register]:
        regs = []
        if len(self.regs_by_operand) > 1:
            for op in self.operands[1:]:
                regs.extend(extract_registers(op))
        return regs

    def replace_dst(self, reg: Register):
        assert len(self.regs_by_operand) > 0
        self.operands[0] = reg.emit()
        self.update_regs_by_operand()

    def update_regs_by_operand(self):
        self.regs_by_operand = []
        for op in self.operands:
            self.regs_by_operand.extend(extract_registers(op))

    ## Replace old_reg with new_reg in this instruction
    def replace_reg(self, old_reg: Register, new_reg: Register):
        for i, op in enumerate(self.operands):
            if op == old_reg.emit():
                self.operands[i] = new_reg.emit()
                break
        self.update_regs_by_operand()

    ## Replace old_reg with new_reg in this instruction's src regs
    def replace_use_reg(self, old_reg: Register, new_reg: Register, repeat=10):
        cnt = 0
        for i, op in enumerate(self.operands):
            if i == 0:
                continue
            if op == old_reg.emit():
                self.operands[i] = new_reg.emit()
                cnt += 1
            if cnt >= repeat:
                break
        self.update_regs_by_operand()

    ## Replace the uses of this inst.def with reg
    def replace_users_with(self, reg: Register, repeat=10):
        def_reg = self.get_dst_regs()
        for user in self.users:
            if user.defs:
                user.replace_use_reg(def_reg, reg, repeat)
            else:
                user.replace_reg(def_reg, reg)

    # ---------- classification ----------
    def is_memory(self):
        return (self.opcode.startswith("ds_") or self.opcode.startswith("buffer_"))

    def is_control(self):
        return self.opcode.startswith("s_branch") or self.opcode.startswith("s_cbranch")

    def is_cmp(self):
        return self.opcode.startswith("s_cmp") or self.opcode.startswith("v_cmp")

    def is_pure(self):
        return not (self.is_memory() or self.is_control() or self.is_cmp())

    # ---------- MFMA-only helpers ----------
    def is_mfma(self) -> bool:
        return self.opcode.startswith("v_mfma")


    # ---------- copy inst ----------
    def is_copy(self):
        return self.opcode.startswith(COPY_DATA)

    def get_copy_src(self) -> Register | str:
        '''
        return the source register of this copy instruction
        src can be
        1. s, a, v
        2. l: this refers to a memory loc with off, e.g.
           l0 means the src of
           scratch_load_dword off v0, off, off
           l4 means the src of
           scratch_load_dword off v0, off, off offset:4
        '''
        assert self.is_copy()
        if self.opcode.startswith("v_"):
            src = self.get_src_regs()
            return src[0] if src else self.operands[1]
        if "load" in self.opcode:
            return self._scratch_loc_reg()
        if "store" in self.opcode:
            return parse_register(self.operands[1])

    def _scratch_loc_reg(self) -> Register:
        # An 'l' register names a scratch memory location, keyed by its byte offset
        # (0 when the instruction carries no explicit offset).
        off = self.operands[-1]
        if 'offset' not in off:
            return Register('l', [0])
        return Register('l', [int(off.split(':')[1])])

    def get_copy_dst(self) -> Register:
        assert self.is_copy()
        if self.opcode.startswith("v_"):
            return self.get_dst_regs()
        if "load" in self.opcode:
            return self.get_dst_regs()
        if "store" in self.opcode:
            return self._scratch_loc_reg()

    def compute_def_use(self):
        self.defs.clear()
        self.uses.clear()

        if self.opcode in NO_DEF_OPS:
            return

        if 'scratch_load' in self.opcode:
            self.defs |= flatten_regs(self.regs_by_operand[0])
            return

        if 'scratch_store' in self.opcode:
            self.uses |= flatten_regs(self.regs_by_operand[0])
            return

        if self.opcode.startswith(ALL_USERS) or \
           (self.opcode.startswith('buffer_load') and 'lds' in self.operands):
            for reg in self.regs_by_operand:
                self.uses |= flatten_regs(reg)
            return

        if self.opcode.startswith(ALL_DEFS_USES):
            for reg in self.regs_by_operand:
                self.defs |= flatten_regs(reg)
                self.uses |= flatten_regs(reg)
            return

        # Normal case
        if self.regs_by_operand:
            self.defs |= flatten_regs(self.regs_by_operand[0])
            if len(self.regs_by_operand) > 1:
                for reg in self.regs_by_operand[1:]:
                    self.uses |= flatten_regs(reg)


class BasicBlock:

    def __init__(self, name):
        self.name = name
        self.instructions = []
        self.succs = []  # filled later
        self.preds = []

        self.defs = set()
        self.uses = set()

        self.live_in = set()
        self.live_out = set()

        self.free_regs = set()  # all_reg - (defs | uses | live_in)
        ## memory locations this block read from and write to
        ## set of (kind, idx) objects
        self.read_from = set()
        self.write_to = set()

    def add_inst(self, inst):
        inst.index = len(self.instructions)
        self.instructions.append(inst)

    def instructions_before(self, inst, including=False):
        idx = self.instructions.index(inst)
        if including:
            return self.instructions[:idx + 1]
        else:
            return self.instructions[:idx]

    def next_instruction(self, inst):
        if inst == self.instructions[-1]:
            return None
        idx = self.instructions.index(inst)
        return self.instructions[idx + 1]

    def get_reaching_defs(self, inst, reg, including=False):
        """
        reg: Register (possibly a range)
        Returns: set of Instructions
        """
        needed = set(reg.ids)
        reaching_defs = set()

        for prev in reversed(self.instructions_before(inst, including)):
            if not prev.defs:
                continue
            dreg = coalesce_regs(prev.defs)[0]
            if not dreg:
                continue
            if dreg.kind != reg.kind:
                continue

            overlap = needed & set(dreg.ids)
            if overlap:
                reaching_defs.add(prev)
                needed -= overlap

            if not needed:
                break

        return reaching_defs

    def is_loop(self):
        if len(self.preds) == 0 or len(self.succs) == 0:
            return False
        return (self in self.preds) and (self in self.succs)

    def is_prologue(self):
        if len(self.succs) == 0:
            return False

        for succ in self.succs:
            if succ != self and succ.is_loop():
                return True

        return False

    def cleanup_bb(self):
        self.instructions = [inst for inst in self.instructions if not getattr(inst, "mark_dead", False)]

    def swap_inst(self, ida, idb):
        s = len(self.instructions)
        if ida >= s or idb >= s:
            return
        if ida == idb:
            return
        self.instructions[ida], self.instructions[idb] = self.instructions[idb], self.instructions[ida]

    def compute_bb_def_use(self):
        self.defs.clear()
        self.uses.clear()

        for inst in self.instructions:
            if inst.mark_dead:
                continue
            inst.compute_def_use()

            for u in inst.uses:
                if u not in self.defs:
                    self.uses.add(u)

            self.defs |= inst.defs


class Program:

    def __init__(self):
        self.header_lines = []  # before first BB
        self.blocks = []  # parsed basic blocks
        self.tail_lines = []  # after s_endpgm

    def get_prologue(self):
        for bb in self.blocks:
            if bb.is_prologue():
                return bb

    def get_loop(self):
        for bb in self.blocks:
            if bb.is_loop():
                return bb

    def update_inst_index(self):
        for bb in self.blocks:
            for idx, inst in enumerate(bb.instructions):
                inst.index = idx

    def compute_liveness(self, indent):

        dbg("========== liveness ==========", indent)
        changed = True
        for bb in self.blocks:
            bb.live_in = set()
            bb.live_out = set()

        while changed:
            changed = False

            for bb in reversed(self.blocks):
                new_out = set()
                for s in bb.succs:
                    new_out |= s.live_in

                new_in = bb.uses | (new_out - bb.defs)

                if new_out != bb.live_out or new_in != bb.live_in:
                    bb.live_out = new_out
                    bb.live_in = new_in
                    changed = True

        dbg("========== liveness done =====", indent)

    def build_cfg(self):
        label_map = {bb.name: bb for bb in self.blocks}

        for i, bb in enumerate(self.blocks):
            bb.preds = []
            bb.succs = []
            if not bb.instructions:
                continue

            last = bb.instructions[-1].opcode
            text = bb.instructions[-1].raw_line

            if last == 's_branch':
                tgt = text.split()[-1]
                bb.succs.append(label_map[tgt])

            elif last.startswith('s_cbranch'):
                tgt = text.split()[-1]
                bb.succs.append(label_map[tgt])
                if i + 1 < len(self.blocks):
                    bb.succs.append(self.blocks[i + 1])

            else:
                if i + 1 < len(self.blocks):
                    bb.succs.append(self.blocks[i + 1])

        for bb in self.blocks:
            for s in bb.succs:
                s.preds.append(bb)

    def update_free_regs(self, indent):

        dbg("========== update blocks regs ==========", indent)
        for bb in self.blocks:
            bb.compute_bb_def_use()

        self.build_cfg()
        self.compute_liveness(indent + 2)

        for bb in self.blocks:
            live_in = bb.live_in
            bb_uses = bb.defs | bb.uses
            live_through = live_in - bb_uses
            free_regs = ALL_REGS - bb_uses - live_through
            bb.free_regs = free_regs
            ## collect memory locations for scratch load and store
            bb.read_from = set()
            bb.write_to = set()
            for inst in bb.instructions:
                if inst.mark_dead:
                    continue
                if 'scratch_load' in inst.opcode:
                    bb.read_from |= flatten_regs(inst.get_copy_src())
                if 'scratch_store' in inst.opcode:
                    bb.write_to |= flatten_regs(inst.get_copy_dst())

        dbg("========== update blocks regs done =====", indent)

    def build_def_use_chains_linear(self, indent):
        """
        Correct reaching-definition-based def-use chains across blocks
        """

        dbg("========== build def-use chains ==========", indent)
        current_def = {}  # (kind, id) -> Instruction

        for bb in self.blocks:
            for inst in bb.instructions:
                inst.users.clear()
                # ---------
                # Uses: link this inst as a user of each reg's current producer
                # ---------
                for reg in inst.uses:
                    if reg in current_def:
                        current_def[reg].users.add(inst)

                # ---------
                # Defs: overwrite current definition
                # ---------
                for reg in inst.defs:
                    current_def[reg] = inst

        dbg("========== build def-use chains done =====", indent)

    def process_blocks(self, indent):

        dbg("========== process blocks =========", indent)

        self.update_free_regs(indent + 2)

        self.build_def_use_chains_linear(indent + 2)

        dbg("========== process blocks done ====", indent)


REG_PATTERNS = [
    r'(?<![A-Za-z0-9_])s\d+(?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])v\d+(?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])a\d+(?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])m0(?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])s\[\d+:\d+\](?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])v\[\d+:\d+\](?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])a\[\d+:\d+\](?![A-Za-z0-9_])',
]

REGEXES = [re.compile(p) for p in REG_PATTERNS]


def parse_register(text):
    if text == 'm0':
        return Register('m', [0])

    kind = text[0]
    if '[' in text:
        lo, hi = map(int, text[text.find('[') + 1:text.find(']')].split(':'))
        return Register(kind, list(range(lo, hi + 1)))
    else:
        return Register(kind, [int(text[1:])])


def split_top_level_commas(text: str) -> list[str]:
    parts = []
    cur = []
    depth = 0

    for ch in text:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1

        if ch == ',' and depth == 0:
            part = ''.join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch)

    part = ''.join(cur).strip()
    if part:
        parts.append(part)

    return parts


def split_by_whitespace(text: str) -> list[str]:
    tokens = []
    cur = []
    depth = 0

    for ch in text:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1

        if ch.isspace() and depth == 0:
            tok = ''.join(cur).strip()
            if tok:
                tokens.append(tok)
            cur = []
        else:
            cur.append(ch)

    tok = ''.join(cur).strip()
    if tok:
        tokens.append(tok)

    return tokens


def split_operands(text: str) -> list[str]:
    if not text:
        return []

    operands = []
    for part in split_top_level_commas(text):
        operands.extend(split_by_whitespace(part))

    return operands


def extract_registers(op: str):
    regs = []
    for rx in REGEXES:
        for m in rx.findall(op):
            regs.append(parse_register(m))
    return regs


def parse_instruction(line, loc, bb):
    line = line.strip()
    if not line or line.startswith(';'):
        return None

    # Remove trailing comments
    line = line.split(';', 1)[0].strip()

    parts = line.split(None, 1)
    opcode = parts[0]

    operand_text = parts[1] if len(parts) > 1 else ""
    raw_operands = split_operands(operand_text)

    operands = []
    regs_by_operand = []

    for tok in raw_operands:
        regs = extract_registers(tok)
        operands.append(tok)
        if regs:
            regs_by_operand.append(regs)

    return Instruction(
        opcode=opcode,
        operands=operands,
        regs_by_operand=regs_by_operand,
        loc=loc,
        raw_line=line,
        bb=bb,
    )


def extract_label(line):
    """
    Extract the label at the start of the line, before the first ':'.
    Returns None if no label is present.
    """
    line = line.strip()
    if ':' not in line:
        return None
    # Split at the first colon
    label = line.split(':', 1)[0]
    return label.strip()


def parse_asm(text):
    program = Program()

    cur_block = None
    cur_loc = None
    files = {}
    in_blocks = False
    ended = False

    for raw in text.splitlines():
        line = raw.rstrip()

        # Program tail (after s_endpgm)
        if ended:
            program.tail_lines.append(line)
            continue

        # Detect program end
        if 's_endpgm' in line:
            if cur_block:
                inst = parse_instruction(line, cur_loc, cur_block)
                if inst:
                    cur_block.add_inst(inst)
            program.tail_lines.append(line)
            ended = True
            continue

        # Before first basic block → header
        if not in_blocks:
            program.header_lines.append(line)
            if line.strip().startswith('; %bb.') or line.strip().startswith('.LBB'):
                in_blocks = True
                program.header_lines.pop()  # this line is a BB label
            else:
                continue

        # file directive
        if line.strip().startswith('.file'):
            _, fid, _, name = line.split(maxsplit=3)
            files[int(fid)] = name.strip('"')
            continue

        # loc directive
        if line.strip().startswith('.loc'):
            parts = line.split()
            fid = int(parts[1])
            line_no = int(parts[2])
            col = int(parts[3])
            cur_loc = (files.get(fid, "unknown"), line_no, col)
            continue

        # ignore debug labels
        if line.strip().startswith('.Ltmp'):
            continue

        # new basic block
        if line.strip().startswith('; %bb.') or line.strip().startswith('.LBB'):
            label = extract_label(line)
            cur_block = BasicBlock(label)
            program.blocks.append(cur_block)
            continue

        # sched barrier comment
        if 'sched_barrier' in line:
            continue

        # instruction
        inst = parse_instruction(line, cur_loc, cur_block)
        if inst:
            cur_block.add_inst(inst)

    return program


def emit_blocks(blocks):
    out = []
    for bb in blocks:
        out.append(f"{bb.name}:")
        for inst in bb.instructions:
            out.append(inst.emit())
    return out


def emit_program(program):
    out = []
    out.extend(program.header_lines)
    out.extend(emit_blocks(program.blocks))
    out.extend(program.tail_lines)
    return "\n".join(out)


############################
## Start def-use utilities
############################


def flatten_regs(regs):
    """
    regs: Register | Iterable[Register]
    return: set of (kind, id)
    """
    assert isinstance(regs, (Register, list)), type(regs)

    out = set()

    if isinstance(regs, Register):
        regs = [regs]

    for r in regs:
        for rid in r.ids:
            out.add((r.kind, rid))

    return out


def coalesce_regs(flat_regs):
    """
    flat_regs: set of (kind, id)
    returns: list[Register]
    """
    by_kind = defaultdict(list)

    # 1. Group by register kind
    for kind, rid in flat_regs:
        by_kind[kind].append(rid)

    regs = []

    # 2. For each kind, sort and coalesce contiguous ids
    for kind, ids in by_kind.items():
        ids = sorted(ids)

        start = ids[0]
        prev = ids[0]

        for cur in ids[1:]:
            if cur == prev + 1:
                prev = cur
                continue

            # end of contiguous range
            regs.append(Register(kind, list(range(start, prev + 1))))
            start = prev = cur

        # last range
        regs.append(Register(kind, list(range(start, prev + 1))))

    return regs


def pick_and_remove_contiguous_regs(reg_pool: Set[Tuple[str, int]], x: int,
                                    myKind: str = None) -> Optional[Set[Tuple[str, int]]]:
    if x <= 0:
        raise ValueError("x must be positive")

    # Group register ids by kind
    by_kind = defaultdict(list)
    for kind, rid in reg_pool:
        by_kind[kind].append(rid)

    for kind, ids in by_kind.items():
        if myKind and (kind != myKind):
            continue
        ids = sorted(ids)
        id_set = set(ids)

        # Try every possible contiguous window
        for start in ids:
            # Alignment constraint
            if x > 1 and start % 2 != 0:
                continue

            # Check contiguous existence
            window = list(range(start, start + x))
            if all(i in id_set for i in window):
                chosen = {(kind, i) for i in window}

                # Remove from original pool (in place)
                reg_pool.difference_update(chosen)

                return chosen

    return None


def eliminate_save_restore(bb):
    """
    Eliminate save-modify-restore patterns:
        v_mov tmp, orig         ; save
        <op>  orig, tmp, ...    ; modify (overwrites orig, reads tmp)
        ...                     ; uses of orig
        v_mov orig, tmp         ; restore

    Transform to:
        <op>  tmp, orig, ...    ; write to tmp instead
        ...                     ; replace uses of orig with tmp
        ; save and restore movs are dead
    """
    changed = True
    while changed:
        changed = False
        for i, save_mov in enumerate(bb.instructions):
            if save_mov.mark_dead:
                continue
            # Step 1: Find a v_mov tmp, orig (save instruction)
            if not save_mov.opcode.startswith('v_mov'):
                continue
            tmp_reg = save_mov.get_dst_regs()
            src_regs = save_mov.get_src_regs()
            if not tmp_reg or not src_regs:
                continue
            orig_reg = src_regs[0]
            if tmp_reg.kind != orig_reg.kind:
                continue
            if tmp_reg.size != 1 or orig_reg.size != 1:
                continue

            # Step 2: Find the next instruction that defines orig and uses tmp
            orig_flat = flatten_regs(orig_reg)
            tmp_flat = flatten_regs(tmp_reg)
            modify_inst = None
            modify_idx = None
            for j in range(i + 1, len(bb.instructions)):
                candidate = bb.instructions[j]
                if candidate.mark_dead:
                    continue
                if (orig_flat & candidate.defs) and (tmp_flat & candidate.uses):
                    modify_inst = candidate
                    modify_idx = j
                    break
                # If something else defines tmp or orig before we find the modify, abort
                if (tmp_flat & candidate.defs) or (orig_flat & candidate.defs):
                    break

            if modify_inst is None:
                continue

            # Step 3: Find the restore mov: v_mov orig, tmp
            restore_mov = None
            restore_idx = None
            for j in range(modify_idx + 1, len(bb.instructions)):
                candidate = bb.instructions[j]
                if candidate.mark_dead:
                    continue
                if (candidate.opcode.startswith('v_mov') and candidate.get_dst_regs() == orig_reg
                        and candidate.get_src_regs() and candidate.get_src_regs()[0] == tmp_reg):
                    restore_mov = candidate
                    restore_idx = j
                    break
                # If tmp is redefined before we find restore, abort
                if tmp_flat & candidate.defs:
                    break

            if restore_mov is None:
                continue

            # Step 4: Verify tmp has no other uses between save and restore
            #         (besides the modify instruction and the restore mov)
            other_use = False
            for j in range(i + 1, restore_idx):
                candidate = bb.instructions[j]
                if candidate.mark_dead or candidate is modify_inst:
                    continue
                if tmp_flat & candidate.uses:
                    other_use = True
                    break
            if other_use:
                continue

            # All conditions met — apply transformation
            logging.debug(f"eliminate_save_restore: save={save_mov.emit()}, "
                          f"modify={modify_inst.emit()}, restore={restore_mov.emit()}")

            # 4a. In modify_inst, swap dst from orig to tmp, and replace use of tmp with orig
            modify_inst.replace_dst(tmp_reg)
            modify_inst.replace_use_reg(tmp_reg, orig_reg)

            # 4b. Replace uses of orig with tmp between modify and restore
            for j in range(modify_idx + 1, restore_idx):
                candidate = bb.instructions[j]
                if candidate.mark_dead:
                    continue
                if orig_flat & candidate.uses:
                    candidate.replace_use_reg(orig_reg, tmp_reg)

            # 4c. Mark save and restore as dead
            save_mov.mark_dead = True
            restore_mov.mark_dead = True

            changed = True
            break  # restart scan since indices may have shifted


def optimize_buffer_load_m0(bb):
    logging.debug("========== optimize buffer load m0 ==========")

    i = 0
    end = len(bb.instructions)
    while i < end:
        inst = bb.instructions[i]
        if "buffer_load" in inst.opcode:
            idx = i
            ## pattern 1: s_mov_b32 m0 --> s_nop 0 --> buffer_load --> mfma
            ## pattern 2: s_mov_b32 m0 --> buffer_load --> mfma
            ## swap buffer_load and mfma
            mfma = bb.instructions[idx + 1]
            if not mfma.is_mfma():
                i += 1
                continue
            bb.instructions[idx], bb.instructions[idx + 1] = bb.instructions[idx + 1], bb.instructions[idx]
            ## remove s_nop
            if 'nop' in bb.instructions[idx - 1].opcode:
                bb.instructions[idx - 1].mark_dead = True

            i += 2
        else:
            i += 1

    logging.debug("========== optimize buffer load m0 done =====")


def optimize_nops(bb):
    logging.debug("========== optimize no-ops ==========")

    for idx, inst in enumerate(bb.instructions):
        if 'nop' not in inst.opcode:
            continue
        ## The only allowed nop is the one between a set-m0 and a buffer_load;
        ## everything else is dead.
        if idx == 0 or inst is bb.instructions[-1]:
            inst.mark_dead = True
            continue
        prev_dst = bb.instructions[idx - 1].get_dst_regs()
        suc = bb.instructions[idx + 1]
        if prev_dst is None or prev_dst.kind != 'm' or 'buffer_load' not in suc.opcode:
            inst.mark_dead = True

    bb.cleanup_bb()

    logging.debug("========== optimize no-ops done =====")


def find_loop_invariants(bb: BasicBlock):
    invariant_regs = set()
    invariant_insts = set()

    # Collect defs inside loop
    defs_in_loop = {}
    for inst in bb.instructions:
        for r in inst.defs:
            defs_in_loop.setdefault(r, set()).add(inst)

    # Step 1: live-in registers
    for inst in bb.instructions:
        for r in inst.uses:
            if r not in defs_in_loop:
                invariant_regs.add(r)

    logging.debug(f"defs_in_loop: {coalesce_regs(set(defs_in_loop))}")
    logging.debug(f"invariant regs: {coalesce_regs(invariant_regs)}")

    changed = True
    while changed:
        changed = False
        for inst in bb.instructions:
            if inst in invariant_insts:
                continue
            if not inst.is_pure():
                continue
            if not inst.regs_by_operand:
                continue
            dst = inst.get_dst_regs()
            if dst and dst.kind == 'm':
                continue
            if inst.mark_dead:
                continue
            if 'cndmask' in inst.opcode:
                ## Here we simply assume cndmask is not loop invariant,
                ## neither is the dst reg
                invariant_regs -= flatten_regs(inst.get_dst_regs())
                continue

            if all(r in invariant_regs for r in inst.uses):
                if 'scratch_load' in inst.opcode:
                    read_from = flatten_regs(inst.get_copy_src())
                    if read_from & bb.write_to:
                        invariant_regs -= read_from
                    else:
                        invariant_insts.add(inst)
                        invariant_regs |= read_from
                        logging.debug(f"Adding {inst.emit()}, which defines {inst.get_dst_regs()}")
                    continue
                invariant_insts.add(inst)
                logging.debug(f"Adding {inst.emit()}, which uses {coalesce_regs(inst.uses)}")
                for r in inst.defs:
                    if r not in invariant_regs:
                        invariant_regs.add(r)
                        changed = True
            else:
                ## This instruction is not loop invariant, so
                ## what it defines needs to be removed from the
                ## invariant_reg set
                for r in inst.defs:
                    invariant_regs.discard(r)

    return invariant_insts, invariant_regs


def can_hoist(inst, bb, invariant_regs):
    # Must be pure
    if not inst.is_pure():
        return False

    # Never hoist an mfma out of the loop (its accumulator is loop-carried, so it
    # would not be flagged invariant anyway — this is a cheap belt-and-suspenders).
    if inst.is_mfma():
        return False

    # Each dest reg must:
    # 1. Have exactly one reaching def inside loop (itself)
    reg = inst.get_dst_regs()
    assert reg
    reaching = bb.get_reaching_defs(inst, reg, True)
    if len(reaching) != 1 or inst not in reaching:
        return False

    # 2. No redefinition after inst
    reg_flat = flatten_regs(reg)
    for later in bb.instructions:
        if later == inst:
            continue
        # Use the defs set (from compute_def_use) instead of get_dst_regs(),
        # because get_dst_regs() always returns operand[0] which is wrong for
        # store instructions (ds_write, buffer_store) that don't define regs.
        if later.defs & reg_flat:
            redef = True
            logging.debug(f"  reg redefined by {later.emit()}")
            break
    else:
        redef = False
    if redef:
        ## Rename the hoisted inst's output to a free reg and update its users.
        ## Leave the other defs of the same register untouched.
        def_reg = coalesce_regs(inst.defs)[0]
        logging.debug(f"  reg redefined, trying to rename hoisted inst's output {def_reg.emit()}")
        kind, ids = def_reg.kind, def_reg.ids
        num = len(ids)

        ## Dry-run: verify the hoisted inst and all its users can be rewritten.
        if def_reg.emit() not in inst.operands:
            logging.debug(f"  cannot rename {inst.emit()} (register mismatch), cannot hoist")
            return False
        for user in inst.users:
            has_match = any(op == def_reg.emit() for op in user.operands)
            if not has_match:
                logging.debug(f"  cannot rewrite user {user.emit()} (register embedded in wider group), cannot hoist")
                return False

        free_reg = pick_and_remove_contiguous_regs(bb.free_regs, num, kind)
        if not free_reg:
            logging.debug("Not enough free regs")
            return False
        free_reg = coalesce_regs(free_reg)[0]
        logging.debug(f"  found free reg: {free_reg.emit()}")
        ## Rename the hoisted inst's output and update its users
        before = inst.emit()
        inst.replace_users_with(free_reg)
        inst.replace_reg(def_reg, free_reg)
        logging.debug(f"  renamed: {before} -> {inst.emit()}")

    return True


def hoist_loop_invariants(bb: BasicBlock):
    invariant_insts, invariant_regs = find_loop_invariants(bb)

    hoistable = []
    for inst in invariant_insts:
        logging.debug(f"loop invariant: {inst.emit()}")
        if inst.users:
            logging.debug(f"  users: {[u.emit() for u in inst.users]}")
        if can_hoist(inst, bb, invariant_regs):
            hoistable.append(inst)
            if 'scratch_load' in inst.opcode:
                next_inst = bb.next_instruction(inst)
                if 'vmcnt(0)' in next_inst.operands[0]:
                    logging.debug(f"    Also hoist {next_inst.emit()}")
                    hoistable.append(next_inst)
            logging.debug("  can hoist!!")

    if not hoistable:
        return [], bb.instructions

    hoistable.sort(key=lambda i: i.index)

    new_loop = []
    hoisted_set = set(hoistable)

    for inst in bb.instructions:
        if inst not in hoisted_set:
            new_loop.append(inst)

    return hoistable, new_loop


def remove_debug_info_section(asm_text: str) -> str:
    """
    Remove the .debug_info section from an AMDGPU assembly file.
    """
    lines = asm_text.splitlines(keepends=True)
    output = []

    in_debug_section = False

    for line in lines:
        if line.strip().startswith(".section") and ".debug_info" in line:
            in_debug_section = True
            continue

        if in_debug_section:
            if line.strip().startswith(".Ldebug_info_end"):
                in_debug_section = False
            continue

        output.append(line)

    return "".join(output)


def remove_debug_ranges_section(asm_text: str) -> str:
    """
    Remove the .debug_ranges section from an AMDGPU assembly file.
    """
    lines = asm_text.splitlines(keepends=True)
    output = []

    in_debug_section = False

    for line in lines:
        if line.strip().startswith(".section") and ".debug_ranges" in line:
            in_debug_section = True
            continue

        if in_debug_section:
            if line.strip().startswith(".section"):
                in_debug_section = False
            else:
                continue

        output.append(line)

    return "".join(output)


def _max_reg_index(text: str, letter: str) -> int:
    """Highest index of an arch-VGPR (letter='v') or AGPR (letter='a') referenced
    in the assembly, honoring both single regs (v12) and ranges (v[8:15]). Comments
    (after ';') are stripped so a stray note can never inflate the count."""
    mx = -1
    single = re.compile(rf"\b{letter}(\d+)\b")
    ranged = re.compile(rf"\b{letter}\[(\d+):(\d+)\]")
    for raw in text.splitlines():
        line = raw.split(";", 1)[0]
        for m in single.finditer(line):
            mx = max(mx, int(m.group(1)))
        for m in ranged.finditer(line):
            mx = max(mx, int(m.group(1)), int(m.group(2)))
    return mx


def rewrite_next_free_vgpr(text: str) -> str:
    """Recompute the VGPR count directives from the real register usage.

    The peephole passes (notably LICM) can allocate additional VGPRs in the
    prologue, which would leave the .amdhsa_next_free_vgpr / .amdhsa_accum_offset /
    .vgpr_count values emitted for the pre-peephole code stale. Instead of hardcoding
    a worst-case total (e.g. 512), scan the final assembly for the highest arch-VGPR
    and AGPR actually referenced and size the directives to match, reproducing the
    layout the in-tree code generator uses: arch VGPRs first, AGPRs starting at
    accum_offset (the arch count aligned up to a multiple of 4), and total VGPRs =
    accum_offset + agpr_count."""
    num_arch_vgpr = _max_reg_index(text, "v") + 1
    num_agpr = _max_reg_index(text, "a") + 1
    if num_agpr > 0:
        accum_offset = ((num_arch_vgpr + 3) // 4) * 4
        total_vgpr = accum_offset + num_agpr
    else:
        accum_offset = None
        total_vgpr = num_arch_vgpr

    out = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        leading_ws = line[:len(line) - len(stripped)]
        if stripped.startswith(".amdhsa_next_free_vgpr"):
            out.append(f"{leading_ws}.amdhsa_next_free_vgpr {total_vgpr}\n")
        elif stripped.startswith(".amdhsa_accum_offset") and accum_offset is not None:
            out.append(f"{leading_ws}.amdhsa_accum_offset {accum_offset}\n")
        elif stripped.startswith(".vgpr_count:"):
            out.append(f"{leading_ws}.vgpr_count: {total_vgpr}\n")
        else:
            out.append(line)

    return "".join(out)


def licm(program):
    logging.debug("========== LICM ==========")
    loop = program.get_loop()
    hoisted, new_loop = hoist_loop_invariants(loop)
    loop.instructions = new_loop

    logging.debug("Hoisting the following before the loop:")
    prologue = program.get_prologue()
    for inst in hoisted:
        logging.debug(f"{inst.emit()}")
        prologue.add_inst(inst)
    logging.debug("========== LICM done =====")


def _make_inst(text, bb):
    return parse_instruction(text, 0, bb)


def rotate_lgkmcnt(program):
    loop = program.get_loop()
    prologue = program.get_prologue()
    insts = loop.instructions

    ## Step 1: Identify region-start sync points.
    ## Each region in the loop begins before a group of mfma or ds_read.
    ## Before that, there may be:
    ##   a) s_waitcnt lgkmcnt(0) alone (for mfma), followed later by
    ##      s_waitcnt vmcnt(x), lgkmcnt(0) + s_barrier (for ds_read)
    ##   b) s_waitcnt vmcnt(x), lgkmcnt(0) + s_barrier (already merged)
    ## We want to merge case (a) into a single waitcnt+barrier pair.

    ## Find all s_waitcnt with lgkmcnt and s_barrier instructions.
    ## For each waitcnt+barrier pair, look backward for a standalone
    ## s_waitcnt lgkmcnt(0) that precedes it in the same region.
    ## The standalone may have mfma instructions between it and the pair,
    ## but no other waitcnt+barrier pair.
    sync_pairs = []  # list of (waitcnt_inst, barrier_inst)
    i = 0
    while i < len(insts):
        inst = insts[i]
        if 's_waitcnt' not in inst.opcode or not any('lgkmcnt' in op for op in inst.operands):
            i += 1
            continue

        # Check if this waitcnt has a barrier right after
        if i + 1 < len(insts) and 's_barrier' in insts[i + 1].opcode:
            barrier_inst = insts[i + 1]

            # Look backward for a standalone s_waitcnt lgkmcnt(0) in the same region.
            # Stop at any s_barrier (region boundary).
            prev_standalone = None
            for k in range(i - 1, -1, -1):
                if 's_barrier' in insts[k].opcode:
                    break
                if 's_waitcnt' in insts[k].opcode and any('lgkmcnt' in op for op in insts[k].operands):
                    prev_standalone = insts[k]
                    break

            if prev_standalone is not None:
                # Merge: replace the standalone with the waitcnt+barrier pair,
                # and kill the original pair at its later position.
                logging.debug(
                    f"rotate_lgkmcnt: merging [{prev_standalone.emit()}] into [{inst.emit()}] + [{barrier_inst.emit()}]"
                )
                standalone_idx = insts.index(prev_standalone)
                # Insert waitcnt+barrier at the standalone's position
                new_waitcnt = _make_inst(inst.emit(), loop)
                new_barrier = _make_inst(barrier_inst.emit(), loop)
                insts.insert(standalone_idx, new_barrier)
                insts.insert(standalone_idx, new_waitcnt)
                # Kill the standalone and the original pair
                prev_standalone.mark_dead = True
                inst.mark_dead = True
                barrier_inst.mark_dead = True
                sync_pairs.append((new_waitcnt, new_barrier))
                # Adjust i for the 2 inserted instructions + skip the pair
                i += 4
            else:
                sync_pairs.append((inst, barrier_inst))
                i += 2
        else:
            i += 1

    if not sync_pairs:
        return

    ## Step 2: Rotate the first sync pair.
    ## Remove from current position, add to end of prologue and end of loop.
    first_waitcnt, first_barrier = sync_pairs[0]
    waitcnt_text = first_waitcnt.emit()
    barrier_text = first_barrier.emit()
    logging.debug(f"rotate_lgkmcnt: rotating [{waitcnt_text}] + [{barrier_text}]")

    # Mark originals as dead
    first_waitcnt.mark_dead = True
    first_barrier.mark_dead = True

    # Add to end of prologue
    prologue.add_inst(_make_inst(waitcnt_text, prologue))
    prologue.add_inst(_make_inst(barrier_text, prologue))

    # Add before the s_cbranch at end of loop
    cbranch_idx = len(insts) - 1
    while cbranch_idx >= 0 and not insts[cbranch_idx].is_control():
        cbranch_idx -= 1
    if cbranch_idx >= 0:
        logging.debug(f"rotate_lgkmcnt: inserting before [{insts[cbranch_idx].emit()}] at idx {cbranch_idx}")
        insts.insert(cbranch_idx, _make_inst(barrier_text, loop))
        insts.insert(cbranch_idx, _make_inst(waitcnt_text, loop))

    # Clean up dead instructions
    loop.cleanup_bb()


def separate_waitcnt_and_barrier(loop):
    for inst in loop.instructions:
        if 's_barrier' in inst.opcode:
            ## pattern
            ## mfma --> waitcnt --> barrier
            ## change to
            ## waitcnt --> mfma --> barrier
            idx = loop.instructions.index(inst)
            if idx < 2:
                continue
            maybe_mfma = loop.instructions[idx - 2]
            if 'mfma' not in maybe_mfma.opcode:
                continue
            maybe_waitcnt = loop.instructions[idx - 1]
            if 'waitcnt' not in maybe_waitcnt.opcode:
                continue
            loop.swap_inst(idx - 2, idx - 1)


def schedule_window(window):
    mfmas = [i for i in window if i.is_mfma()]
    nonmfmas = [i for i in window if not i.is_mfma() and not i.is_control()]
    barriers = [i for i in window if i.is_control()]

    if len(nonmfmas) <= 5 and window[0].is_mfma():
        return window[:]  # unchanged

    logging.debug("Found window:")
    for inst in window:
        logging.debug(f"  {inst.emit()}")

    out = []
    mi = ni = 0

    while mi < len(mfmas) or ni < len(nonmfmas):
        if mi < len(mfmas):
            out.append(mfmas[mi])
            mi += 1
        for _ in range(5):
            if ni < len(nonmfmas):
                out.append(nonmfmas[ni])
                ni += 1

        if mi >= len(mfmas):
            out.extend(nonmfmas[ni:])
            break
        if ni >= len(nonmfmas):
            out.extend(mfmas[mi:])
            break

    # Always put one MFMA before barrier if possible
    if barriers:
        if out and not out[-1].is_mfma() and mfmas:
            out.append(mfmas[-1])
        out.extend(barriers)

    return out


def optimize_mfma_density(block):
    i = 0
    n = len(block)
    out = []

    while i < n:
        # ---- CASE A: start of block ----
        if i == 0:
            j = i
            while j < n and not block[j].is_mfma():
                j += 1
            if j < n:  # found mfma
                k = j
                while k < n and block[k].is_mfma():
                    k += 1
                window_end = k - 1
                window = block[i:window_end + 1]
                rewritten = schedule_window(window)
                out.extend(rewritten)
                i = window_end + 1
                continue

        # ---- CASE B: non-mfma -> mfma transition ----
        if i > 0 and not block[i - 1].is_mfma() and block[i].is_mfma():
            start = i
            j = i
            while j < n and block[j].is_mfma():
                j += 1
            k = j
            while k < n and not block[k].is_mfma() and not block[k].is_control():
                k += 1
            window_end = k - 1
            window = block[start:window_end + 1]
            rewritten = schedule_window(window)
            out.extend(rewritten)
            i = window_end + 1
            continue

        # ---- Default: copy ----
        out.append(block[i])
        i += 1

    return out


def amdgcn_as(text, verbose=False):

    setup_logging(debug=verbose)

    program = parse_asm(text)
    indent = 0

    program.update_inst_index()
    program.process_blocks(indent)

    loop = program.get_loop()
    program.update_free_regs(indent)

    # Step 3: hoist loop-invariant address math out of the loop into the prologue.
    dbg("## Step 3: licm", indent)
    licm(program)

    program.update_free_regs(indent)

    # Step 6: drop save/restore mov pairs and redundant nops, and sink an mfma
    # between the m0 write and the buffer_load that consumes it.
    dbg("## Step 6: eliminate save-restore, optimize nops, buffer_load m0", indent)
    eliminate_save_restore(loop)
    optimize_nops(loop)
    optimize_buffer_load_m0(loop)

    program.process_blocks(indent)
    program.update_free_regs(indent)

    loop.cleanup_bb()

    # Step 8: loop-carried scheduling tweaks (rotate the lgkmcnt/barrier sync pair,
    # split merged waitcnt+barrier, tighten mfma packing).
    dbg("## Step 8: loophole optimizations", indent)
    rotate_lgkmcnt(program)
    separate_waitcnt_and_barrier(loop)

    loop.instructions = optimize_mfma_density(loop.instructions)

    program.process_blocks(indent)
    program.update_free_regs(indent)

    loop.cleanup_bb()
    emitted_text = emit_program(program)
    emitted_text = remove_debug_info_section(emitted_text)
    emitted_text = remove_debug_ranges_section(emitted_text)
    emitted_text = rewrite_next_free_vgpr(emitted_text)
    setup_logging(debug=False)
    return emitted_text


def main():
    parser = argparse.ArgumentParser(description="AMDGPU assembly optimizer")
    parser.add_argument("input", help="Input AMDGCN assembly file")
    parser.add_argument("output", help="Output AMDGCN assembly file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")

    with open(args.input, "r") as f:
        text = f.read()

    emitted_text = amdgcn_as(text)

    with open(args.output, "w") as f:
        f.write(emitted_text)

    logging.debug("Emitted program written to %s", args.output)


if __name__ == "__main__":
    main()
