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

"""Out-of-tree amdgcnas peephole, delivered via Triton's stages-inspection hook.

The amdgcnas post-assembly peephole (LICM + MFMA/scalar interleave) is a pure
Python transform on the final assembly text: `amdgcn_as(text) -> text` in
amdgcnas_ext.py (reduced from Triton's python/triton/tools/amdgcnas.py to the
LICM, save/restore, and loop-scheduling peepholes; the VGPR-count directives are
recomputed from the kernel's real register usage rather than hardcoded).
Because it operates on the `amdgcn` stage's string output, it needs no .so, no
LLVM symbols, and no Triton rebuild — just wrap `stages["amdgcn"]`.

Enable from bench.py by setting knobs.runtime.add_stages_inspection_hook to
inspect_stages_hook when TRITON_AMDGCNAS_PLUGIN is set.

Note: this is the *peephole* half of amdgcnas only. The register-allocation
hints are handled separately — amdgpu-agpr-alloc via the kernel's llvm_fn_attrs
option, and amdgpu-mfma-vgpr-form via TRITON_ENABLE_AMDGPU_RA_HINTS (in llvm.cc).
"""
import hashlib
import os
import pathlib

_HERE = pathlib.Path(__file__).parent


def get_key():
    # Fold both this hook and the peephole module into the cache key so any edit
    # re-compiles kernels (peephole-processed vs not must not collide).
    return (_HERE / "amdgcnas_plugin.py").read_text() + (_HERE / "amdgcnas_ext.py").read_text()


def get_hash():
    return hashlib.sha256(get_key().encode("utf-8")).hexdigest()


def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    # No-arg early return: cache key/hash only.
    if all(a is None for a in (stages, options, language, capability)):
        return get_key(), get_hash()
    if stages is None or "amdgcn" not in stages:
        return get_key(), get_hash()

    import amdgcnas_ext

    orig_make_amdgcn = self.make_amdgcn
    verbose = os.environ.get("TRITON_AMDGCNAS_PLUGIN", "1") == "2"

    def make_amdgcn_wrapper(src, metadata):
        # In-tree codegen first (with the RA hints already applied in make_llir /
        # translate_to_asm), then the out-of-tree peephole on the assembly text.
        asm = orig_make_amdgcn(src, metadata, options)
        # The peephole is a pure optimization: a parse/transform failure must never
        # turn a correct kernel into a hard compile error. Fall back to the
        # un-peepholed (correct) assembly on any failure.
        try:
            return amdgcnas_ext.amdgcn_as(asm, verbose)
        except Exception as e:
            import warnings
            warnings.warn(f"amdgcnas peephole skipped, using un-optimized assembly: {e!r}")
            return asm

    stages["amdgcn"] = make_amdgcn_wrapper
    return get_key(), get_hash()
