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

"""Bind a Triton/Gluon specialization once for low-overhead repeated launch.

The ordinary JITFunction call path binds Python arguments and looks up the
specialization on every dispatch.  That work is useful in applications where
arguments or launch options change, but it needlessly increases the gaps
between otherwise identical dispatches in a profiler timing loop.

``PreparedKernel.create`` compiles with the public ``warmup`` interface (which
does not launch), binds the complete kernel argument list for each rotating
tensor set, and caches launch metadata.  Calling the resulting object enters
the compiled launch stub directly.  No Triton compiler changes are required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _canonical_grid(grid: Sequence[int]) -> tuple[int, int, int]:
    grid = tuple(grid)
    if not 1 <= len(grid) <= 3:
        raise ValueError(f"grid must have one to three dimensions, got {grid}")
    if any(int(value) <= 0 for value in grid):
        raise ValueError(f"grid dimensions must be positive, got {grid}")
    return tuple(int(value) for value in grid) + (1,) * (3 - len(grid))


class PreparedKernel:
    """A compiled kernel with pre-bound arguments for one or more tensor sets."""

    def __init__(self, compiled_kernel, grid, stream, argument_sets):
        from triton import knobs

        self.compiled_kernel = compiled_kernel
        self.metadata_grid = tuple(grid)
        self.grid = _canonical_grid(grid)
        self.stream = stream
        self.argument_sets = tuple(tuple(args) for args in argument_sets)
        if not self.argument_sets:
            raise ValueError("at least one kernel argument set is required")

        # Resolve the lazy module/function handles before reading ``function``.
        self._run = compiled_kernel.run
        self._enter_hook = knobs.runtime.launch_enter_hook
        self._exit_hook = knobs.runtime.launch_exit_hook
        self._launch_metadata = tuple(
            compiled_kernel.launch_metadata(self.metadata_grid, stream, *args)
            for args in self.argument_sets
        )

    @classmethod
    def create(
        cls,
        jit_kernel,
        grid: Sequence[int],
        runtime_argument_sets: Sequence[Sequence[Any]],
        *,
        constexprs: Mapping[str, Any] | None = None,
        compiler_options: Mapping[str, Any] | None = None,
    ) -> "PreparedKernel":
        """Compile and bind ``jit_kernel`` without executing a GPU dispatch.

        ``runtime_argument_sets`` contains positional arguments for each
        rotating tensor set.  ``constexprs`` contains named arguments from the
        JIT kernel signature.  Backend launch options such as ``num_warps`` and
        ``llvm_fn_attrs`` belong in ``compiler_options``.
        """
        from triton.runtime import driver

        runtime_argument_sets = tuple(tuple(args) for args in runtime_argument_sets)
        if not runtime_argument_sets:
            raise ValueError("at least one runtime argument set is required")
        constexprs = dict(constexprs or {})
        compiler_options = dict(compiler_options or {})

        bound_argument_sets = []
        for runtime_args in runtime_argument_sets:
            bound = jit_kernel.signature.bind(*runtime_args, **constexprs)
            bound.apply_defaults()
            bound_argument_sets.append(tuple(bound.arguments.values()))

        compiled_kernel = jit_kernel.warmup(
            *runtime_argument_sets[0],
            grid=tuple(grid),
            **constexprs,
            **compiler_options,
        )
        if compiled_kernel is None:
            raise RuntimeError("kernel warmup did not return a compiled specialization")

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        return cls(compiled_kernel, grid, stream, bound_argument_sets)

    @property
    def slots(self) -> int:
        return len(self.argument_sets)

    def __call__(self, slot: int = 0) -> None:
        slot %= self.slots
        grid_x, grid_y, grid_z = self.grid
        self._run(
            grid_x,
            grid_y,
            grid_z,
            self.stream,
            self.compiled_kernel.function,
            self.compiled_kernel.packed_metadata,
            self._launch_metadata[slot],
            self._enter_hook,
            self._exit_hook,
            *self.argument_sets[slot],
        )
