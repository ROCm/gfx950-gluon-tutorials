# Performance Philosophy

**High-performance Gluon kernels are a co-design between kernel author and compiler.** This page explains what that means, why it produces a different split of responsibilities from traditional GPU programming, and where `llirSched`, `force-agpr`, and `amdgcnas` fit in.

## 1. Traditional compilation: discovery and heuristics

In a traditional flow (C/C++ → LLVM IR → assembly), the compiler is given thread-level source that expresses *what* to compute. It does not know which operations are independent, which loads can overlap which stores, or how registers should be assigned. It *discovers* these facts from the IR.

Both of the hardest backend problems — instruction scheduling and register allocation — are consequences of this discovery model:

- **Instruction scheduling is NP-hard.** Optimal scheduling under resource and register-pressure constraints is NP-complete, so production compilers rely on heuristics (LLVM's `misched`, `post-misched`). The heuristics are conservative because the compiler does not know the programming model the IR came from, and a wrong reordering breaks correctness.
- **Register allocation is graph coloring.** The compiler builds an interference graph from discovered live ranges and colors it. Heuristic again, for the same reason.

Neither problem is hard because the hardware is hard. They are hard because the compiler is given weak input — thread-level code without structural guarantees — and has to recover structure by analysis.

## 2. Block-level programming: dependencies and registers by construction

Gluon is a **block-level** programming model. Kernels operate on tiles and express pipelines in terms of `DOT`, `local_load`, `async_copy`, and related block-level ops. Layouts are explicit. The kernel author designs, at block level, the things a traditional compiler tries to recover from thread-level IR:

- **Dependencies are a design decision.** When a Gluon author writes a 3-stage pipeline where `DOT`, `local_load`, and `buffer_load` are independent within an iteration, that independence is a *structural property of the kernel*, not a fact to be recovered. Downstream, `mfma`, `ds_read`, and `buffer_load_to_lds` inherit that independence and can be interleaved freely based on throughput — not on dependency analysis.
- **Register usage has a closed form.** At block level, register requirements are arithmetic: `(M × N × elemType × sharing_factor) / (num_warps × waveSize)` per tile. The kernel author evaluates the formula up front, budgets registers against the SIMD's 512 VGPRs, and slices along M or N if the budget does not fit (see [v7_sliceN](../kernels/gemm/intra_wave/a16w16/v7_sliceN/README.md)). Allocation is not graph coloring at this level — it is bookkeeping.

> [!IMPORTANT]
> The methodological shift: **what used to be compiler problems become kernel design problems.** And kernel design problems are tractable — the author has full block-level visibility and can evaluate register formulas, pipeline depths, and dependency chains by hand.

## 3. What is left for the compiler

Moving dependencies and register budgets to the kernel level does not eliminate the compiler — it narrows the *discovery-driven* part of its job. The block-level kernel still has to be lowered to thread-level instructions without breaking the invariants the author engineered, and that lowering is what Gluon distinctively asks of a compiler. Everything else a compiler normally does — instruction selection, scalar register management, ABI handling, address arithmetic — continues unchanged; what's different is that scheduling and register allocation are no longer hard problems it has to solve alone.

The narrowed responsibilities are:

- **Interleaving, not scheduling.** Once independence is guaranteed, the compiler's job is to interleave instructions according to the hardware throughput model (e.g., 16 cycles between `ds_read_b128` issues, 64 cycles between `buffer_load` issues, appropriate MFMAs in between). This is O(n) in the number of instructions, not NP-hard. A traditional scheduler's dependency-analysis machinery is unnecessary here and, in practice, gets in the way — it may reorder or cluster MFMAs, destroying the pipeline the author built.
- **Honoring the register budget.** The author has already proved the block-level budget fits. The compiler allocates accordingly and avoids spills. When it inserts AGPR ↔ VGPR copies or clusters live ranges in ways that blow past the budget, it is failing to honor a design that was already valid on paper.

The compiler is still essential. But the hardest parts of its traditional job — the NP-hard scheduling and graph-coloring allocation — are done before it runs.

## 4. `llirSched`, `force-agpr`, and `amdgcnas`: scaffolding for the new model

Today's LLVM pipeline was designed for the discovery model. Its IR has no place to express "these operations are independent by kernel construction," so its passes cannot exploit that guarantee. On Gluon kernels, `misched` reorders conservatively because it assumes it needs to discover dependencies, and the register allocator treats MFMA accumulators as generic live ranges, inserting `v_accvgpr` copies that break MFMA continuity.

`llirSched`, `force-agpr`, and `amdgcnas` are the minimum tools that honor the block-level contract today. They do not solve hard scheduling or allocation problems — the contract has already made those problems small:

- **`llirSched`** (scheduling) applies the O(n) throughput-model interleaving that block-level independence makes safe, and pins the result with `llvm.amdgcn.sched.barrier(0)` after each memory anchor so LLVM's `misched`/`post-misched` cannot re-cluster it.
- **`force-agpr`** (register allocation) is a pair of LLVM flags — `amdgpu-agpr-alloc=256` and `amdgpu-mfma-vgpr-form=false`, both driven by the single `TRITON_FORCE_MFMA_AGPR` knob — that steer allocation to keep MFMA accumulators in AGPRs. Measured across FP16, BF8, and MXFP4 GEMM, this alone closes 75–85% of the MFMA-efficiency gap between `llir` and the full stack, landing within 1–2% TFLOPS. It maps cleanly to an upstream change: teach LLVM's allocator to recognize the Gluon contract and apply this policy natively. It is not free — forcing *all* MFMA accumulators into AGPRs maximizes `v_accvgpr_read` copies in the epilogue, because `v_cvt` (used to downcast FP32 accumulators to the output dtype) requires VGPR inputs; the tradeoff pays off only for compute-bound kernels with large K, where the epilogue is a small share of runtime. This is why the `amdgpu-mfma-vgpr-form=0` half is a temporary blunt instrument: LLVM's upcoming `RewriteMFMAFormStage` pass will choose AGPR vs. VGPR form per MFMA by register pressure, after which that flag can be dropped.
- **`amdgcnas`** (post-assembly peephole) does no scheduling or allocation. It is post-assembly LICM and an MFMA–SALU peephole on the generated AMDGCN text: it hoists loop-invariant LDS address arithmetic and interleaves MFMA with scalar instructions (`s_waitcnt`, `s_barrier`, scalar address computation) that `llirSched` cannot reach — those instructions are inserted during MIR-level codegen, after LLIR lowering. Its contribution is kernel-dependent: +2pp MFMA efficiency on FP16 and BF8 GEMM, but +6pp on MXFP4 where the scale pipeline creates denser SALU activity. The natural upstream home is a MachineInstr-level backend pass; that work is still ahead of us.

None of the three is a general-purpose replacement for an LLVM pass. They are **prototypes of what the remaining compiler work looks like once the kernel author has done the block-level design.** On Gluon-shaped kernels they recover the MFMA efficiency the upstream LLVM flow loses; on arbitrary C-like code they would not make sense.

See [kernels/gemm/README.md §2.1](../kernels/gemm/intra_wave/README.md#21-triton-build-and-the-out-of-tree-plugins) for the mechanical details of each pass.

## 5. Collaboration with LLVM

The goal is not to keep `llirSched`, `force-agpr`, and `amdgcnas` outside the standard Triton/LLVM flow forever. The goal is to fold their ideas into the LLVM backend in three phases, smallest-lift first:

1. **`llirSched` → an LLVM backend scheduling pass**, gated on backend and kernel shape. This retires most of the friction: users on stock Triton + LLVM reach the O(n)-interleaving regime without loading a plugin.
2. **`force-agpr` → LLVM's AMDGPU register allocator.** The flags are small, local changes; the challenge is making the policy *selective* — the `RewriteMFMAFormStage` pass, which chooses AGPR vs. VGPR form per MFMA by register pressure so it need not fall back to the blunt all-AGPR form for kernels where the epilogue is a larger share of runtime (see [a16w16 v7 §4.3](../kernels/gemm/intra_wave/a16w16/v7_sliceN/README.md#43-register-allocation-workaround-force-agpr)).
3. **`amdgcnas` → an LLVM AMDGPU MachineInstr-level pass.** The biggest engineering lift and the smallest measured impact on FP16/BF8 (~2pp MFMA efficiency); may remain a prototype indefinitely.

This work is in progress in collaboration with LLVM engineers. When phases 1 and 2 land, upstream Triton + stock LLVM will produce most of what the three components produce today on a pinned build, and this tutorial's out-of-tree plugin dependency can retire.

The lasting contribution is not the tools. It is the **design split**:

- **Kernel author (at block level):** dependency engineering, pipeline stages, register budgeting, slicing, layout choice.
- **Compiler (bridge to thread level):** faithful lowering, throughput-model interleaving, budget-honoring allocation.

Traditional compilers are general-purpose because they receive general-purpose input. Gluon gives the compiler a stronger contract, which lets the compiler be simpler — and the kernel author more precise.
