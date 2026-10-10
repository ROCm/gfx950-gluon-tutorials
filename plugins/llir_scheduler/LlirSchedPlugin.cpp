// ############################################################################
//  MIT License
//
//  Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.
// ############################################################################

// The gfx950 LLIR scheduler as an out-of-tree LLVM pass plugin: load it with
// LLVM_PASS_PLUGIN_PATH and it runs at the OptimizerLast extension point of
// make_llir's O3 pipeline.
//
// Since gfx950-tutorial-v3.0 the plugin carries only the MFMA <-> VALU
// co-execution model, for warp-pipelined attention kernels: every vector op
// has to land in a specific MFMA's shadow, so the pass does not reorder; it
// declares the intended pipeline with sched_group_barrier and lets AMDGPU's
// IGroupLP build it (namespace WP). Regions whose VALU demand exceeds the
// available shadow use a second algorithm, and the memory stages are paced
// with s_nop. The MFMA <-> memory interleave of the GEMM hot loops moved into
// Triton itself (triton-lang/triton#12209, schedule_hint="mfma-schedule");
// the GEMM kernels pass that option and do not load this plugin.
//
// A kernel is scheduled span by span between the llvm.amdgcn.sched.barriers
// that ConvertWarpPipeline emits at its stage boundaries; a kernel without
// such barriers is left untouched. Every block is scheduled transactionally:
// snapshot, declare, verifyFunction, and roll the block back on invalid IR.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tritonamdgpu-llir-schedule"

// Inlined from Triton's TritonAMDGPUToLLVM/MfmaUtility.h so the plugin needs no
// Triton headers.
namespace mlir::triton::AMD {
inline bool isMFMAorWMMA(const llvm::Instruction &I) {
  const auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
  if (!CI || CI->isInlineAsm())
    return false;
  const llvm::Function *Callee = CI->getCalledFunction();
  if (!Callee || !Callee->isIntrinsic())
    return false;
  llvm::StringRef Name = Callee->getName();
  return Name.contains("mfma") || Name.contains("wmma");
}
} // namespace mlir::triton::AMD

namespace {

using namespace llvm;
using mlir::triton::AMD::isMFMAorWMMA;

// Classification of an instruction for scheduling purposes.
enum class SchedKind { MFMA, GR, LR, LW, Other };

// LDS resides in address space 3 on AMDGPU.
constexpr unsigned kLDSAddressSpace = 3;
constexpr unsigned kGlobalAddressSpace = 1;

// Stateless helpers shared by region classification and the cost model.
namespace Utils {
bool isHoistTransparentInst(const Instruction &I) {
  return isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I);
}

bool isSinkTransparentInst(const Instruction &I) {
  return isa<ExtractElementInst>(I);
}

SchedKind classifySchedInst(Instruction &I) {
  if (isMFMAorWMMA(I))
    return SchedKind::MFMA;

  if (auto *CI = dyn_cast<CallInst>(&I)) {
    if (Function *F = CI->getCalledFunction()) {
      if (F->isIntrinsic()) {
        StringRef Name = F->getName();
        // GR: buffer.load (into regs), buffer.load.lds / .async.lds,
        //     raw.ptr.buffer.store (gmem store from regs), and the global.*
        //     memory intrinsics (global.load.lds, global.load.async.lds)
        if (Name.contains("buffer.load") ||
            Name.contains("raw.ptr.buffer.store") ||
            Name.contains("global.load") || Name.contains("global.store"))
          return SchedKind::GR;
        // LR: ds_read (ds.read.*) or ds_load (ds.load.*)
        if (Name.contains("ds.read") || Name.contains("ds.load"))
          return SchedKind::LR;
      }
    }
  }

  // LR: load from LDS (addrspace 3). GR: plain load from global memory
  // (addrspace 1), which is what a kernel gets when buffer ops are off or do
  // not apply to its pointers.
  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    if (LI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LR;
    if (LI->getPointerAddressSpace() == kGlobalAddressSpace)
      return SchedKind::GR;
  }

  // LW: store to LDS (addrspace 3). GR: plain store to global memory.
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    if (SI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LW;
    if (SI->getPointerAddressSpace() == kGlobalAddressSpace)
      return SchedKind::GR;
  }

  return SchedKind::Other;
}

unsigned getMFMACycles(const Instruction &I) {
  if (!isMFMAorWMMA(I))
    return 0;
  const auto *CI = cast<CallInst>(&I);
  const Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return 0;
  StringRef Name = Callee->getName();

  // Cycles here are the MFMA's pass count times 4, the issue-to-issue cost
  // of the matrix unit on gfx950 (SISchedule.td: 16x16xK are 4 passes,
  // 32x32xK 8 passes, whatever the element type).
  //
  // Scaled f8f6f4 MFMAs: the cost depends on the operand formats encoded in
  // cbsz (arg 3) and blgp (arg 4); a value above 1 is a 4- or 6-bit format,
  // which the unit runs at the 4-bit rate.
  if (Name.contains("mfma.scale.f32.16x16x128.f8f6f4")) {
    // both operands 4- or 6-bit -> 16 cycles, otherwise (either f8) -> 32.
    if (auto *CbszC = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
      if (auto *BlgpC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
        return (CbszC->getZExtValue() > 1 && BlgpC->getZExtValue() > 1) ? 16
                                                                        : 32;
    return 32; // Fallback if cbsz/blgp are not constants
  }
  if (Name.contains("mfma.scale.f32.32x32x64.f8f6f4")) {
    // both operands 4- or 6-bit -> 32 cycles, otherwise (either f8) -> 64.
    if (auto *CbszC = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
      if (auto *BlgpC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
        return (CbszC->getZExtValue() > 1 && BlgpC->getZExtValue() > 1) ? 32
                                                                        : 64;
    return 64; // Fallback if cbsz/blgp are not constants
  }

  // Fixed-cost MFMAs.
  static constexpr struct {
    StringRef Name;
    unsigned Cycles;
  } kFixedCycles[] = {
      // gfx950 shapes
      {"mfma.f32.16x16x32.f16", 16},
      {"mfma.f32.16x16x32.bf16", 16},
      {"mfma.i32.16x16x64.i8", 16},
      {"mfma.f32.32x32x16.f16", 32},
      {"mfma.f32.32x32x16.bf16", 32},
      {"mfma.i32.32x32x32.i8", 32},
      // fp8 / bf8 (all four operand combinations, gfx940 encodings)
      {"mfma.f32.16x16x32.fp8", 16},
      {"mfma.f32.16x16x32.bf8", 16},
      {"mfma.f32.32x32x16.fp8", 32},
      {"mfma.f32.32x32x16.bf8", 32},
      // K = 16 / K = 8 fallbacks the lowering picks for small BLOCK_K
      {"mfma.f32.16x16x16f16", 16},
      {"mfma.f32.16x16x16bf16.1k", 16},
      {"mfma.f32.32x32x8f16", 32},
      {"mfma.f32.32x32x8bf16.1k", 32},
  };
  for (const auto &Entry : kFixedCycles)
    if (Name.contains(Entry.Name))
      return Entry.Cycles;

  // Unknown / unmodeled shape: the scheduler bails on this region and leaves
  // it to the default LLVM schedulers (such kernels are not perf-critical).
  return 0;
}
} // namespace Utils

// Roll a block back to a pre-scheduling snapshot: erase the instructions the
// scheduler inserted (the sched_group_barrier / s_nop
// intrinsic calls, all void with no uses) and restore
// the recorded instruction order.
static void restoreBlock(BasicBlock &BB,
                         const SmallVectorImpl<Instruction *> &snapshot) {
  SmallPtrSet<const Instruction *, 32> orig(snapshot.begin(), snapshot.end());
  SmallVector<Instruction *, 8> inserted;
  for (Instruction &I : BB)
    if (!orig.count(&I))
      inserted.push_back(&I);
  for (Instruction *I : inserted) {
    if (!I->use_empty())
      I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    I->eraseFromParent();
  }
  for (size_t i = 1; i < snapshot.size(); ++i)
    snapshot[i]->moveAfter(snapshot[i - 1]);
}

// ===========================================================================
// warp_pipeline (Flash-Attention) region analysis.
//
// Unlike the GEMM path (regions inferred from "MFMA after a memory op"), a
// warp-pipeline kernel already carries its cluster boundaries:
// ConvertWarpPipeline lowers each stage boundary to
// llvm.amdgcn.sched.barrier(i32 0). So here a region is simply the instruction
// span between two sched.barriers, and the compute (MFMA) stages are the
// regions that contain MFMAs.
//
// The co-execution model does not reorder: it declares the intended pipeline
// with sched_group_barrier and lets IGroupLP build it.
// ===========================================================================
// ===========================================================================
// Which scheduling model a region wants
// ===========================================================================
// The GEMM throughput model (mfma + memory, no valu) lives in Triton since
// gfx950-tutorial-v3.0 (schedule_hint="mfma-schedule"); this plugin only
// declares the co-execution model, and classifyRegion() decides which spans
// qualify by what they CONTAIN, not by which kernel they came from:
//
//   mfma + valu, no memory
//       -> CO-EXEC model: fill each mfma's 24-cycle shadow with co-issuable
//       VALU
//          and declare it with sched_group_barrier (this namespace). Typical of
//          inter-wave Flash-Attention, whose DOT clusters are register-only --
//          (the FA dot stages have no LDS traffic).
//
//   mfma + valu + memory
//       -> NOT HANDLED. Intra-wave FA would land here. The two models disagree
//          about what an mfma is for (covering VALU issue vs covering memory
//          latency), so such a region is skipped rather than fed to a model
//          whose assumptions it breaks. Revisit when that kernel exists.
//
//   no mfma
//       -> nothing to schedule around. FA's mem stages are this case (they
//       carry
//          LDS + global traffic but no mfma), which is why the co-exec model
//          never sees them; insertMemRegionNops paces them separately.
//
// Inter-wave GEMM needs no scheduling at all and simply produces no qualifying
// region.
namespace WP {

// The sched.barrier(i32 0) intrinsic marking a warp-pipeline cluster boundary.
static bool isSchedBarrier(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      return F->getName().contains("amdgcn.sched.barrier");
  return false;
}

// A function that already carries sched.barriers (ConvertWarpPipeline's stage
// boundaries, BlockPingpong's clusters, or explicit hints) is scheduled span by
// span between them, each span by the model its contents ask for; the
// throughput model must not move anything across such a barrier. This only
// decides *where
// the region boundaries come from* -- which model each region then gets is
// decided by classifyRegion() below, per region.
static bool hasSchedBarrier(Function &F) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (isSchedBarrier(I))
        return true;
  return false;
}

enum class RegionModel { None, Throughput, CoExec, Mixed };

// Defined below, once the cost model it belongs to is in scope.
static int valuWeight(const Instruction &I);

// What does this region contain? Counts are returned so the caller can log
// them.
static RegionModel classifyRegion(Instruction *Begin, BasicBlock::iterator End,
                                  int &numMfma, int &numValu, int &numMem) {
  numMfma = numValu = numMem = 0;
  for (auto It = Begin->getIterator(); It != End; ++It) {
    Instruction &I = *It;
    if (isMFMAorWMMA(I)) {
      ++numMfma;
      continue;
    }
    // Reuse the throughput model's own classifier so the two paths cannot
    // disagree about what counts as memory: GR = buffer/global, LR/LW = LDS
    // read/write.
    SchedKind K = Utils::classifySchedInst(I);
    if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW) {
      ++numMem;
      continue;
    }
    if (valuWeight(I) > 0)
      ++numValu;
  }
  if (numMfma == 0)
    return RegionModel::None;
  if (numMem > 0)
    return numValu > 0 ? RegionModel::Mixed : RegionModel::Throughput;
  return numValu > 0 ? RegionModel::CoExec : RegionModel::None;
}

static const char *modelName(RegionModel M) {
  switch (M) {
  case RegionModel::None:
    return "no-mfma";
  case RegionModel::Throughput:
    return "mfma+mem -> THROUGHPUT model";
  case RegionModel::CoExec:
    return "mfma+valu -> CO-EXEC model";
  case RegionModel::Mixed:
    return "mfma+valu+mem -> MIXED (not handled)";
  }
  return "?";
}

// Transcendental (v_exp/v_rcp/...): never co-issues with MFMA -- kept separate
// from plain VALU (matches the hardware isNeverCoissue rule).
static bool isTransCompute(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      if (F->isIntrinsic()) {
        StringRef N = F->getName();
        return N.contains("exp2") || N.contains("exp.") || N.contains("log2") ||
               N.contains("sin") || N.contains("cos") || N.contains("sqrt") ||
               N.contains("rcp");
      }
  return false;
}

// Transitive: does I reach any instruction in `targets` through its operands,
// WITHIN the same iteration? We stop at PHI nodes so loop-carried (backedge)
// values are not followed -- a VALU consuming the *previous* iteration's MFMA
// (the warp-pipeline decoupling) is independent for interleaving purposes; only
// an intra-iteration MFMA->VALU path counts as a real dependency.
static bool dependsOnAny(Instruction *I,
                         const SmallPtrSetImpl<const Instruction *> &targets) {
  SmallVector<const Value *, 32> Work(I->op_begin(), I->op_end());
  SmallPtrSet<const Value *, 32> Seen;
  while (!Work.empty()) {
    const Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    if (const auto *DI = dyn_cast<Instruction>(V)) {
      if (targets.count(DI))
        return true;
      if (isa<PHINode>(DI))
        continue; // don't cross loop-carried / cross-iteration edges
      Work.append(DI->op_begin(), DI->op_end());
    }
  }
  return false;
}

// Co-issue weight of a VALU op for the interleave count (0 if not counted):
//   regular unpacked scalar FP = 1;  packed (vector FP) = 2;
//   transcendental (exp/...) = 2;  v_permlane = 2.
// True if I is an fmaximum/fminimum/maxnum/minnum intrinsic call -- the ops the
// AMD backend folds pairwise (max(max(a,b),c)) into v_maximum3/v_minimum3.
static bool isMaxMin(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      if (F->isIntrinsic()) {
        StringRef N = F->getName();
        return N.contains("maximum") || N.contains("minimum") ||
               N.contains("maxnum") || N.contains("minnum");
      }
  return false;
}

// Mark the "inner" max/mins that the backend absorbs into a v_maximum3, so they
// count 0 (they vanish at isel). Replicates isel's greedy bottom-up fold: walk
// the region's max/mins in program order; each one absorbs at most one single-
// use, not-yet-committed max/min operand as its inner. Absorbed -> Folded (0);
// the absorber becomes a v_maximum3 (full weight). Matches the measured count
// (LLIR llvm.maximum -> ASM v_maximum3) exactly on the FA softmax reduction.
static void computeMax3Folds(Instruction *Begin, BasicBlock::iterator End,
                             const SmallPtrSetImpl<const Instruction *> &Region,
                             SmallPtrSetImpl<const Instruction *> &Folded) {
  SmallPtrSet<const Instruction *, 32> Committed;
  for (auto It = Begin->getIterator(); It != End; ++It) {
    if (!isMaxMin(*It))
      continue;
    for (Value *Op : It->operands()) {
      auto *In = dyn_cast<Instruction>(Op);
      if (In && isMaxMin(*In) && Region.count(In) && In->hasOneUse() &&
          !Folded.count(In) && !Committed.count(In)) {
        Folded.insert(In);
        Committed.insert(&*It);
        break;
      }
    }
  }
}

static int valuWeight(const Instruction &I) {
  if (isMFMAorWMMA(I))
    return 0;
  if (isTransCompute(I))
    return 2;
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      if (F->isIntrinsic()) {
        StringRef N = F->getName();
        if (N.contains("permlane"))
          return 2;
        // Count maximum/minimum (the softmax row-max reduction) by DEFAULT.
        // The 2:1 v_maximum3 fold is handled at collection time by
        // computeMax3Folds (inner ops -> weight 0), so with fold-aware
        // weighting the reduction contributes its real issued count (validated:
        // 89 vs the naive 164).
        //
        // Counting them is worth ~1.5%: with weight 0 the reduction is
        // INVISIBLE to the interleave, so its ~16 v_maximum3 pile up *before*
        // the stage's first mfma and no window ever covers them (measured on
        // FAv4: 76 cycles of co-exec-capable VALU stranded at the PV stage head
        // while that stage's own windows sat 92 cycles under-filled; counting
        // moves head 76 -> 0 cyc and fill 292 -> 368 / 384).
        if (N.contains("maxnum") || N.contains("minnum") ||
            N.contains("maximum") || N.contains("minimum") ||
            N.contains("fmuladd") || N.contains("fma."))
          return I.getType()->isVectorTy() ? 2 : 1;
        // llvm.fabs is NOT counted, for the same reason as fneg below: it
        // becomes a source modifier, not an instruction. (No current kernel has
        // one, so this only guards the future.)
      }
  // NOTE: an fmul feeding a single fadd/fsub contracts into one v_pk_fma (->
  // two v_fma), so in principle the pair is 2 issue slots not 4. But zeroing
  // the fmul weight here creates weight-0 runs that make the error-diffusion
  // cluster mfmas (measured a 5-long mfma run, 1035 vs 1043). Left
  // double-counted on purpose -- the denser qk*scale placement it produces is
  // empirically better. A packed convert (fptrunc/fpext of a <2 x T>) lowers to
  // ONE co-issuable v_cvt_pk_f16 (4 cyc) and is NOT scalarized like packed
  // fmul/fadd, so it is a single co-issue unit -> weight 1, not 2. (Counting it
  // 2 under-filled the cvt windows: 4 cvt = weight 8 "full" but only 16 cyc of
  // the 24-cyc window, so the interleave placed 4/window instead of the 6 that
  // fit.) fneg is NOT an instruction on AMDGPU: it folds into its consumer as a
  // source modifier (`v_fma_f32 v0, v0, s44, -v129`). Counting it inflates a
  // declared sched_group_barrier group by an instruction that never exists, and
  // IGroupLP cannot fill that group -- so its pipeline solver gives up and
  // leaves ISel's order for the WHOLE region. Measured on FAv4 with
  // SCALE_ON_Q=0, whose QK stage carries one fneg for fma(qk, qk_scale,
  // -m_new): 7 of 16 groups were placed and the remaining 8 mfma were emitted
  // back-to-back, stranding 12 exp2 plus ~20 valu with no co-exec window (the
  // stage's twin, which had no fneg, scheduled fine).
  if (I.getOpcode() == Instruction::FNeg)
    return 0;
  if (isa<FPTruncInst>(I) || isa<FPExtInst>(I))
    return 1;
  if (I.getType()->isFPOrFPVectorTy() &&
      (isa<BinaryOperator>(I) || isa<UnaryOperator>(I) || isa<SelectInst>(I)))
    return I.getType()->isVectorTy() ? 2 : 1; // packed vector = 2, scalar = 1
  return 0;
}

// Transparent glue that builds an input of one of THIS region's mfmas
// (insertelement/shuffle feeding a region mfma). Region-restricted so we don't
// hoist a prep that actually feeds the *next* region's mfma (which may legally
// use this region's valu).
static bool feedsMFMA(Instruction *I,
                      const SmallPtrSetImpl<const Instruction *> &RegionMfmas) {
  SmallVector<Value *, 8> Work;
  SmallPtrSet<Value *, 16> Seen;
  Work.push_back(I);
  while (!Work.empty()) {
    Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    for (User *U : V->users())
      if (auto *UI = dyn_cast<Instruction>(U)) {
        if (RegionMfmas.count(UI))
          return true;
        if (Utils::isHoistTransparentInst(*UI))
          Work.push_back(UI);
      }
  }
  return false;
}

// extractelement of one of THIS region's mfma results.
static bool
definedByMFMA(Instruction *I,
              const SmallPtrSetImpl<const Instruction *> &RegionMfmas) {
  SmallVector<Value *, 8> Work;
  SmallPtrSet<Value *, 16> Seen;
  Work.push_back(I);
  while (!Work.empty()) {
    Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    if (auto *DI = dyn_cast<Instruction>(V)) {
      if (RegionMfmas.count(DI))
        return true;
      if (Utils::isSinkTransparentInst(*DI))
        for (Value *Op : DI->operands())
          Work.push_back(Op);
    }
  }
  return false;
}

// IGroupLP scheduling-group masks (AMDGPUIGroupLP / SCHED_GROUP_BARRIER).
// TRANS is a class of its own: canAddMI()'s VALU branch excludes
// transcendentals, so v_exp matches ONLY the TRANS mask.
static constexpr uint32_t kSGBMaskVALU = 0x002;
static constexpr uint32_t kSGBMaskMFMA = 0x008;
static constexpr uint32_t kSGBMaskTRANS = 0x400;
// One mfma opens a co-execution window shorter than its matrix occupancy: a
// 32x32x16 (32-cycle) mfma exposes 24 cycles. Do not hardcode that -- an FA
// variant built on 16x16x32 mfma has a 16-cycle occupancy, and giving it
// 24-cycle windows would over-fill every one of them by 3x.
static constexpr int kWindowCycles = 24;  // the validated 32-cycle-mfma case
static constexpr int kMFMANonOverlap = 8; // occupancy - window, from that case

// Co-exec window for a region, derived from the mfmas it actually contains.
// Returns 0 when the shape is unmodelled or mixed in a way we should not guess
// at, in which case the caller leaves the region to the default schedulers --
// the same bail-out `Utils::getMFMACycles` already uses.
static int regionWindowCycles(Instruction *Begin, BasicBlock::iterator End) {
  unsigned MinCycles = 0;
  for (auto It = Begin->getIterator(); It != End; ++It) {
    if (!isMFMAorWMMA(*It))
      continue;
    unsigned C = Utils::getMFMACycles(*It);
    if (C == 0)
      return 0; // unmodelled mfma: do not invent a window for it
    // Mixed shapes: size the window for the SHORTEST mfma, so no window
    // overfills.
    MinCycles = MinCycles ? std::min(MinCycles, C) : C;
  }
  if (MinCycles == 0)
    return 0;
  return std::max<int>(4, (int)MinCycles - kMFMANonOverlap);
}
// Mem-stage head pacing: 2 x `s_nop 7` measured best on both FA kernels.
static constexpr int kDefaultMemNops = 2;
// v_permlane is a cross-lane shuffle: model it as a fat 20-cycle VALU.
static constexpr int kPermlaneCycles = 20;
// One 4-cycle VALU op per co-exec slot: 24 cycles / 4 = 6 slots per mfma.
static constexpr int kSlotCycles = 4;

// A packed f32 op this pass can split into per-element scalar ops. Deliberately
// narrow: fmul / fadd / fsub / fma(muladd) on a <N x float>.
//
//  * fptrunc/fpext are excluded -- a packed convert IS one v_cvt_pk_f16_f32,
//  and
//    splitting it would double the issue count for no gain.
//  * <N x half> is excluded -- v_pk_*_f16 is the natural form of f16 math, not
//  a
//    fusion of two scalar ops.
static bool isSplittablePackedFP(const Instruction &I) {
  auto *VT = dyn_cast<FixedVectorType>(I.getType());
  if (!VT || VT->getNumElements() < 2 || !VT->getElementType()->isFloatTy())
    return false;
  if (isa<BinaryOperator>(I))
    return I.getOpcode() == Instruction::FMul ||
           I.getOpcode() == Instruction::FAdd ||
           I.getOpcode() == Instruction::FSub;
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      if (F->isIntrinsic()) {
        Intrinsic::ID Id = F->getIntrinsicID();
        return Id == Intrinsic::fmuladd || Id == Intrinsic::fma;
      }
  return false;
}

// Replace a packed op with one scalar op per element, appending the new scalar
// ops to Out. Same rewrite as Triton's ScalarizePackedFOps, applied to ONE op
// instead of every packed op in the block -- which is the whole point here:
// only the ops that landed in an mfma co-exec window want to be scalar.
static bool scalarizePackedFP(Instruction *I,
                              SmallVectorImpl<Instruction *> &Out) {
  auto *VT = dyn_cast<FixedVectorType>(I->getType());
  if (!VT)
    return false;
  unsigned N = VT->getNumElements();
  IRBuilder<> B(I);
  Value *Vec = UndefValue::get(VT);
  auto *BO = dyn_cast<BinaryOperator>(I);
  auto *CI = dyn_cast<CallInst>(I);
  for (unsigned e = 0; e < N; ++e) {
    Value *R = nullptr;
    if (BO) {
      Value *A = B.CreateExtractElement(BO->getOperand(0), e);
      Value *C = B.CreateExtractElement(BO->getOperand(1), e);
      R = B.CreateBinOp(BO->getOpcode(), A, C);
    } else if (CI) {
      Value *A = B.CreateExtractElement(CI->getArgOperand(0), e);
      Value *C = B.CreateExtractElement(CI->getArgOperand(1), e);
      Value *D = B.CreateExtractElement(CI->getArgOperand(2), e);
      R = B.CreateIntrinsic(VT->getElementType(),
                            CI->getCalledFunction()->getIntrinsicID(),
                            {A, C, D});
    } else {
      return false;
    }
    if (auto *RI = dyn_cast<Instruction>(R)) {
      RI->copyFastMathFlags(
          I); // keep contraction/reassociation rights identical
      Out.push_back(RI);
    }
    Vec = B.CreateInsertElement(Vec, R, e);
  }
  I->replaceAllUsesWith(Vec);
  I->eraseFromParent();
  return true;
}

// ---------------------------------------------------------------------------
// Over-capacity stages: choose what to hide, and pack what cannot be hidden.
// ---------------------------------------------------------------------------
// declareRegionGroups() below assumes the stage's VALU work FITS in the mfma
// co-exec capacity (24 cyc x M) and spreads it so no group overflows. FAv3
// breaks that assumption: its QK stage carries ~470 cycles of VALU against 384
// cycles of capacity and its PV stage ~490 (measured minGroups 20 and 21
// against 16 mfmas, so the balanced packer's merge loop collapses pairs into
// 48-cycle groups).
//
// When the work does not fit, no schedule hides it and the question changes to
// which ops get a window and what shape the rest take:
//
//   1. WHICH ops get covered. An op that cannot be packed (exp2, the v_maximum3
//      reduction, v_cvt_pk, permlane) gains nothing from being left alone, so
//      it gets first claim on the windows. Whatever capacity is left goes to
//      the packable ops, taken from the END of the stage backwards -- mfmas
//      inserted in reverse program order -- which keeps the uncovered remainder
//      contiguous and leaves it where it already sits: at the head in FAv3's QK
//      (the rescale muls), in the middle in its PV (the qk_scale fmas, between
//      the max3 reduction and the exp2s).
//   2. WHAT SHAPE the rest takes. A covered op should be SCALAR, so it
//   co-issues
//      one 4-cycle slot at a time inside its window. An UNCOVERED op should
//      stay PACKED: nothing hides it either way, and one v_pk_mul_f32 retires
//      two elements in one issue where two v_mul_f32 need two.
//
// This is why FAv3 must NOT set AMDGCN_SCALARIZE_PACKED_FOPS: that pass splits
// every packed op in any block containing an mfma, including the uncovered ones
// this pass deliberately keeps packed.
//
// Weights follow the slot model: 1 mfma = 6 slots, 1 unpacked op = 1 slot, 1
// packed op = 2 slots; exp2 = 2 slots (8 cyc) and permlane = 5 (20 cyc).
static bool declareRegionGroupsOverCap(
    Instruction *Begin, Instruction *End, int syncID, int M, int windowCycles,
    const SmallPtrSetImpl<const Instruction *> &Max3Folded,
    const SmallPtrSetImpl<const Instruction *> &transSet) {
  BasicBlock *BB = Begin->getParent();
  auto ItEnd = End ? End->getIterator() : BB->end();

  struct Op {
    Instruction *I;
    int cyc;
    bool isTrans;
    bool splittable;
    bool covered;
  };
  SmallVector<Op, 64> ops;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It) {
    Instruction &I = *It;
    if (isMFMAorWMMA(I) || Max3Folded.count(&I))
      continue;
    int w = valuWeight(I);
    if (w <= 0)
      continue;
    bool isTrans = transSet.count(&I) != 0;
    bool isPermlane = false;
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (Function *F = CI->getCalledFunction())
        isPermlane = F->getName().contains("permlane");
    int cyc = isPermlane ? kPermlaneCycles : w * kSlotCycles;
    ops.push_back({&I, cyc, isTrans, isSplittablePackedFP(I), false});
  }
  if (ops.empty() || M < 2)
    return false;

  const int Cap = windowCycles * M;
  int Total = 0, Fixed = 0;
  for (const Op &o : ops) {
    Total += o.cyc;
    if (!o.splittable)
      Fixed += o.cyc;
  }
  if (Total <= Cap)
    return false; // fits -- the balanced packer handles it better

  // A window is one mfma's 24-cycle shadow. It may hold SEVERAL declared
  // groups, of different IGroupLP classes: [MFMA 1][VALU 1][TRANS 1] asks for
  // one mfma and then a sub and an exp2 behind it, which is one mfma instead of
  // two. FAv3's PV stage ends in exactly that pair (a lone sub, then a lone
  // exp2) and used to spend two windows on 12 cycles of work.
  struct Chunk {
    bool isTrans;
    int cyc, n;
  };
  struct Slot {
    bool hasMfma; // false = uncovered: declared with no mfma in front of it
    int cyc;
    SmallVector<Chunk, 4> chunks;
  };

  // Decide coverage for a given packable-op budget, then lay the result out
  // into slots. Returns the number of windows the layout needs.
  auto decideAndLayout = [&](int avail, SmallVectorImpl<Slot> &out,
                             int &boughtOut, int &nSplitOut) {
    for (Op &o : ops)
      o.covered = !o.splittable; // non-splittable ops get first claim
    int bought = 0, nSplit = 0;
    for (int i = (int)ops.size() - 1; i >= 0; --i) {
      Op &o = ops[i];
      if (o.splittable && bought + o.cyc <= avail) {
        o.covered = true;
        bought += o.cyc;
        ++nSplit;
      }
    }
    boughtOut = bought;
    nSplitOut = nSplit;

    out.clear();
    int nWin = 0;
    for (const Op &o : ops) {
      // Instruction count as IGroupLP counts it: a covered packed op is about
      // to be scalarized into one op per element; an uncovered one stays a
      // single issue.
      int n = 1;
      if (o.covered && o.splittable)
        if (auto *VT = dyn_cast<FixedVectorType>(o.I->getType()))
          n = VT->getNumElements();
      bool wantMfma = o.covered;
      Slot *S = out.empty() ? nullptr : &out.back();
      // Extend the current slot when it is the same kind of slot and the op
      // still fits the window. Class may differ from the previous chunk -- that
      // is the point -- so only the cycle budget and the covered/uncovered
      // split bound it.
      bool fits = S && S->hasMfma == wantMfma &&
                  (!wantMfma || S->cyc + o.cyc <= windowCycles);
      if (!fits) {
        out.push_back({wantMfma, 0, {}});
        S = &out.back();
        if (wantMfma)
          ++nWin;
      }
      S->cyc += o.cyc;
      if (!S->chunks.empty() && S->chunks.back().isTrans == o.isTrans) {
        S->chunks.back().cyc += o.cyc;
        S->chunks.back().n += n;
      } else {
        S->chunks.push_back({o.isTrans, o.cyc, n});
      }
    }
    return nWin;
  };

  // Iterate: a window freed by tighter packing is capacity the packable ops can
  // still use, so feed it back into the coverage budget. FAv3's PV frees the
  // window its trailing sub+exp2 pair used to waste, which buys three more
  // v_pk_fma.
  int Avail = Cap - Fixed, Bought = 0, nSplit = 0;
  SmallVector<Slot, 40> slots;
  int nWin = decideAndLayout(Avail, slots, Bought, nSplit);
  for (int iter = 0; iter < 4 && nWin < M; ++iter) {
    int grown = Avail + (M - nWin) * windowCycles;
    SmallVector<Slot, 40> trial;
    int tb = 0, ts = 0;
    int tw = decideAndLayout(grown, trial, tb, ts);
    if (tw > M)
      break; // overshot: keep the layout that still fits
    Avail = grown;
    slots = std::move(trial);
    Bought = tb;
    nSplit = ts;
    if (tw == nWin)
      break; // nothing more to win
    nWin = tw;
  }
  // Re-run the decision so ops[].covered matches the layout we kept.
  nWin = decideAndLayout(Avail, slots, Bought, nSplit);

  // More windows than mfmas? Merge the cheapest adjacent pair, so the excess
  // lands in one deliberately over-full window instead of dropping the tail
  // slots -- which would cost the LAST exp2 groups their windows, the ops least
  // able to afford it.
  while (nWin > M) {
    size_t bi = slots.size();
    int best = INT_MAX;
    for (size_t i = 0; i + 1 < slots.size(); ++i)
      if (slots[i].hasMfma && slots[i + 1].hasMfma) {
        int cost = slots[i].cyc + slots[i + 1].cyc;
        if (cost < best) {
          best = cost;
          bi = i;
        }
      }
    if (bi == slots.size())
      break; // every window is separated by an uncovered slot; nothing to merge
    slots[bi].cyc += slots[bi + 1].cyc;
    for (const Chunk &c : slots[bi + 1].chunks)
      if (!slots[bi].chunks.empty() &&
          slots[bi].chunks.back().isTrans == c.isTrans) {
        slots[bi].chunks.back().cyc += c.cyc;
        slots[bi].chunks.back().n += c.n;
      } else {
        slots[bi].chunks.push_back(c);
      }
    slots.erase(slots.begin() + bi + 1);
    --nWin;
  }

  // Scalarize exactly the covered packed ops. The layout above already
  // accounted for the instruction count each one becomes, so nothing needs
  // re-walking after this -- which matters because the rewrite erases the
  // original op.
  int PackedLeft = 0;
  SmallVector<Instruction *, 16> ToSplit;
  for (Op &o : ops) {
    if (o.covered && o.splittable)
      ToSplit.push_back(o.I);
    else if (o.splittable)
      ++PackedLeft;
  }
  for (Instruction *I : ToSplit) {
    SmallVector<Instruction *, 4> New;
    scalarizePackedFP(I, New);
  }

  // Emit. Program order is always satisfiable (it is the order IGroupLP already
  // sees), so unlike the fitting path this needs no dependency-ordered blocks:
  // coverage, not reordering, is the decision here.
  int gBare = M - nWin;
  if (gBare < 0)
    gBare = 0;
  Instruction *IP = End ? End : BB->getTerminator();
  if (!IP)
    return false;
  IRBuilder<> B(IP);
  auto emit = [&](uint32_t mask, int size) {
    B.CreateIntrinsic(Intrinsic::amdgcn_sched_group_barrier,
                      {B.getInt32(mask), B.getInt32(size), B.getInt32(syncID)});
  };
  for (size_t i = 0; i < slots.size(); ++i) {
    const Slot &S = slots[i];
    // Partnerless mfmas go just before the LAST slot, not at the region head:
    // an unfilled window is free, but having it first delays every co-issued
    // op.
    if (i + 1 == slots.size())
      for (int k = 0; k < gBare; ++k)
        emit(kSGBMaskMFMA, 1);
    if (S.hasMfma)
      emit(kSGBMaskMFMA, 1);
    for (const Chunk &c : S.chunks)
      emit(c.isTrans ? kSGBMaskTRANS : kSGBMaskVALU, c.n);
  }

  LLVM_DEBUG({
    dbgs() << "  [sgb-overcap] sync=" << syncID << " M=" << M << " cap=" << Cap
           << " total=" << Total << "cyc fixed=" << Fixed
           << "cyc avail=" << Avail << "cyc bought=" << Bought
           << "cyc  scalarized=" << nSplit << " packed-left=" << PackedLeft
           << "  windows=" << nWin << "/" << M << "  slots:";
    for (const Slot &S : slots) {
      dbgs() << " " << (S.hasMfma ? "[" : "*[");
      bool first = true;
      for (const Chunk &c : S.chunks) {
        dbgs() << (first ? "" : "+") << (c.isTrans ? "T" : "V") << c.n;
        first = false;
      }
      dbgs() << "]" << S.cyc;
    }
    dbgs() << "  (* = no mfma) bare=" << gBare << "\n";
  });
  return true;
}

// Declare the co-exec schedule with sched_group_barrier
// instead of physically reordering the region and pinning it with
// sched_barrier(0).
//
// Why: sched_barrier(0) is only an advisory "do not cross" marker for the
// machine scheduler. Measured on FAv4, codegen still consolidates the last two
// sub-regions of a stage (an mfma migrates toward the region front, so ~5
// v_cvt_pk or ~3 v_exp end up past the final mfma with no window over them)
// even though the plugin's own IR had every group inside its 24-cycle window.
// sched_group_barrier is the stronger form: AMDGPUIGroupLP *builds* the
// requested pipeline in the machine scheduler rather than merely forbidding
// motion. This is the mechanism ROCm/FlyDSL uses -- it emits no reordering at
// all, just
// {[MFMA 1][VALU 5..6]} and {[MFMA 1][TRANS 3]} group declarations per cluster
// on stock upstream LLVM.
//
// Group sizing follows the validated split from the FAv3 co-issue work: treat
// one TRANS as two VALU slots (8 vs 4 cycles), so with M mfma, V valu slots and
// E trans ops -> K1 = ceil((V + 2E)/M) valu per mfma, K2 = ceil(K1/2) trans per
// mfma, g0 = round(V/K1) mfmas take VALU groups and the remaining g1 take TRANS
// groups. VALU groups are declared FIRST (valu-first measured 1071 vs exp-first
// 1047 TFLOPS on FAv3).
//
// Placement matters: IGroupLP forms groups scanning UPWARD from the barrier, so
// the whole declaration must sit AFTER every real op of the region -- emitting
// it at the top yields empty groups and silently does nothing.
static bool declareRegionGroups(Instruction *Begin, Instruction *End,
                                int syncID) {
  BasicBlock *BB = Begin->getParent();
  auto ItEnd = End ? End->getIterator() : BB->end();
  while (Begin && isa<PHINode>(Begin))
    Begin = Begin->getNextNode();
  if (!Begin || Begin == End)
    return false;

  SmallPtrSet<const Instruction *, 32> RegionInsts, Max3Folded;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It)
    RegionInsts.insert(&*It);
  computeMax3Folds(Begin, ItEnd, RegionInsts, Max3Folded);

  // Collect the region's co-issuable ops as maximal RUNS of a single IGroupLP
  // class, in program order.
  //
  // Why runs and not one VALU block + one TRANS block: the declaration must be
  // satisfiable, and satisfiability is a dataflow property. FAv4's DOT1 stage
  // after opt7 is `VALU x8 -> TRANS x8 -> VALU x50` (T3's subs, then its exp2s,
  // then the sum-reduction adds and the p->fp16 converts, which CONSUME those
  // exps). A two-block "all VALU then all TRANS" declaration asks IGroupLP to
  // schedule 58 VALU before the first TRANS, which the dependency forbids --
  // the solver then abandons the pipeline and leaves ISel's order, collapsing 8
  // exps plus their dependent adds into one 132-cycle group and leaving 6 mfmas
  // bare (measured). Emitting one group sequence per run, in order, is always
  // satisfiable because it *is* the program order, and it handles any number of
  // class transitions.
  //
  // Cost model per op (cycles / instruction count as IGroupLP counts them):
  //   * plain valu   4 cyc, 1 instruction
  //   * packed valu  8 cyc, 1 instruction -- two slots' worth of window, but
  //   ONE
  //                  entry: sched_group_barrier sizes are instruction counts,
  //                  and SIPreEmitPeephole splits whatever lands in a shadow
  //                  later
  //   * v_permlane   20 cyc, 1 instruction (cross-lane shuffle; one window can
  //   hide
  //                  a permlane plus a single 4-cycle op and no more)
  //   * TRANS (exp2) 8 cyc, 1 instruction, its own mask
  // Blocks are ordered by DEPENDENCY, not by raw program order.
  //
  // Raw program order is too fine: FAv4's PV stage emits sub(T0) exp(T0)
  // sub(T1) exp(T1) ... = 8 alternating runs, and one mandatory group sequence
  // per run does not fit 16 windows (measured: budget widened to 32, 156 cyc of
  // overflow). But those runs are freely reorderable -- sub(T1) does not depend
  // on exp(T0) -- so they belong in ONE VALU block.
  //
  // What is *not* reorderable is a VALU op that consumes a TRANS result. FAv4's
  // DOT1 stage has exactly that: T3's subs (independent) -> its exp2s -> the
  // sum-reduction adds and p->fp16 converts, which CONSUME those exps. So
  // classify:
  //
  //   block 0: VALU that does NOT depend on any TRANS in this region
  //   block 1: TRANS
  //   block 2: VALU that DOES depend on a TRANS in this region
  //
  // That is always satisfiable (it is a topological order of the class
  // dependency), collapses to the old two-block form when block 2 is empty, and
  // handles any number of program-order transitions.
  SmallPtrSet<const Instruction *, 32> transSet;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It)
    if (!isMFMAorWMMA(*It) && !Max3Folded.count(&*It) && isTransCompute(*It))
      transSet.insert(&*It);

  struct ClassRun {
    bool isTrans;
    SmallVector<int, 32> cyc;
  };
  SmallVector<ClassRun, 3> runs;
  runs.push_back({false, {}}); // VALU, TRANS-independent
  runs.push_back({true, {}});  // TRANS
  runs.push_back({false, {}}); // VALU, TRANS-dependent
  int M = 0;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It) {
    Instruction &I = *It;
    if (isMFMAorWMMA(I)) {
      ++M;
      continue;
    }
    if (Max3Folded.count(&I))
      continue; // folds into a v_maximum3, no issue slot of its own
    if (isTransCompute(I)) {
      runs[1].cyc.push_back(8);
      continue;
    }
    if (int w = valuWeight(I)) {
      bool isPermlane = false;
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *F = CI->getCalledFunction())
          isPermlane = F->getName().contains("permlane");
      int idx = dependsOnAny(&I, transSet) ? 2 : 0;
      // ONE entry per instruction, priced by its weight -- the same convention
      // the TRANS branch above uses (one entry of 8) and the same one
      // declareRegionGroupsOverCap uses for its Chunks. It matters because the
      // entry count becomes the sched_group_barrier group SIZE, and that size
      // is a count of INSTRUCTIONS. Pushing one 4-cycle entry per ELEMENT
      // instead would declare 6 where a window holds 3 packed ops; IGroupLP
      // cannot fill the group and its solver then abandons the whole region's
      // pipeline. With instruction counts the declaration is satisfiable as
      // emitted, and SIPreEmitPeephole splits whatever ends up in a shadow --
      // so no kernel needs Triton's ScalarizePackedFOps.
      if (isPermlane)
        runs[idx].cyc.push_back(kPermlaneCycles);
      else
        runs[idx].cyc.push_back(4 * w);
    }
  }
  // Drop empty blocks so they cost no groups.
  {
    SmallVector<ClassRun, 3> kept;
    for (ClassRun &r : runs)
      if (!r.cyc.empty())
        kept.push_back(std::move(r));
    runs = std::move(kept);
  }
  if (M < 2 || runs.empty())
    return false;

  // Size the co-exec window from the mfmas this region actually contains,
  // rather than assuming the 32x32x16 shape the model was validated on.
  const int Window = regionWindowCycles(Begin, ItEnd);
  if (Window <= 0)
    return false; // unmodelled mfma shape: leave it to the default schedulers
  LLVM_DEBUG({
    if (Window != kWindowCycles)
      dbgs() << "  [sgb] sync=" << syncID << " window=" << Window
             << "cyc (derived; the validated 32-cycle-mfma case is "
             << kWindowCycles << ")\n";
  });

  // Over capacity? Then no arrangement hides the work and the balanced packer
  // below is solving the wrong problem -- it would spread, find it needs more
  // groups than there are mfmas, and merge cheap pairs into double-width
  // groups. Hand those stages to the packed-aware path, which decides what to
  // cover and keeps the uncovered remainder packed.
  {
    int totalCyc = 0;
    for (const ClassRun &r : runs)
      for (int c : r.cyc)
        totalCyc += c;
    if (totalCyc > Window * M &&
        declareRegionGroupsOverCap(Begin, End, syncID, M, Window, Max3Folded,
                                   transSet))
      return true;
  }

  // Pack one block into groups. BALANCED, not greedy first-fit.
  //
  // First-fit fills each group to the brim and leaves the block's tail in a
  // stub, which wastes windows: a 20-cycle permlane after a full group opens a
  // group that can then take only one more 4-cycle op. Worse, when the
  // resulting group count exceeded the mfma count the old code widened `budget`
  // for the WHOLE stage, so every group was allowed to run 28 cycles and
  // overflow -- 40 cycles of exposed VALU in a stage whose 364 cycles of work
  // fit inside 384 cycles of capacity.
  //
  // Instead: take the minimum number of groups the cycle total needs,
  // g = ceil(total / budget), then aim for total/g per group. That keeps every
  // group at or under `budget` while spreading the slack evenly, so no group
  // overflows and the tail stub disappears.
  auto packGroups = [](ArrayRef<int> cycles, int budget) {
    SmallVector<int, 32> sizes;
    if (cycles.empty())
      return sizes;
    int total = 0;
    for (int c : cycles)
      total += c;
    // The minimum number of windows this block's cycle total needs. Hitting
    // exactly this count matters: one group too many trips the widen-the-window
    // fallback, which then lets EVERY group in the stage overflow. Measured on
    // FAv4's QK stage: 360 cyc of work, 384 cyc of capacity, 16 windows
    // available and 16 needed -- yet fragmentation around the 20-cycle permlane
    // produced 17 groups, the budget widened to 28, and 16 cycles of VALU ended
    // up exposed in a stage that fits perfectly.
    int g = (total + budget - 1) / budget;
    if (g < 1)
      g = 1;
    // Adaptive target: aim each group at the average of what is LEFT over the
    // groups still to come, capped by the window. This self-corrects after a
    // wide op (a permlane forces a short group; the following groups take a
    // little more) and lands on exactly `g` groups instead of fragmenting.
    int remCyc = total, remG = g, cur = 0, n = 0;
    for (int c : cycles) {
      int target = (remG > 0) ? (remCyc + remG - 1) / remG : budget;
      target = std::min(target, budget);
      if (n > 0 && (cur + c > budget || (cur >= target && remG > 1))) {
        sizes.push_back(n);
        remCyc -= cur;
        if (remG > 1)
          --remG;
        cur = 0;
        n = 0;
      }
      cur += c;
      ++n;
    }
    if (n)
      sizes.push_back(n);
    return sizes;
  };
  SmallVector<SmallVector<int, 32>, 8> groups;
  int budget = Window, total = 0;
  groups.clear();
  for (const ClassRun &r : runs) {
    groups.push_back(packGroups(r.cyc, budget));
    total += (int)groups.back().size();
  }
  // Too many groups for the available mfmas? Do NOT widen the window for the
  // whole stage -- that lets every group overflow (measured: 24 cyc exposed in
  // a stage whose 360 cyc of work fits in 384 cyc of capacity). Instead MERGE
  // the cheapest adjacent pair, repeatedly, so the excess is confined to one or
  // two groups.
  //
  // Fragmentation is why the count can exceed the cycle-minimum at all: a
  // 20-cycle permlane cannot share a 24-cycle window with more than one 4-cycle
  // op, so the group before it closes short. That costs 8 cycles of capacity,
  // not 24.
  while (total > M) {
    size_t bi = 0, gi = 0;
    int best = INT_MAX;
    for (size_t b = 0; b < groups.size(); ++b)
      for (size_t g = 0; g + 1 < groups[b].size(); ++g) {
        int cost = groups[b][g] + groups[b][g + 1];
        if (cost < best) {
          best = cost;
          bi = b;
          gi = g;
        }
      }
    if (best == INT_MAX)
      break; // nothing left to merge (every block is a single group)
    groups[bi][gi] += groups[bi][gi + 1];
    groups[bi].erase(groups[bi].begin() + gi + 1);
    --total;
  }
  int gBare = M - total;
  if (gBare < 0)
    gBare = 0;

  Instruction *IP = End ? End : BB->getTerminator();
  if (!IP)
    return false;
  IRBuilder<> B(IP);
  auto emit = [&](uint32_t mask, int size) {
    B.CreateIntrinsic(Intrinsic::amdgcn_sched_group_barrier,
                      {B.getInt32(mask), B.getInt32(size), B.getInt32(syncID)});
  };
  for (size_t i = 0; i < runs.size(); ++i) {
    // Partnerless mfmas go just before the LAST run, not at the region head: an
    // unfilled window is free, but having it first delays every co-issued op
    // behind it.
    if (i + 1 == runs.size())
      for (int k = 0; k < gBare; ++k)
        emit(kSGBMaskMFMA, 1);
    for (int n : groups[i]) {
      emit(kSGBMaskMFMA, 1);
      emit(runs[i].isTrans ? kSGBMaskTRANS : kSGBMaskVALU, n);
    }
  }
  LLVM_DEBUG({
    dbgs() << "  [sgb] sync=" << syncID << " M=" << M << " budget=" << budget;
    {
      int tc = 0, need = 0;
      for (const ClassRun &r : runs) {
        int c = 0;
        for (int x : r.cyc)
          c += x;
        tc += c;
        need += (c + Window - 1) / Window;
        dbgs() << " [" << (r.isTrans ? "T" : "V") << " " << c << "cyc/"
               << ((c + Window - 1) / Window) << "g]";
      }
      dbgs() << " total=" << tc << "cyc cap=" << 24 * M
             << " minGroups=" << need;
    }
    dbgs() << " runs:";
    for (size_t i = 0; i < runs.size(); ++i) {
      dbgs() << " " << (runs[i].isTrans ? "TRANS" : "VALU") << "("
             << runs[i].cyc.size() << " ops){";
      for (int n : groups[i])
        dbgs() << n << " ";
      dbgs() << "}";
    }
    dbgs() << " bare=" << gBare << "\n";
  });
  return true;
}

// Emit k x `s_nop 7` (8 idle scalar cycles each) at the
// START of every MEM region -- a region between two cluster barriers that
// carries memory ops but no mfma.
//
// Borrowed from ROCm/FlyDSL's gfx950 FA kernel, which opens every one of its
// mem clusters with `_s_nop(7)` immediately before that cluster's
// sched_barrier(0). The point is not to waste time: in an inter-wave ping-pong
// pipeline the two waves alternate dot/mem clusters, and a fixed delay at the
// head of the mem cluster shifts this wave's ds_read burst slightly later, so
// it does not collide with the other wave's LDS traffic / issue slots. It is a
// phase-tuning knob for the two-wave interleave, so the right value is
// empirical.
//
// FlyDSL writes this as raw inline asm (llvm.inline_asm "s_nop 7",
// side-effecting) because ROCDL has no s.nop op; at the LLVM-IR level we can
// use the real llvm.amdgcn.s.nop(i16) intrinsic instead.
static bool insertMemRegionNops(Function &F, int count) {
  if (count <= 0)
    return false;
  auto isRealBarrier = [](const Instruction &I) {
    if (const auto *CI = dyn_cast<CallInst>(&I))
      if (const Function *F = CI->getCalledFunction())
        return F->getName().contains("amdgcn.s.barrier");
    return false;
  };
  // Flatten the function into layout order. A stage is delimited by the REAL
  // cluster barrier (amdgcn.s.barrier), and a stage can SPAN SEVERAL BASIC
  // BLOCKS: FAv4's mem2 carries the lazy-rescale branch (map_elementwise),
  // which splits the stage across 2-4 blocks. Walking per-block therefore never
  // sees mem2's closing barrier and skipped it entirely (mem1, a single block,
  // was the only stage that got its nops).
  SmallVector<Instruction *, 512> flat;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      flat.push_back(&I);

  bool changed = false;
  for (size_t i = 0; i < flat.size(); ++i) {
    if (!isRealBarrier(*flat[i]))
      continue;
    // Classify the stage that FOLLOWS this barrier, up to the next one.
    bool hasMfma = false, hasMem = false;
    size_t j = i + 1;
    for (; j < flat.size() && !isRealBarrier(*flat[j]); ++j) {
      if (isMFMAorWMMA(*flat[j])) {
        hasMfma = true;
        break;
      }
      SchedKind K = Utils::classifySchedInst(*flat[j]);
      if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW)
        hasMem = true;
    }
    if (hasMfma || !hasMem)
      continue;
    // Insert at the head of that mem stage, i.e. right after the barrier.
    Instruction *ip = flat[i]->getNextNode();
    while (ip && isa<PHINode>(ip))
      ip = ip->getNextNode();
    if (!ip)
      continue;
    IRBuilder<> B(ip);
    for (int k = 0; k < count; ++k)
      B.CreateIntrinsic(Intrinsic::amdgcn_s_nop,
                        {B.getInt16(7)}); // s_nop 7 == 8 idle cycles
    changed = true;
  }
  return changed;
}

// Declare the co-execution schedule of one independent span [Begin, End):
// returns true if IGroupLP was given the pipeline. Independence must hold
// BOTH ways (intra-iteration): no valu uses a region mfma, AND no mfma uses a
// region valu (e.g. an mfma input built from a fptrunc). Only then can mfmas
// move freely among the valu; a dependent span (prologue / coarse) is skipped.
// SyncID identifies the declared region to IGroupLP and is bumped only for a
// declared one, so IGroupLP solves each stage's pipeline independently.
static bool declareCoExecRegion(Instruction *Begin, Instruction *End,
                                int &SyncID) {
  BasicBlock *BB = Begin->getParent();
  auto EIt = End ? End->getIterator() : BB->end();
  SmallVector<Instruction *, 32> Mfmas, Valus;
  SmallPtrSet<const Instruction *, 32> MfmaSet;
  for (auto It = Begin->getIterator(); It != EIt; ++It) {
    if (isMFMAorWMMA(*It)) {
      Mfmas.push_back(&*It);
      MfmaSet.insert(&*It);
    } else if (valuWeight(*It) > 0) {
      Valus.push_back(&*It);
    }
  }
  if (Mfmas.empty() || Valus.empty())
    return false; // classifyRegion already guarantees both, belt and braces
  SmallPtrSet<const Instruction *, 32> ValuSet(Valus.begin(), Valus.end());
  int Dep = 0;
  for (Instruction *V : Valus)
    if (dependsOnAny(V, MfmaSet))
      ++Dep;
  for (Instruction *M : Mfmas)
    if (dependsOnAny(M, ValuSet))
      ++Dep;
  LLVM_DEBUG(
      dbgs() << "[wp-region] mfma=" << Mfmas.size() << " valu=" << Valus.size()
             << " valu_dep_on_mfma=" << Dep
             << (Dep == 0 ? " INDEPENDENT -> declare" : " DEPENDENT -> skip")
             << "\n");
  if (Dep != 0)
    return false;
  // Pure declaration: do NOT physically reorder. The pass only *computes* the
  // interleave (group sizes) and hands it to IGroupLP as sched_group_barrier
  // hints, which then builds the schedule itself. Pinning a physical order
  // with sched_barrier(0) was measured weaker: codegen still consolidated a
  // stage's last groups despite it.
  return declareRegionGroups(Begin, End, ++SyncID);
}

} // namespace WP

} // namespace

namespace mlir::triton::AMD {

// Schedule every block of F, span by span between its sched.barriers. A span
// made of MFMAs and VALU (a warp-pipeline dot stage) gets the co-execution
// declaration; anything else is left alone, including the MFMA + memory spans
// that Triton's own MFMA scheduler handles. Each block is scheduled
// transactionally: snapshot, declare, verifyFunction, and roll just that
// block back if the result is invalid. Returns true iff a region was
// scheduled; the memory-stage pacing that follows does not count.
bool runLLIRSchedulePass(llvm::Function &F) {
  if (F.isDeclaration() || !WP::hasSchedBarrier(F))
    return false;

  int Scheduled = 0, SyncID = 1;
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 64> Snapshot;
    for (Instruction &I : BB)
      Snapshot.push_back(&I);
    int BlockScheduled = 0, SyncBefore = SyncID;

    // Spans between consecutive sched.barriers, in program order.
    SmallVector<std::pair<Instruction *, Instruction *>, 16> Spans;
    Instruction *SpanBegin = &BB.front();
    for (Instruction &I : BB)
      if (WP::isSchedBarrier(I)) {
        Spans.push_back({SpanBegin, &I});
        SpanBegin = I.getNextNode();
      }
    if (SpanBegin)
      Spans.push_back({SpanBegin, nullptr});
    for (auto &[B, E] : Spans) {
      if (!B || B == E)
        continue;
      int nMfma = 0, nValu = 0, nMem = 0;
      WP::RegionModel Model = WP::classifyRegion(
          B, E ? E->getIterator() : BB.end(), nMfma, nValu, nMem);
      LLVM_DEBUG(dbgs() << "[span] mfma=" << nMfma << " valu=" << nValu
                        << " mem=" << nMem << "  " << WP::modelName(Model)
                        << "\n");
      if (Model == WP::RegionModel::CoExec &&
          WP::declareCoExecRegion(B, E, SyncID))
        ++BlockScheduled;
    }

    if (BlockScheduled && verifyFunction(F, /*OS=*/nullptr)) {
      LLVM_DEBUG(dbgs() << "  invalid schedule in " << BB.getName()
                        << ", rolling the block back\n");
      restoreBlock(BB, Snapshot);
      SyncID = SyncBefore;
      continue;
    }
    Scheduled += BlockScheduled;
  }

  // Memory-stage head pacing: a stage may span several blocks, so it is
  // inserted function-wide, and rolled back function-wide if it does not
  // verify.
  if (Scheduled > 0) {
    SmallVector<SmallVector<Instruction *, 64>, 8> Snapshots;
    for (BasicBlock &BB : F) {
      Snapshots.emplace_back();
      for (Instruction &I : BB)
        Snapshots.back().push_back(&I);
    }
    if (WP::insertMemRegionNops(F, WP::kDefaultMemNops) &&
        verifyFunction(F, /*OS=*/nullptr)) {
      unsigned i = 0;
      for (BasicBlock &BB : F)
        restoreBlock(BB, Snapshots[i++]);
    }
  }
  return Scheduled > 0;
}

} // namespace mlir::triton::AMD

// ---- Plugin wrapper: the same entry point as Triton's in-tree pass, run as a
// new-PassManager function pass at the OptimizerLast extension point of
// make_llir's O3 pipeline (load with LLVM_PASS_PLUGIN_PATH).
namespace {
struct LlirSchedPass : llvm::PassInfoMixin<LlirSchedPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &) {
    if (!mlir::triton::AMD::runLLIRSchedulePass(F))
      return llvm::PreservedAnalyses::all();
    // Only reorders / inserts within blocks; the CFG is preserved.
    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
  }
};
} // namespace

llvm::PassPluginLibraryInfo getLlirSchedPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "LlirSched", "v0.3", [](llvm::PassBuilder &PB) {
        PB.registerOptimizerLastEPCallback([](llvm::ModulePassManager &MPM,
                                              llvm::OptimizationLevel,
                                              llvm::ThinOrFullLTOPhase) {
          MPM.addPass(llvm::createModuleToFunctionPassAdaptor(LlirSchedPass()));
        });
        // Also allow explicit `-passes=llir-sched` for triton-opt/opt.
        PB.registerPipelineParsingCallback(
            [](llvm::StringRef Name, llvm::FunctionPassManager &FPM,
               llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
              if (Name == "llir-sched") {
                FPM.addPass(LlirSchedPass());
                return true;
              }
              return false;
            });
      }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getLlirSchedPluginInfo();
}
