//############################################################################
// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//############################################################################

// Out-of-tree LLVM new-PassManager pass plugin: the gfx950 LLIR scheduler
// (MFMA <-> memory interleave for GEMM hot loops), ported from AMD-Triton
// triton-mi450 PR #73 (the sched.barrier variant). Self-contained: depends only
// on LLVM headers. Load into Triton via LLVM_PASS_PLUGIN_PATH; it auto-inserts
// at the OptimizerLast extension point of make_llir's optimize_module O3 run.
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
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "tritonamdgpu-llir-schedule"

// Inlined from PR#73's TritonAMDGPUToLLVM/MfmaUtility.h so the plugin needs no
// triton headers.
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

// Classification of an instruction for scheduling purposes.
enum class SchedKind { MFMA, GR, LR, LW, Other };

// LDS resides in address space 3 on AMDGPU.
constexpr unsigned kLDSAddressSpace = 3;

// Structures used for region analysis/scheduling
struct AnchorInst {
  Instruction *I = nullptr;
  SchedKind Kind = SchedKind::Other;
};

struct MFMARegionInfo {
  Instruction *RegionStart = nullptr;
  unsigned TotalMFMA = 0;
};

using MFMARegionList = SmallVector<MFMARegionInfo, 8>;
using BBMFMAAnalysisMap = DenseMap<const BasicBlock *, MFMARegionList>;

struct BBRegion {
  BasicBlock *BB = nullptr;
  Instruction *Begin = nullptr; // First instruction in region (inclusive)
  Instruction *End =
      nullptr; // First instruction of next region or nullptr (exclusive)
};

struct MFMARegionCollectResult {
  // MFMA-input prep instructions to hoist to the region start.
  SmallVector<Instruction *, 16> Hoist;
  // MFMA-result users to sink toward the region end.
  SmallVector<Instruction *, 16> Sink;
  // Last memory anchor seen while collecting (used to place trailing MFMAs).
  Instruction *LastAnchor = nullptr;
  // Memory ops (GR/LR/LW) that the region's MFMAs are spaced around.
  SmallVector<AnchorInst, 32> Anchors;
  // The region's MFMA instructions, in program order.
  SmallVector<Instruction *, 32> MFMAInsts;
};

// Stateless helpers shared by region analysis and scheduling.
namespace Utils {
bool isMFMAorWMMA(const Instruction &I) {
  // Shared matrix-core predicate (also used by the scalarize-packed-fops
  // pass). gfx950 only exposes MFMA, but the helper is family-agnostic.
  return mlir::triton::AMD::isMFMAorWMMA(I);
}

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
        //     raw.ptr.buffer.store (gmem store from regs)
        if (Name.contains("buffer.load") ||
            Name.contains("raw.ptr.buffer.store"))
          return SchedKind::GR;
        // LR: ds_read (ds.read.*) or ds_load (ds.load.*)
        if (Name.contains("ds.read") || Name.contains("ds.load"))
          return SchedKind::LR;
      }
    }
  }

  // LR: load from LDS (addrspace 3)
  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    if (LI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LR;
  }

  // LW: store to LDS (addrspace 3)
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    if (SI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LW;
  }

  return SchedKind::Other;
}

iterator_range<BasicBlock::iterator> instructionsInRegion(const BBRegion &R) {
  BasicBlock *BB = R.BB;
  // Begin is now inclusive (region starts at this instruction)
  auto ItBegin = R.Begin ? R.Begin->getIterator() : BB->begin();
  auto ItEnd = R.End ? R.End->getIterator() : BB->end();
  return make_range(ItBegin, ItEnd);
}

unsigned getMFMACycles(const Instruction &I) {
  if (!isMFMAorWMMA(I))
    return 0;
  const auto *CI = cast<CallInst>(&I);
  const Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return 0;
  StringRef Name = Callee->getName();

  // Scaled f8f6f4 MFMAs: the cost depends on the operand formats encoded in
  // cbsz (arg 3) and blgp (arg 4).
  if (Name.contains("mfma.scale.f32.16x16x128.f8f6f4")) {
    // both operands f4 -> 16 cycles, otherwise (either operand f8) -> 32.
    if (auto *CbszC = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
      if (auto *BlgpC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
        return (CbszC->getZExtValue() > 1 && BlgpC->getZExtValue() > 1) ? 16
                                                                        : 32;
    return 32; // Fallback if cbsz/blgp are not constants
  }
  if (Name.contains("mfma.scale.f32.32x32x64.f8f6f4")) {
    // both operands f4 -> 32 cycles, otherwise (either operand f8) -> 64.
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
      {"mfma.f32.16x16x32.f16", 16},  {"mfma.f32.16x16x32.bf16", 16},
      {"mfma.i32.16x16x64.i8", 16},   {"mfma.f32.32x32x16.f16", 32},
      {"mfma.f32.32x32x16.bf16", 32}, {"mfma.i32.32x32x32.i8", 32},
  };
  for (const auto &Entry : kFixedCycles)
    if (Name.contains(Entry.Name))
      return Entry.Cycles;

  // Unknown / unmodeled shape: the scheduler bails on this region and leaves
  // it to the default LLVM schedulers (such kernels are not perf-critical).
  return 0;
}

// Width in bits of the value moved by an LDS-access anchor.
unsigned getLDSAccessBits(const Instruction *I) {
  if (const auto *LI = dyn_cast<LoadInst>(I))
    return LI->getType()->getPrimitiveSizeInBits();
  if (const auto *SI = dyn_cast<StoreInst>(I))
    return SI->getValueOperand()->getType()->getPrimitiveSizeInBits();
  if (const auto *CI = dyn_cast<CallInst>(I))
    return CI->getType()->getPrimitiveSizeInBits();
  return 0;
}

// LDS instruction throughput during steady state, which is proportional to
// the access bits.
unsigned getLDSCoverCycles(const Instruction *I, unsigned MFMACycles) {
  unsigned Bits = getLDSAccessBits(I);
  return Bits ? (Bits / 8) : MFMACycles;
}

// MFMAs to emit at this LDS access under a throughput model: reads and writes
// share the one LDS issue port, so we carry a running cycle balance across
// the region's accesses and emit floor(balance / MFMACycles) MFMAs here,
// keeping the remainder for the next access.
unsigned takeMFMAsForLDS(const Instruction *I, unsigned MFMACycles,
                         unsigned &AccumCycles) {
  AccumCycles += getLDSCoverCycles(I, MFMACycles);
  unsigned N = AccumCycles / MFMACycles; // floor; carry the remainder
  AccumCycles -= N * MFMACycles;
  return N;
}
} // namespace Utils

// Region analysis and scheduling logic grouped into a helper class
class LLIRScheduler {
public:
  explicit LLIRScheduler() = default;

  // Roll a block back to a pre-scheduling snapshot: erase the instructions the
  // scheduler inserted (the region-comment inline-asm calls and the
  // llvm.amdgcn.sched.barrier intrinsics, all void with no uses) and restore
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

  // Schedule every block in the function. Region detection + the per-region
  // structural invariant make this safe: a block with no eligible MFMA region
  // is simply left untouched (so no loop-finding heuristic is needed, and an
  // odd prologue / multiple loops / no loop are all handled uniformly).
  // Each block is scheduled transactionally: if its schedule fails
  // verification (e.g. an epilogue the main logic can't safely interleave),
  // only that block is rolled back, so good blocks keep their schedule.
  // Returns true if any region was scheduled.
  bool run(Function &F) {
    LLVM_DEBUG(dbgs() << "LLIR scheduler analyzing function: " << F.getName()
                      << "\n");
    BBMFMAAnalysisMap BBMFMAMap;
    bool scheduled = false;
    for (BasicBlock &BB : F) {
      LLVM_DEBUG(dbgs() << "BB: " << BB.getName() << "\n");
      analyzeBB(BB, BBMFMAMap);

      // Snapshot the block so we can revert just this block on failure.
      SmallVector<Instruction *, 64> snapshot;
      for (Instruction &I : BB)
        snapshot.push_back(&I);

      if (!scheduleBB(BB, BBMFMAMap))
        continue;

      if (verifyFunction(F, nullptr)) {
        // This block's schedule is invalid; bail gracefully on it alone.
        LLVM_DEBUG(dbgs() << "  reverting unschedulable block " << BB.getName()
                          << "\n");
        restoreBlock(BB, snapshot);
      } else {
        scheduled = true;
      }
    }
    return scheduled;
  }

private:
  // Split a basic block into MFMA regions in a single program-order pass,
  // recording each region's first MFMA (its RegionStart) and its MFMA count.
  // A new region opens at every MFMA that follows a memory op (GR/LR/LW) seen
  // since the region began; by construction an MFMA's input loads land in an
  // earlier region, so intra-region reordering is dependency-safe.
  static void analyzeBB(BasicBlock &BB, BBMFMAAnalysisMap &Out) {
    MFMARegionList Regions;
    unsigned CurRegion = 0;
    bool SeenMemoryOps = false;
    bool InRegion = false;

    for (Instruction &I : BB) {
      SchedKind SK = Utils::classifySchedInst(I);
      if (SK == SchedKind::GR || SK == SchedKind::LR || SK == SchedKind::LW)
        SeenMemoryOps = true;

      if (!Utils::isMFMAorWMMA(I))
        continue;

      // This MFMA opens a new region when we are not already in one, or when a
      // memory op has appeared since the current region's MFMAs began (that op
      // feeds this MFMA, so it belongs to the next region). Otherwise the MFMA
      // just extends the current region.
      bool startNewRegion = !InRegion || SeenMemoryOps;
      if (startNewRegion) {
        if (InRegion)
          CurRegion++; // close the region we were in, open the next
        InRegion = true;
        // Memory ops seen *before* a region's first MFMA are that region's own
        // setup (their data feeds these MFMAs), not a boundary; clear the flag
        // so they don't spuriously split off the next MFMA.
        SeenMemoryOps = false;
        if (CurRegion >= Regions.size())
          Regions.resize(CurRegion + 1);
        Regions[CurRegion].RegionStart = &I; // first MFMA is the region start
      }
      Regions[CurRegion].TotalMFMA++;
    }

    if (Regions.empty())
      return;

    LLVM_DEBUG({
      for (unsigned i = 0; i < Regions.size(); ++i)
        dbgs() << "Region " << i << ": total MFMA: " << Regions[i].TotalMFMA
               << "\n";
    });

    Out[&BB] = std::move(Regions);
  }

  static bool feedsMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;

    Worklist.push_back(I);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      for (User *U : V->users()) {
        if (auto *UI = dyn_cast<Instruction>(U)) {
          if (Utils::isMFMAorWMMA(*UI))
            return true;
          if (Utils::isHoistTransparentInst(*UI))
            Worklist.push_back(UI);
        }
      }
    }
    return false;
  }

  static bool definedByMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;

    Worklist.push_back(I);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      if (auto *DefI = dyn_cast<Instruction>(V)) {
        if (Utils::isMFMAorWMMA(*DefI))
          return true;

        if (Utils::isSinkTransparentInst(*DefI)) {
          for (Value *Op : DefI->operands())
            Worklist.push_back(Op);
        }
      }
    }
    return false;
  }

  static MFMARegionCollectResult
  collectMFMAAndTransparentInstsInRegion(const BBRegion &R) {
    MFMARegionCollectResult Res;

    // All instructions in this region. Hoisting moves a prep to the region
    // start (right after R.Begin). That is only safe if the prep's operands are
    // already available there; if an operand is defined later inside the
    // region, hoisting the prep above it would use a value before its
    // definition (an SSA dominance violation / invalid IR). This set lets us
    // detect that case and leave such preps in place.
    SmallPtrSet<const Instruction *, 32> RegionInsts;
    for (Instruction &I : Utils::instructionsInRegion(R))
      RegionInsts.insert(&I);

    // Preps cleared for hoisting so far (collected in program order). An
    // operand that is a same-region prep we already hoisted stays ahead of its
    // use.
    SmallPtrSet<const Instruction *, 16> Hoisted;

    for (Instruction &I : Utils::instructionsInRegion(R)) {
      SchedKind K = Utils::classifySchedInst(I);
      if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW) {
        Res.LastAnchor = &I;
        Res.Anchors.push_back({&I, K});
        continue;
      }

      if (K == SchedKind::MFMA) {
        Res.MFMAInsts.push_back(&I);
        continue;
      }

      if (Utils::isHoistTransparentInst(I)) {
        // Hoisting moves I to right after the region start, so it is safe only
        // if every operand still dominates that position: operands defined
        // before the region already do, R.Begin does, and a prep we are also
        // hoisting keeps its relative order ahead of I. An operand defined
        // inside the region that we are NOT hoisting — an LR/LW/GR anchor, or a
        // prep we rejected — would end up after its use, so I must stay put.
        // This covers both shuffle and insertelement, all anchor kinds, and
        // multi-hop chains.
        bool safeToHoist = true;
        for (Value *Op : I.operands()) {
          auto *OpI = dyn_cast<Instruction>(Op);
          if (!OpI || OpI == R.Begin)
            continue;
          if (RegionInsts.count(OpI) && !Hoisted.count(OpI)) {
            safeToHoist = false;
            break;
          }
        }
        if (safeToHoist && feedsMFMA(&I)) {
          Res.Hoist.push_back(&I);
          Hoisted.insert(&I);
        }
        continue;
      }

      if (isa<ExtractElementInst>(I)) {
        if (definedByMFMA(&I))
          Res.Sink.push_back(&I);
      }
    }

    return Res;
  }

  static MFMARegionCollectResult
  preprocessMFMAInstsInRegion(const BBRegion &R) {
    auto Res = collectMFMAAndTransparentInstsInRegion(R);

    if (Res.Hoist.empty() && Res.Sink.empty())
      return Res;

    Instruction *HoistPos = R.Begin; // Region start (the region's first MFMA)
    Instruction *SinkPos = Res.LastAnchor; // last anchor in region

    if (HoistPos)
      for (Instruction *I : llvm::reverse(Res.Hoist)) {
        // Don't hoist R.Begin after itself
        if (I != HoistPos)
          I->moveAfter(HoistPos);
      }

    // Sinking needs a trailing anchor to move past. A region with MFMAs but no
    // GR/LR/LW anchor (LastAnchor stays null) has no valid sink point, so leave
    // the extractelements in place rather than dereferencing a null insert
    // position. Such a region carries no anchors, so scheduleMFMAWithSpacing
    // bails on it anyway.
    // Sink ALL MFMA-result extractelements to just past the region's last
    // anchor. This is required, not just opportunistic: it clears them out of
    // the MFMA run so the subsequent interleaving (which only reorders
    // instructions before LastAnchor) cannot move an MFMA past one of its own
    // result extracts. A region with MFMAs but no anchor has no sink point, so
    // the SinkPos null-guard leaves those extracts in place (scheduleMFMAWith-
    // Spacing bails on such a region anyway). Any genuinely unsafe sink is
    // caught by the per-block verifyFunction rollback.
    if (SinkPos)
      for (Instruction *I : llvm::reverse(Res.Sink))
        I->moveAfter(SinkPos);

    return Res;
  }

  static StringRef schedKindName(SchedKind K) {
    switch (K) {
    case SchedKind::GR:
      return "GR";
    case SchedKind::LR:
      return "LR";
    case SchedKind::LW:
      return "LW";
    case SchedKind::MFMA:
      return "mfma";
    case SchedKind::Other:
      return "other";
    }
    llvm_unreachable("unknown SchedKind");
  }

  // Helper: move N MFMAs after InsertPt using moveAfter.
  // moveAfter naturally produces correct order: each new MFMA goes right
  // after InsertPt, pushing previous ones further away.
  // Result: InsertPt, MFMA[N-K], ..., MFMA[N-2], MFMA[N-1]
  static unsigned moveMFMAsAfter(SmallVectorImpl<Instruction *> &MFMAInsts,
                                 unsigned &MFMAIdx, unsigned Count,
                                 Instruction *InsertPt) {
    unsigned moved = 0;
    for (unsigned j = 0; j < Count && MFMAIdx > 0; ++j) {
      MFMAInsts[--MFMAIdx]->moveAfter(InsertPt);
      moved++;
    }
    return moved;
  }

  // Interleave MFMA with anchor instructions using moveAfter.
  //
  // Pure throughput model — every count is "how many MFMAs of compute cover
  // this memory op's issue-port occupancy needs", never a latency-hiding
  // reorder:
  //   - GR (global load): ceil(64 / mfma_cycles) MFMAs (1 if immediately
  //                       followed by an LR).
  //   - LR / LW (LDS read/write): floor(carried LDS-cycle balance /
  //   mfma_cycles)
  //                       MFMAs. Reads and writes share the one LDS issue port,
  //                       so both use the same width-proportional pairing.
  //   - 2 MFMAs drain the tail; any leftover compute (more MFMAs than the
  //   memory
  //     ops demand cover for) is split evenly between the region's head and
  //     tail, with an odd MFMA favoring the head.
  static void
  scheduleMFMAWithSpacing(SmallVectorImpl<AnchorInst> &Anchors,
                          SmallVectorImpl<Instruction *> &MFMAInsts) {
    if (Anchors.empty() || MFMAInsts.empty())
      return;

    unsigned mfmaCycles = Utils::getMFMACycles(*MFMAInsts.front());
    if (mfmaCycles == 0)
      return;
    // A global load occupies the global-load path for ~64 cycles, so it needs
    // 64 cycles of MFMA cover — ceil(64 / mfma_cycles) MFMAs (4 for a 16-cycle
    // MFMA, 2 for 32-cycle). Same throughput basis as the LDS-access pairing.
    unsigned mfmaPerGR = llvm::divideCeil(64, mfmaCycles);

    unsigned MFMAIdx = MFMAInsts.size();
    unsigned Total = MFMAIdx;

    // Count anchors by kind. ldsBudget is the total MFMA cover the region's LDS
    // accesses demand, floor(sum(cycles_per_access) / mfma_cycles) — the
    // throughput ratio over reads *and* writes alike, so cheap accesses share
    // an MFMA and wide ones draw several.
    unsigned numGR = 0, numGRBeforeLR = 0, totalLDSCycles = 0;
    for (size_t j = 0; j < Anchors.size(); ++j) {
      if (Anchors[j].Kind == SchedKind::GR) {
        numGR++;
        if (j + 1 < Anchors.size() && Anchors[j + 1].Kind == SchedKind::LR)
          numGRBeforeLR++;
      } else if (Anchors[j].Kind == SchedKind::LR ||
                 Anchors[j].Kind == SchedKind::LW) {
        totalLDSCycles += Utils::getLDSCoverCycles(Anchors[j].I, mfmaCycles);
      }
    }
    unsigned ldsBudget = totalLDSCycles / mfmaCycles;

    // gfx950 scheduling:
    //   GR: mfmaPerGR MFMAs each (except GR→LR gets 1)
    //   LR/LW: cycle-paired at the true throughput ratio (takeMFMAsForLDS) —
    //          cheap accesses share an MFMA, wide ones draw several
    //   2 MFMAs drain the end; leftover compute is split evenly head/tail
    unsigned grBudget = mfmaPerGR * (numGR - numGRBeforeLR);
    unsigned needed = grBudget + numGRBeforeLR + ldsBudget + 2;
    unsigned leftover = (Total > needed) ? Total - needed : 0;

    // Surplus compute (more MFMAs than the memory ops need cover for) is split
    // evenly between the region's head and tail; an odd MFMA favors the head.
    unsigned tailLeftover = leftover / 2;            // floor → tail
    unsigned headLeftover = leftover - tailLeftover; // ceil → head

    LLVM_DEBUG(dbgs() << "  MFMA budget: total=" << Total
                      << ", needed=" << needed << ", leftover=" << leftover
                      << " (head=" << headLeftover << ", tail=" << tailLeftover
                      << ")\n");

    // The 2-MFMA tail drain plus the tail's share of the surplus. The head's
    // share is whatever stays unmoved at the front after the reverse walk.
    unsigned MFMAAtEnd =
        moveMFMAsAfter(MFMAInsts, MFMAIdx, 2 + tailLeftover, Anchors.back().I);
    // Running LDS-cycle balance carried across the region's LDS accesses (LR
    // and LW, processed in reverse) so the MFMA:access pairing follows the true
    // throughput ratio.
    unsigned ldsAccum = 0;
    DenseMap<SchedKind, unsigned> MFMAPerAnchorKind;

    for (int i = static_cast<int>(Anchors.size()) - 1; i >= 0 && MFMAIdx > 0;
         --i) {
      size_t idx = static_cast<size_t>(i);
      Instruction *InsertPt = Anchors[idx].I;
      SchedKind Kind = Anchors[idx].Kind;

      unsigned Count = 0;
      if (Kind == SchedKind::LR || Kind == SchedKind::LW) {
        // Cycle model: emit floor(balance / mfma_cycles) MFMAs, carrying the
        // remainder — cheap accesses share an MFMA, wide ones draw several.
        // Reads and writes use the same shared LDS-cycle balance.
        Count = Utils::takeMFMAsForLDS(Anchors[idx].I, mfmaCycles, ldsAccum);
      } else if (Kind == SchedKind::GR) {
        bool followedByLR = (idx + 1 < Anchors.size() &&
                             Anchors[idx + 1].Kind == SchedKind::LR);
        Count = followedByLR ? 1 : mfmaPerGR;
      }

      [[maybe_unused]] unsigned moved =
          moveMFMAsAfter(MFMAInsts, MFMAIdx, Count, InsertPt);
      LLVM_DEBUG(MFMAPerAnchorKind[Kind] += moved);
    }

    LLVM_DEBUG({
      dbgs() << "  MFMA insertion summary: total=" << Total
             << ", at_front=" << MFMAIdx << ", at_end=" << MFMAAtEnd;
      for (auto &KV : MFMAPerAnchorKind) {
        dbgs() << ", after_" << schedKindName(KV.first) << "=" << KV.second;
      }
      dbgs() << "\n";
    });
  }

  // Insert an inline asm comment before the given instruction.
  // Emit a non-side-effecting inline-asm comment (a pure annotation, NOT a
  // reorder barrier). The sched.barriers we emit at anchors -- not these region
  // markers -- pin the schedule, so the markers must stay side-effect-free to
  // avoid adding scheduling constraints of their own.
  static void insertAsmComment(Instruction *IP, const std::string &Comment) {
    LLVMContext &Ctx = IP->getContext();
    IRBuilder<> Builder(Ctx);
    Builder.SetInsertPoint(IP);
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
    InlineAsm *IA =
        InlineAsm::get(FTy, ";; " + Comment, "", /*hasSideEffects=*/false);
    Builder.CreateCall(IA);
  }

  // Insert llvm.amdgcn.sched.barrier(Mask) immediately after AfterI so the pre-
  // and post-RA machine schedulers cannot move instructions across this anchor.
  // Mask bits name the instruction classes allowed to cross; 0 is a full
  // barrier.
  static void insertSchedBarrier(Instruction *AfterI, uint32_t Mask) {
    Instruction *Next = AfterI->getNextNode();
    if (!Next)
      return; // anchors are never terminators, but be defensive
    IRBuilder<> Builder(AfterI->getContext());
    Builder.SetInsertPoint(Next);
    Builder.CreateIntrinsic(Intrinsic::amdgcn_sched_barrier,
                            {Builder.getInt32(Mask)});
  }

  static bool scheduleBB(BasicBlock &BB, const BBMFMAAnalysisMap &Analysis) {
    auto It = Analysis.find(&BB);
    if (It == Analysis.end())
      return false;

    const MFMARegionList &Regions = It->second;

    unsigned NumRegions = Regions.size();
    unsigned ScheduledRegionIdx = 0;

    for (unsigned i = 0; i < NumRegions; ++i) {
      const MFMARegionInfo &R = Regions[i];
      if (!R.RegionStart)
        continue;

      if (R.TotalMFMA != 0) {
        BBRegion bbR;
        bbR.BB = &BB;
        bbR.Begin = Regions[i].RegionStart;
        bbR.End = (i + 1 < NumRegions) ? Regions[i + 1].RegionStart : nullptr;

        // Schedulability check, performed BEFORE any mutation: bail on a region
        // whose MFMA shape we don't model (getMFMACycles == 0) or that has no
        // memory anchor to interleave the MFMAs against. Skipping here --
        // before preprocessMFMAInstsInRegion moves anything -- ensures the pass
        // only schedules (and sched.barrier-pins) regions it actually models,
        // and only reports success (which gates the AGPR-form flags) for those;
        // unmodeled regions are left to the default LLVM schedulers.
        unsigned MFMACycles = 0;
        bool SeenMFMA = false, HasAnchor = false;
        for (Instruction &I : Utils::instructionsInRegion(bbR)) {
          SchedKind K = Utils::classifySchedInst(I);
          if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW)
            HasAnchor = true;
          else if (K == SchedKind::MFMA && !SeenMFMA) {
            SeenMFMA = true;
            MFMACycles = Utils::getMFMACycles(I);
          }
        }
        if (MFMACycles == 0 || !HasAnchor)
          continue;

        MFMARegionCollectResult Res = preprocessMFMAInstsInRegion(bbR);

        // --- Build region comment ---
        std::string Comment;
        raw_string_ostream OS(Comment);

        // Count anchors by kind
        unsigned numGR = 0, numLR = 0, numLW = 0;
        for (auto &A : Res.Anchors) {
          if (A.Kind == SchedKind::GR)
            numGR++;
          else if (A.Kind == SchedKind::LR)
            numLR++;
          else if (A.Kind == SchedKind::LW)
            numLW++;
        }
        OS << "Region " << ScheduledRegionIdx << ": " << Res.MFMAInsts.size()
           << " mfma, " << numGR << " GR, " << numLR << " LR, " << numLW
           << " LW";
        ScheduledRegionIdx++;

        insertAsmComment(bbR.Begin, Comment);

        LLVM_DEBUG({
          dbgs() << "Cluster " << i << " structure:";
          SchedKind RunKind = SchedKind::Other;
          unsigned RunCount = 0;
          for (Instruction &Inst : Utils::instructionsInRegion(bbR)) {
            SchedKind K = Utils::classifySchedInst(Inst);
            if (K != SchedKind::MFMA && K != SchedKind::GR &&
                K != SchedKind::LR && K != SchedKind::LW)
              continue;
            if (K == RunKind) {
              RunCount++;
            } else {
              if (RunCount > 0)
                dbgs() << " " << RunCount << " " << schedKindName(RunKind);
              RunKind = K;
              RunCount = 1;
            }
          }
          if (RunCount > 0)
            dbgs() << " " << RunCount << " " << schedKindName(RunKind);
          dbgs() << "\n";
        });

        scheduleMFMAWithSpacing(Res.Anchors, Res.MFMAInsts);

        // Pin this region's schedule with a full sched.barrier (mask 0) after
        // each memory anchor, so LLVM's pre- and post-RA machine schedulers
        // preserve the MFMA<->mem interleave. This keeps misched enabled for
        // the rest of the function -- prologue/epilogue and any region the pass
        // bailed on still get machine-scheduled -- so no global misched-disable
        // is needed.
        for (const AnchorInst &A : Res.Anchors)
          insertSchedBarrier(A.I, /*Mask=*/0);
      }
    }
    return ScheduledRegionIdx > 0;
  }
};


// ---- New-PassManager wrapper (out-of-tree port of PR#73's legacy pass) ----
// Transactional: clone the function, schedule, verifyFunction, and roll back to
// the pristine body on failure (so a bad schedule never reaches codegen). The
// schedule is pinned by the sched.barriers the scheduler inserts, so nothing
// downstream (make_amdgcn) needs to disable LLVM's machine scheduler.
static bool runLlirScheduleTransactional(Function &F) {
  if (F.isDeclaration())
    return false;

  ValueToValueMapTy VMap;
  Function *Backup = CloneFunction(&F, VMap);
  Backup->setName(F.getName() + ".llirsched.bak");

  LLIRScheduler Scheduler;
  bool didSchedule = Scheduler.run(F);

  if (verifyFunction(F, /*OS=*/nullptr)) {
    LLVM_DEBUG(dbgs() << "LLIR schedule produced invalid IR; rolling back.\n");
    auto OrigLinkage = F.getLinkage();
    F.deleteBody();
    F.splice(F.end(), Backup);
    F.setLinkage(OrigLinkage);
    for (unsigned i = 0, e = F.arg_size(); i != e; ++i)
      Backup->getArg(i)->replaceAllUsesWith(F.getArg(i));
    Backup->eraseFromParent();
    return false;
  }

  Backup->eraseFromParent();
  return didSchedule;
}

struct LlirSchedPass : PassInfoMixin<LlirSchedPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    bool changed = runLlirScheduleTransactional(F);
    if (!changed)
      return PreservedAnalyses::all();
    // Only reorders/insert within blocks; CFG is preserved.
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }
};

} // end anonymous namespace

llvm::PassPluginLibraryInfo getLlirSchedPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LlirSched", "v0.1",
          [](llvm::PassBuilder &PB) {
            // Auto-insert at the very end of the O3 function pipeline so the
            // pass sees near-final IR (individual mfma / ds_read / buffer_load).
            PB.registerOptimizerLastEPCallback(
                [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel,
                   llvm::ThinOrFullLTOPhase) {
                  MPM.addPass(
                      llvm::createModuleToFunctionPassAdaptor(LlirSchedPass()));
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
