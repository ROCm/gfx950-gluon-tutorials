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

"""Transform PR#73's in-tree LLIRSchedule.cpp into a self-contained, out-of-tree
new-PassManager LLVM pass plugin (libLlirSched.so).

- strips the two triton-internal headers (MfmaUtility.h, Passes.h) and inlines
  mlir::triton::AMD::isMFMAorWMMA
- keeps the anonymous-namespace Utils + LLIRScheduler class verbatim
- drops the legacy FunctionPass wrapper + legacy runLLIRSchedulePass
- appends a new-PM pass (transactional clone/verify/rollback preserved) and
  llvmGetPassPluginInfo() that auto-inserts at the OptimizerLast extension point
"""
import argparse
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description="Regenerate LlirSchedPlugin.cpp from the upstream in-tree "
    "LLIRSchedule.cpp (AMD-Triton triton-mi450 PR #73, the sched.barrier variant).")
parser.add_argument(
    "--src", required=True,
    help="Path to a checkout of the upstream in-tree LLIRSchedule.cpp to port.")
parser.add_argument(
    "--out", default=os.path.join(_HERE, "LlirSchedPlugin.cpp"),
    help="Output path for the generated plugin source "
    "(default: the checked-in LlirSchedPlugin.cpp next to this script).")
args = parser.parse_args()

SRC = args.src
OUT = args.out
OUT_DIR = os.path.dirname(OUT) or "."

lines = open(SRC).read().splitlines()

# locate anchors
def find(pred, start=0):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise RuntimeError("anchor not found")

i_ns = find(lambda l: l.strip() == "namespace {")
# cut the legacy wrapper: from the "// Pass wrapper" comment (fallback: the struct)
try:
    i_cut = find(lambda l: "Pass wrapper" in l, i_ns)
except RuntimeError:
    i_cut = find(lambda l: "struct LLIRSchedulePass" in l, i_ns)

body = "\n".join(lines[i_ns:i_cut]).rstrip()

PREAMBLE = r'''// Out-of-tree LLVM new-PassManager pass plugin: the gfx950 LLIR scheduler
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

'''

SUFFIX = r'''

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
'''

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT, "w") as f:
    f.write(PREAMBLE)
    f.write(body)
    f.write("\n")
    f.write(SUFFIX)

print(f"wrote {OUT}")
print(f"  namespace {{ at src line {i_ns+1}, cut legacy wrapper at src line {i_cut+1}")
print(f"  body lines kept: {i_cut - i_ns}, total output lines: {open(OUT).read().count(chr(10))}")
