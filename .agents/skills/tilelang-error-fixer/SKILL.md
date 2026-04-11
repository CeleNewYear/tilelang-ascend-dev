---
name: tilelang-error-fixer
description: TileLang npuir 错误诊断与修复技能。用户提及编译失败、运行错误、pass 异常、结果错误、性能回退、Core Dump、段错误、BishengIR 编译报错、sync 死锁、load/store 维度不一致时必须使用本技能。
---

# TileLang Error Fixer (npuir)

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Scope

- compile errors in npuir path
- runtime failures and invalid results
- pass pipeline divergence
- performance regressions

## Diagnosis workflow

1. Confirm environment and target setting
2. Reproduce with smallest kernel
3. Classify issue type: compile, runtime, pass, precision, performance
4. Capture evidence: logs, IR snapshot, failing stage
5. Propose minimal patch and validate

## NPUIR-specific checks

- verify default vector API style uses v-prefix ops
- verify alias callsites are semantically equivalent
- verify load_nd2nz and store_fixpipe size/layout consistency
- verify sync_block_set and sync_block_wait pairing

## Runtime signature quick triage (Mandatory)

When logs contain one or more of the following signatures:

- `AclSetCompileopt(... ACL_PRECISION_MODE ...)`, `error code is 500001`
- `ERR00100 PTA call acl api failed`
- `Environment_Error_Failed_To_Import_Python_Module`
- `Cannot find global function cce.product_init`

Prioritize environment initialization diagnosis before kernel-level debugging:

1. Verify CANN environment scripts are sourced correctly (`set_env.sh`) and Python path matches the active environment.
2. Verify torch_npu lazy-init path (`torch_npu.npu._lazy_init()` / `current_device`) with a minimal script.
3. For CANN versions earlier than 8.5, try compatibility workaround: `export ACL_OP_INIT_MODE=1`.
4. For CANN 8.5 and later, this issue is expected to be fixed; keep step 3 only as fallback when the above signatures still appear.

## Official docs to consult

- docs/Tilelang算子调试指南.md
- docs/开发指南.md
- docs/developer/EnvironmentVariables.md
- docs/Tilelang.language/内存操作/T.load_nd2nz.md
- docs/Tilelang.language/内存操作/T.store_fixpipe.md

## Output template

## TileLang JIT Issue Report

### Summary
- Symptom:
- Repro script:
- Impact:

### Root Cause
- Layer: frontend or pass or codegen or runtime
- Fault pattern:

### Fix
- Minimal change:
- Why this fixes it:

### Verification
- Repro after fix:
- Numerical check:
- Regression risk:
