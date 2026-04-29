# Security Policy

## Reporting a Vulnerability

This repository contains **educational reference kernels**, not production-deployed code. We do not expect typical security vulnerabilities (memory corruption, privilege escalation, etc.) to apply here. That said, if you discover any issue you believe constitutes a security risk — for example, a kernel that produces incorrect results in a way that could mislead downstream users, a build script that runs untrusted code, or any embedded credential or PII — please report it privately rather than opening a public issue.

To report privately:

1. Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) feature on this repository, or
2. Email `rocm-security@amd.com` with details and a suggested mitigation.

Please include enough information to reproduce the issue: the affected kernel/version, the build environment (ROCm version, Triton commit), the input shapes, and a minimal repro if possible.

## Disclosure Timeline

We aim to acknowledge reports within five business days and to triage non-trivial reports within two weeks. Public disclosure (e.g. via a CVE or a security advisory) will be coordinated with the reporter.

## Out of Scope

- Vulnerabilities in third-party dependencies (Triton, LLVM, ROCm) should be reported to those projects directly.
- Performance-only issues (e.g. "I can construct an input that runs slowly") are not security issues — please open a regular issue.
