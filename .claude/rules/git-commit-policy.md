---
root: false
targets: ["*"]
description: Git commit policy
globs: []
---

# Git Commit Policy

## Pre-commit hooks

If the project has pre-commit hooks configured (via `.git/hooks/pre-commit`, the `pre-commit` Python tool, husky, or any other mechanism), all commits MUST pass them.

- NEVER use `--no-verify`.
- NEVER set environment variables that disable hooks (e.g. `HUSKY=0`, `SKIP=hook-id`).
- If a hook fails, fix the underlying issue. Do not bypass it.

## GPG signing (when configured)

If the project or developer has `commit.gpgsign = true` configured, all commits MUST remain signed.

- NEVER use `--no-gpg-sign`.
- NEVER override with `-c commit.gpgsign=false`.
- If signing fails, diagnose the GPG agent or key configuration. Do not bypass.

## Commit messages

- Write a concise subject line (under ~72 chars) describing the change.
- Use the imperative mood (`add`, `fix`, `update`), not past tense.
- Add a body when the change is non-obvious. Explain the why, not the what.
- Do not include AI-attribution co-author lines unless the user explicitly asks.
