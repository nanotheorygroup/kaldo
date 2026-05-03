#!/bin/bash
# Claude Code PreToolUse hook: block bypass flags on git commit commands.
# Prevents the AI from circumventing any pre-commit hooks the project has configured.
#
# Exit codes: 0 = allow, 2 = block (Claude Code convention)

# Block by default on unexpected errors
trap 'echo "BLOCKED: Unexpected error in pre-commit-check hook." >&2; exit 2' ERR

# If jq is not available, we cannot parse the command. Allow it through.
if ! command -v jq &>/dev/null; then
  exit 0
fi

COMMAND=$(jq -r '.tool_input.command // empty')

# Only intercept commands containing "git commit" (accounts for git -c flags before commit)
if ! echo "$COMMAND" | grep -qE 'git\s.*commit(\s|$|")'; then
  exit 0
fi

# Block --no-verify (skips all git hooks)
if echo "$COMMAND" | grep -qE -- '--no-verify'; then
  echo "BLOCKED: --no-verify is not allowed. Pre-commit hooks must run." >&2
  exit 2
fi

# Block --no-gpg-sign (skips GPG signing if configured)
if echo "$COMMAND" | grep -qE -- '--no-gpg-sign'; then
  echo "BLOCKED: --no-gpg-sign is not allowed. Commits must remain signed." >&2
  exit 2
fi

# Block -c commit.gpgsign=false (disables signing via config override)
if echo "$COMMAND" | grep -qiE -- '-c\s+.?commit\.gpgsign\s*=\s*false'; then
  echo "BLOCKED: Disabling commit.gpgsign is not allowed." >&2
  exit 2
fi

exit 0
