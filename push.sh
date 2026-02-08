#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./push.sh \"commit message\""
  exit 1
fi

MSG="$1"
REMOTE="${REMOTE_NAME:-origin}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Ensure we're in a git repo
git rev-parse --is-inside-work-tree >/dev/null

echo "== Repo:   $(basename "$(git rev-parse --show-toplevel)")"
echo "== Branch: $BRANCH"
echo "== Remote: $REMOTE"

# Stage everything (including new files)
git add -A

# If nothing staged, exit gracefully
if git diff --cached --quiet; then
  echo "⚠️  Nothing to commit."
  git status -sb
  exit 0
fi

git commit -m "$MSG"
git push "$REMOTE" "$BRANCH"

echo "✅ pushed."