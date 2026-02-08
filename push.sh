#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./push.sh \"commit message\""
  exit 1
fi

MSG="$1"
REMOTE="${REMOTE_NAME:-origin}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# 1) stage everything (including new files)
git add -A

# 2) commit if there is anything staged
if ! git diff --cached --quiet; then
  git commit -m "$MSG"
fi

# 3) push if we have commits ahead (or if a new commit was created)
AHEAD="$(git rev-list --count @{u}..HEAD 2>/dev/null || echo 0)"
if [[ "$AHEAD" != "0" ]]; then
  git push "$REMOTE" "$BRANCH"
else
  echo "Nothing to push."
fi

git status -sb