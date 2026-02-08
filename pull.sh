#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# sync_pull.sh
# Safe pull (rebase) with optional auto-stash.
#
# Usage:
#   ./sync_pull.sh
#   ./sync_pull.sh --no-stash
#   ./sync_pull.sh --ff-only
# ----------------------------

NO_STASH=0
FF_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --no-stash) NO_STASH=1 ;;
    --ff-only)  FF_ONLY=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not inside a git repo."
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
REMOTE="${REMOTE_NAME:-origin}"

echo "== Repo: $(basename "$(git rev-parse --show-toplevel)")"
echo "== Branch: $BRANCH"
echo "== Remote: $REMOTE"

# Check upstream
if ! git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  echo "⚠️  No upstream set for $BRANCH."
  echo "   You can set it once via:"
  echo "   git push -u $REMOTE $BRANCH"
  exit 1
fi

# Optional stash
STASHED=0
if [[ $NO_STASH -eq 0 ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "🧳 Working tree not clean. Stashing local changes..."
    git stash push -u -m "sync_pull auto-stash ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    STASHED=1
  fi
else
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "⚠️  Working tree not clean and --no-stash set. Pull may conflict."
  fi
fi

echo "🌐 Fetching..."
git fetch --prune "$REMOTE"

if [[ $FF_ONLY -eq 1 ]]; then
  echo "⬇️  Pulling (ff-only)..."
  git pull --ff-only "$REMOTE" "$BRANCH"
else
  echo "⬇️  Pulling (rebase)..."
  git pull --rebase "$REMOTE" "$BRANCH"
fi

# Restore stash
if [[ $STASHED -eq 1 ]]; then
  echo "📦 Restoring stashed changes..."
  if git stash pop; then
    echo "✅ Stash applied."
  else
    echo "⚠️  Stash pop had conflicts. Resolve conflicts, then:"
    echo "   git status"
    echo "   git add <files>"
    echo "   git rebase --continue   (if rebase in progress)"
    echo "   or simply commit after resolving."
    exit 1
  fi
fi

echo "✅ sync_pull done."
git status -sb