#!/usr/bin/env bash
set -e

if [[ $# -lt 1 ]]; then
  echo "Usage: ./push.sh \"commit message\""
  exit 1
fi

MSG="$1"

git add -A
git commit -m "$MSG"
git push

echo "✅ pushed."