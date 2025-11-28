#!/usr/bin/env bash

set -euo pipefail

# Usage: ./gitr.sh "commit message here"
# Commits all changes and pushes to the CURRENT BRANCH.
# If the push is rejected (non-fast-forward), it will pull --rebase and retry.

msg=${1:-"chore: update"}

# Ensure we're inside a git repo and move to repo root
repo_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "${repo_root}" ]]; then
    echo "❌ Not a git repository."
    exit 1
fi
cd "$repo_root"

branch=$(git rev-parse --abbrev-ref HEAD)
remote="origin"

echo "📦 Repo: $repo_root"
echo "🌿 Branch: $branch"

echo "➕ Adding all changes..."
git add -A

if git diff --cached --quiet; then
    echo "ℹ️  No staged changes to commit."
else
    echo "📝 Committing with message: '$msg'..."
    git commit -m "$msg" || true
fi

# Check if upstream is set
if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
    upstream_set=true
else
    upstream_set=false
fi

push_once() {
    if $upstream_set; then
        git push "$remote" "$branch"
    else
        git push -u "$remote" "$branch"
    fi
}

echo "🚀 Pushing to $remote/$branch..."
if push_once; then
    echo "✅ Pushed successfully."
    exit 0
fi

echo "⚠️  Push failed. Attempting 'git pull --rebase --autostash' and retry..."
if git pull --rebase --autostash "$remote" "$branch"; then
    if push_once; then
        echo "✅ Rebased and pushed successfully."
        exit 0
    fi
fi

echo "❌ Push still failing. You may need to resolve conflicts manually:"
echo "   1) git status"
echo "   2) Fix conflicts"
echo "   3) git rebase --continue"
echo "   4) git push"
exit 1
