#!/usr/bin/env bash
set -euo pipefail

# Generate a single markdown file containing the project's code (excluding data/out/runs)
OUT_FILE="tickfirecode.md"

echo "# Project Code Dump" > "$OUT_FILE"
echo >> "$OUT_FILE"
echo "This is the project's code. Analyze thoroughly and do not make suggestions right away." >> "$OUT_FILE"
echo "Acknowledge when seen with: seen" >> "$OUT_FILE"
echo >> "$OUT_FILE"

echo "Scanning repository for code files..." >&2

# Directories to exclude (paths relative to repo root)
EXCLUDE_DIRS=("./data" "./out" "./runs" "./.git" "./node_modules" "./.venv" "./venv")

# Build find prune args
PRUNE_ARGS=()
for d in "${EXCLUDE_DIRS[@]}"; do
  PRUNE_ARGS+=( -path "$d" -prune -o )
done

# Regex for file extensions considered "code"
EXT_REGEX='.*\.(py|sh|js|ts|go|rs|java|c|cpp|h|hpp|rb|kt|scala|ipynb|yaml|yml|toml)$'

# Also include Makefile and Dockerfile

# Use a single find invocation with explicit prune list (avoids eval / quoting issues)
# Build a prune expression string for find
PRUNE_EXPR=""
for d in "${EXCLUDE_DIRS[@]}"; do
  PRUNE_EXPR+=" -path $d -o"
done
# remove trailing -o
PRUNE_EXPR=${PRUNE_EXPR% -o}

# Run find: prune dirs, then select files by regex or by specific names
find . \( $PRUNE_EXPR \) -prune -o -type f \( -regextype posix-extended -regex "$EXT_REGEX" -o -name Makefile -o -name Dockerfile \) -print | sort | while read -r f; do
  # normalize path (strip leading ./)
  nf="${f#./}"
  echo "---" >> "$OUT_FILE"
  echo "## File: $nf" >> "$OUT_FILE"
  echo >> "$OUT_FILE"
  echo '```' >> "$OUT_FILE"
  # Use sed to ensure binary files (like large JSON notebooks) don't break the output
  if [[ "$nf" == *.ipynb ]]; then
    # For notebooks, print a short header and the raw JSON (keep manageable)
    echo "(notebook JSON)" >> "$OUT_FILE"
    sed -n '1,400p' "$nf" >> "$OUT_FILE" || true
  else
    sed -n '1,20000p' "$nf" >> "$OUT_FILE" || true
  fi
  echo '```' >> "$OUT_FILE"
  echo >> "$OUT_FILE"
done

echo "tickfirecode.md generated at: $OUT_FILE" >&2
