#!/bin/bash
# Runs each test file listed in runtests.jl in parallel, saves output to /tmp/nl_test_results/
set -euo pipefail

PROJ=/Users/manuel/Documents/LongitudinalData.jl/NoLimits
OUT=/tmp/nl_test_results
mkdir -p "$OUT"

# Test files derived from runtests.jl's TEST_FILES (single source of truth).
TESTS=($(sed -n '/^const TEST_FILES/,/^\]/p' "$PROJ/test/runtests.jl" | grep -oE '"[^"]+\.jl"' | tr -d '"'))

run_test() {
  local t="$1"
  local logfile="$OUT/${t%.jl}.log"
  local statusfile="$OUT/${t%.jl}.status"

  if julia --project="$PROJ" -e "
    using Test
    using NoLimits
    include(\"$PROJ/test/$t\")
  " > "$logfile" 2>&1; then
    echo "PASS" > "$statusfile"
    echo "PASS: $t"
  else
    echo "FAIL" > "$statusfile"
    echo "FAIL: $t"
  fi
}

export -f run_test
export PROJ OUT

# Run all tests in parallel (max 8 at a time to avoid resource exhaustion)
printf '%s\n' "${TESTS[@]}" | xargs -P 8 -I {} bash -c 'run_test "$@"' _ {}

echo ""
echo "=== SUMMARY ==="
PASSED=0
FAILED=0
for t in "${TESTS[@]}"; do
  statusfile="$OUT/${t%.jl}.status"
  if [ -f "$statusfile" ]; then
    status=$(cat "$statusfile")
    if [ "$status" = "PASS" ]; then
      PASSED=$((PASSED+1))
    else
      FAILED=$((FAILED+1))
      echo "FAILED: $t"
    fi
  else
    echo "NO STATUS: $t"
    FAILED=$((FAILED+1))
  fi
done
echo "Passed: $PASSED / $((PASSED+FAILED))"
