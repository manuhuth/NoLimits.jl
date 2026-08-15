#!/usr/bin/env bash
# Run the test suite as parallel NL_TEST_GROUP shards (the same split CI uses),
# so a full local run costs the slowest group instead of the sequential sum.
# Each group is its own julia process (a few GB RAM each; plan ~7 concurrent).
#
# Usage: test/parallel.sh          # all groups
#        test/parallel.sh 4 5     # only groups 4 and 5
set -u
cd "$(dirname "$0")/.."

# Group count = TEST_GROUPS elements in runtests.jl (each opens with `[` or `["...`).
total=$(grep -cE '^\s+\[($|")' test/runtests.jl)
groups=("$@")
[ ${#groups[@]} -eq 0 ] && groups=($(seq 1 "$total"))

logdir=$(mktemp -d)
echo "logs: $logdir"
pids=()
for g in "${groups[@]}"; do
    NL_TEST_GROUP=$g julia --color=yes --project -e \
        'using Pkg; Pkg.test(; julia_args=["--check-bounds=auto"])' \
        >"$logdir/group$g.log" 2>&1 &
    pids+=($!)
done

fail=0
for i in "${!groups[@]}"; do
    if wait "${pids[$i]}"; then
        echo "group ${groups[$i]}: PASS"
    else
        echo "group ${groups[$i]}: FAIL  (log: $logdir/group${groups[$i]}.log)"
        fail=1
    fi
done
exit $fail
