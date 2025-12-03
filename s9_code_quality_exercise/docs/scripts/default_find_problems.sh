#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

output=$(./run_tests.sh 2>&1)
echo "$output"

total_coverage=$(echo "$output" | grep "^TOTAL" | awk '{print $NF}' | sed 's/%//')

if [ -z "$total_coverage" ]; then
    echo "Could not parse coverage percentage"
    exit 1
fi

if (( $(echo "$total_coverage < 99" | bc -l) )); then
    exit 0
else
    exit 1
fi
