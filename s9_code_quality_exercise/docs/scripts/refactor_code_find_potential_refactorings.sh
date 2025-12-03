#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

claude --dangerously-skip-permissions < docs/scripts/refactor_code_identify_potential_refactors.md

if [ -f "__potential_refactorings.md" ]; then
    exit 0
else
    exit 1
fi
