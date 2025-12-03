#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

claude --dangerously-skip-permissions < docs/scripts/increase_coverage.md
