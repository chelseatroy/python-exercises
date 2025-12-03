#!/usr/bin/env bash
set -euo pipefail

python docs/scripts/ai_command_loop.py --find increase_coverage_find_untested_code.sh --fix increase_coverage_fix_add_one_test.sh
