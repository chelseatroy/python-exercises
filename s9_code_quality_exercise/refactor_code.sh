#!/usr/bin/env bash
set -euo pipefail

python docs/scripts/ai_fixer_loop.py --find find_problems_refactor_code --fix fix_problem_refactor_code --tcr tcr_refactor_code
