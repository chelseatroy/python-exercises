#!/usr/bin/env bash
set -euo pipefail

python docs/scripts/ai_fixer_loop.py --find find_problems_increase_code_coverage --fix fix_problem_increase_code_coverage
