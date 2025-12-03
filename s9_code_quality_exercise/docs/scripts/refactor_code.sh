#!/usr/bin/env bash
set -euo pipefail

python docs/scripts/ai_command_loop.py --find find_potential_refactorings --fix perform_one_refactor --tcr tcr_refactor_code
