#!/usr/bin/env bash
set -euo pipefail

pip install coverage > /dev/null
coverage run -m pytest tests/
coverage report
coverage html
