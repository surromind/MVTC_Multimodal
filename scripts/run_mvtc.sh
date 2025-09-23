#!/usr/bin/env bash
set -euo pipefail

# Determine project root relative to this script
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"

python -m main "$@"
