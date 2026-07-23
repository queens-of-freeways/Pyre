#!/usr/bin/env bash
set -e

echo "==> Uninstalling Pyre..."

# 1. Remove the pip-installed package and entry point
pip uninstall pyre -y 2>/dev/null || true
rm -f ~/.local/bin/pyre
echo "  ✓ Entry point removed"

# 2. Remove the pixi environment
if [ -d .pixi ]; then
    rm -rf .pixi
    echo "  ✓ Pixi environment removed"
fi

# 3. Remove the project directory
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd /tmp || exit 1
rm -rf "$PROJECT_DIR"
echo "  ✓ Project directory removed"
echo ""
echo "Pyre fully uninstalled."
