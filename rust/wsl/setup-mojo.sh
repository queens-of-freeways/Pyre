#!/bin/bash
# rootless Mojo toolchain setup in WSL (Ubuntu 26.04)
set -e
cd ~
if [ ! -x ~/.local/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/mojo-env --python 3.12 2>/dev/null || uv venv ~/mojo-env
VIRTUAL_ENV="$HOME/mojo-env" uv pip install mojo --index https://whl.modular.com/simple/ --prerelease allow
~/mojo-env/bin/mojo --version
