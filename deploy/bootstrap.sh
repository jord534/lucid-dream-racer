#!/usr/bin/env bash
# Set up a fresh Ubuntu VM (Scaleway or any provider) to run this project.
# Not yet run on a real VM: expect to fix small things on first use.
#
#   git clone https://github.com/jord534/lucid-dream-racer.git && bash lucid-dream-racer/deploy/bootstrap.sh
set -euo pipefail

sudo apt-get update
sudo apt-get install -y git build-essential swig tmux rsync

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."
uv venv --python 3.12 .venv
grep -v '^-e ' requirements.lock > /tmp/requirements.txt          # drop the local editable path
uv pip install --python .venv/bin/python -r /tmp/requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .

export SDL_VIDEODRIVER=dummy                                      # no display on a server
.venv/bin/python -m pytest -q
echo "bootstrap done. Next: copy data/latents.npz and runs/ from your laptop (see the deployment steps)."
