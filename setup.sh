#!/usr/bin/env bash
# Fetch the reference implementation under study at the exact commit measured,
# and expose it where the analysis scripts expect it.
#
# The upstream package is NOT vendored into this repository: it is third-party
# code, used unmodified, and pinning the commit is a stronger reproducibility
# guarantee than a copy that could silently drift.
set -euo pipefail

UPSTREAM_URL="https://github.com/SakanaAI/pc-alm.git"
UPSTREAM_COMMIT="660747f61a8a7e547c0ecd2c48c8883380a7d1f6"   # dated 2026-09-08

if [ ! -d pcalm-upstream ]; then
  git clone "$UPSTREAM_URL" pcalm-upstream
fi
git -C pcalm-upstream fetch --all --quiet
git -C pcalm-upstream checkout --quiet "$UPSTREAM_COMMIT"

# analysis/ imports `pcalm` and reads `configs/` by relative path
ln -sfn pcalm-upstream/pcalm  pcalm
ln -sfn pcalm-upstream/configs configs

echo "Upstream pinned at $(git -C pcalm-upstream rev-parse --short HEAD)"
echo "Verify it is unmodified:  git -C pcalm-upstream status --porcelain   (expect empty)"
echo
echo "Then create the environment:"
echo "  uv sync   # inside pcalm-upstream, or pip install jax numpy scipy"
