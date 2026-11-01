#!/usr/bin/env bash
# Clone RETFound (backbone rétinien préentraîné) dans third_party/.
set -euo pipefail

mkdir -p third_party
cd third_party

if [ ! -d retfound ]; then
  git clone https://github.com/rmaphoh/RETFound_MAE.git retfound
fi

echo "Poids préentraînés : python scripts/download_retfound_checkpoint.py --out third_party/retfound/checkpoints"
echo "(dépôt HuggingFace gated, voir README)."
