#!/usr/bin/env bash
# Clone RETFound (backbone rétinien préentraîné) dans third_party/.
set -euo pipefail

mkdir -p third_party
cd third_party

if [ ! -d retfound ]; then
  git clone https://github.com/rmaphoh/RETFound_MAE.git retfound
fi

echo "Récupérer les poids préentraînés (CFP/OCT) selon les instructions de"
echo "third_party/retfound/README.md et les placer sous third_party/retfound/checkpoints/"
