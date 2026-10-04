#!/usr/bin/env bash
# Sets up TruFor OUTSIDE this repo (default ~/vendor/TruFor).
# TruFor is licensed for non-profit use only: never commit its code or weights here.
set -euo pipefail
VENDOR="${VENDOR:-$HOME/vendor}"
mkdir -p "$VENDOR" && cd "$VENDOR"
[ -d TruFor ] || git clone --depth 1 https://github.com/grip-unina/TruFor.git
cd TruFor/test_docker
if [ ! -f weights/trufor.pth.tar ]; then
  wget -c https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip -O src/TruFor_weights.zip
  echo "7bee48f3476c75616c3c5721ab256ff8  src/TruFor_weights.zip" | md5sum -c
  (cd src && unzip -q -n TruFor_weights.zip && rm TruFor_weights.zip)
  mv src/weights ./weights
fi
# PyTorch >= 2.6 needs weights_only=False for this checksum-verified checkpoint
grep -q "weights_only=False" src/trufor_test.py || sed -i 's/map_location=torch.device(device))/map_location=torch.device(device), weights_only=False)/' src/trufor_test.py
pip install yacs tqdm
echo "TruFor ready at $VENDOR/TruFor/test_docker/src"
