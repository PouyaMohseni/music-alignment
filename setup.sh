#!/usr/bin/env bash
# Server bootstrap: env, deps, MSMD download, full preprocessing (with audio),
# then a 10-step smoke train to confirm the pipeline runs end-to-end.
#
# Usage:
#   bash setup.sh                          # full setup, expects CUDA
#   CUDA=cpu       bash setup.sh           # CPU-only torch wheel
#   CUDA=cu118     bash setup.sh           # pick CUDA version
#   SKIP_SMOKE=1   bash setup.sh           # skip the 10-step smoke train
#   SKIP_DOWNLOAD=1 bash setup.sh          # MSMD zip already present
#   SF2=/path/to/piano.sf2 bash setup.sh   # bring your own soundfont
#
# Idempotent: re-running skips already-done work.

set -euo pipefail

CUDA="${CUDA:-cu121}"           # cu121 | cu118 | cpu
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
echo "==> repo: $REPO_ROOT"

# ---------------------------------------------------------------- 1. apt deps
if command -v apt-get >/dev/null 2>&1; then
  echo "==> installing system packages (fluidsynth + soundfont)"
  sudo apt-get update -qq
  sudo apt-get install -y -qq fluidsynth fluid-soundfont-gm unzip
fi
SF2="${SF2:-/usr/share/sounds/sf2/FluidR3_GM.sf2}"
if [[ ! -f "$SF2" ]]; then
  echo "!! soundfont not found at $SF2"
  echo "   either pass SF2=/path/to/your.sf2, or install fluid-soundfont-gm"
  exit 1
fi
echo "==> soundfont: $SF2"

# ---------------------------------------------------------------- 2. venv + pip
if [[ ! -d .venv ]]; then
  echo "==> creating venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
  echo "==> SKIP_INSTALL=1 — assuming deps already in .venv (no pip)"
else
  python -m pip install -q -U pip wheel
  echo "==> installing torch ($CUDA)"
  case "$CUDA" in
    cpu)
      pip install -q torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
      ;;
    cu118|cu121|cu124)
      pip install -q torch torchaudio torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
      ;;
    *)
      echo "unknown CUDA=$CUDA (use cpu | cu118 | cu121 | cu124)"; exit 1
      ;;
  esac
  echo "==> installing project requirements"
  pip install -q -r requirements.txt
fi

python - <<'PY'
import torch
print(f"   torch={torch.__version__}  cuda_available={torch.cuda.is_available()}"
      + (f"  device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY

# ---------------------------------------------------------------- 3. raw MSMD
MSMD_RAW=data/MSMD/msmd_aug_v1-1_no-audio
MSMD_ZIP=data/MSMD/msmd_aug_v1-1_no-audio.zip
mkdir -p data/MSMD
if [[ -d "$MSMD_RAW" ]]; then
  echo "==> MSMD already unzipped at $MSMD_RAW"
elif [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  echo "!! SKIP_DOWNLOAD=1 but $MSMD_RAW missing — aborting"; exit 1
else
  if [[ ! -f "$MSMD_ZIP" ]]; then
    echo "==> downloading MSMD (~9.5 GB) — this is the slow step"
    curl -L -C - -o "$MSMD_ZIP" \
      "https://zenodo.org/record/2597505/files/msmd_aug_v1-1_no-audio.zip?download=1"
  fi
  echo "==> unzipping MSMD"
  ( cd data/MSMD && unzip -q msmd_aug_v1-1_no-audio.zip )
fi

# ---------------------------------------------------------------- 4. CPJKU/msmd repo (for splits)
SPLITS=data/MSMD/msmd/msmd/splits/all_split.yaml
if [[ ! -f "$SPLITS" ]]; then
  echo "==> cloning CPJKU/msmd (just for splits/*.yaml)"
  git clone --depth 1 https://github.com/CPJKU/msmd.git data/MSMD/msmd
fi

# ---------------------------------------------------------------- 5. preprocessing
if [[ -f data/MSMD/processed/manifest.jsonl ]] \
   && [[ -f data/MSMD/processed/AndreJ__O34__andre-sonatine/audio.wav ]]; then
  echo "==> processed dataset already built (with audio) — skipping"
else
  echo "==> running msmd_prep.run_all (strips + annotations + audio synth)"
  python -m msmd_prep.run_all \
    --raw    "$MSMD_RAW" \
    --splits "$SPLITS" \
    --out    data/MSMD/processed \
    --sf2    "$SF2" \
    --jobs   "$(nproc 2>/dev/null || echo 4)"
fi

# ---------------------------------------------------------------- 6. smoke train
if [[ "${SKIP_SMOKE:-0}" == "1" ]]; then
  echo "==> SKIP_SMOKE=1 — skipping smoke train"
else
  echo "==> 10-step smoke training"
  python -m mymodel.v1_baseline.train \
    train.steps=10 \
    train.batch_size=2 \
    train.log_every=1 \
    train.eval_every=999 \
    train.ckpt_every=10
fi

echo
echo "==> all done"
echo "   - processed data:  data/MSMD/processed/"
echo "   - smoke checkpoint: results/v1_baseline/checkpoint_000010.pt (if smoke ran)"
echo
echo "real training:"
echo "  python -m mymodel.v1_baseline.train \\"
echo "    train.steps=20000 train.batch_size=16 \\"
echo "    data.num_workers=8 train.amp=true \\"
echo "    train.eval_every=500 train.ckpt_every=2000"
echo
echo "inference (after training):"
echo "  python -m mymodel.v1_baseline.infer \\"
echo "    --checkpoint results/v1_baseline/checkpoint_020000.pt \\"
echo "    --piece_id BachCPE__cpe-bach-rondo__cpe-bach-rondo"
echo
echo "full-test eval:"
echo "  python -m mymodel.v1_baseline.eval \\"
echo "    --checkpoint results/v1_baseline/checkpoint_020000.pt --split test"
