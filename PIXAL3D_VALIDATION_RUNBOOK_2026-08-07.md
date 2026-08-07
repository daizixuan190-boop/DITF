# Pixal3D Minimum-Cost Validation Runbook (2026-08-07)

## Fixed question and stopping gates

This run does not train a matcher and does not test a fusion rule. It asks
whether frozen PartField identity, transported through Pixal3D's exact raw
surface interface, separates the PCK-valid member of an existing FLUX
attention top-20.

Run gates in order. Never start the next gate before the previous one passes.

1. One difficult chair image: foreground selection, projection, triangle id,
   barycentrics, raw topology and elapsed time.
2. The same image only: PartField vertex-row equality and finite 448-D output.
3. Frozen adversarial 18-pair smoke, with raw PartField cosine only.
4. Heldout20 only if smoke passes: strict residual top-1 and
   `current union teacher` oracle.

The source cannot support a 75-point system if heldout
`current union teacher < 0.75`. Do not design distillation unless it is at
least `0.77`; require strict-current-residual top-1 at least `0.45`.

## Pinned upstream revisions

```text
Pixal3D   cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af
TRELLIS.2 75fbf0183001ed9876c8dbb35de6b68552ee08bd
PartField 373025dbd283bb44cc4a6dc78c99994dbc91de32
```

## Gate 0: no-card downloads

Do this before renting the A800. The Pixal3D repository is about 24 GB and
the PartField checkpoint about 1.24 GB. Keep at least 45 GB free on the
partition containing `HF_HOME`.

```bash
source /root/miniconda3/etc/profile.d/conda.sh

VALIDATION_ROOT=/root/autodl-tmp/pixal3d_validation
mkdir -p "$VALIDATION_ROOT/repos" "$VALIDATION_ROOT/models" "$VALIDATION_ROOT/hf_cache"
df -h /root/autodl-tmp

git clone https://github.com/TencentARC/Pixal3D.git "$VALIDATION_ROOT/repos/Pixal3D"
git -C "$VALIDATION_ROOT/repos/Pixal3D" checkout cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af

git clone --recursive https://github.com/microsoft/TRELLIS.2.git "$VALIDATION_ROOT/repos/TRELLIS.2"
git -C "$VALIDATION_ROOT/repos/TRELLIS.2" checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C "$VALIDATION_ROOT/repos/TRELLIS.2" submodule update --init --recursive

git clone https://github.com/nv-tlabs/PartField.git "$VALIDATION_ROOT/repos/PartField"
git -C "$VALIDATION_ROOT/repos/PartField" checkout 373025dbd283bb44cc4a6dc78c99994dbc91de32

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$VALIDATION_ROOT/hf_cache"
HFCLI=/root/miniconda3/envs/DiTF/bin/huggingface-cli

"$HFCLI" download TencentARC/Pixal3D \
  --local-dir "$VALIDATION_ROOT/models/Pixal3D" \
  --max-workers 2

mkdir -p "$VALIDATION_ROOT/repos/PartField/model"
"$HFCLI" download mikaelaangel/partfield-ckpt model_objaverse.ckpt \
  --local-dir "$VALIDATION_ROOT/repos/PartField/model" \
  --max-workers 1

du -sh "$VALIDATION_ROOT/models/Pixal3D" \
  "$VALIDATION_ROOT/repos/PartField/model/model_objaverse.ckpt"
```

If the mirror returns a repository error for a public model, repeat only the
failed download after `unset HF_ENDPOINT`; do not redownload completed files.

## Gate 1A: A800 environment build

TRELLIS.2's official setup refuses no-card mode because it detects the GPU.
Start billing only after Gate 0 completes. SDPA is officially supported by
Pixal3D, so this omits the optional flash-attn build.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
VALIDATION_ROOT=/root/autodl-tmp/pixal3d_validation

cd "$VALIDATION_ROOT/repos/TRELLIS.2"
. ./setup.sh --new-env --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

conda activate trellis2
cd "$VALIDATION_ROOT/repos/Pixal3D"
python -m pip install -r requirements.txt
NATTEN_CUDA_ARCH="8.0" NATTEN_N_WORKERS=4 \
  python -m pip install natten==0.21.0 --no-build-isolation
python -m pip install \
  https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

python - <<'PY'
import torch
import natten
import nvdiffrast.torch
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0), "natten": natten.__version__})
PY
```

If any command fails, stop there and return the complete error. Do not install
alternative package versions ad hoc.

## Gate 1B: one-image interface probe

The selected image is deliberately hard: an occluded chair with a person.
This cheaply detects whether category-free foreground removal reconstructs
the wrong object.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

REPO=/root/autodl-tmp/workspace/DiTF-official
VALIDATION_ROOT=/root/autodl-tmp/pixal3d_validation
IMAGE=/root/autodl-tmp/datasets/SPair-71k/JPEGImages/chair/2008_006733.jpg
PROBE_DIR="$REPO/analysis/pixal3d_assets/chair/2008_006733"
LOG="$PROBE_DIR/probe.log"

mkdir -p "$PROBE_DIR"
export HF_HOME="$VALIDATION_ROOT/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export ATTN_BACKEND=sdpa
export OMP_NUM_THREADS=8

cd "$REPO"
/usr/bin/time -v python probe_pixal3d_surface_interface.py \
  --pixal3d_repo "$VALIDATION_ROOT/repos/Pixal3D" \
  --model_path "$VALIDATION_ROOT/models/Pixal3D" \
  --image "$IMAGE" \
  --output_dir "$PROBE_DIR" \
  --resolution 1024 \
  --seed 42 \
  > "$LOG" 2>&1

tail -n 100 "$LOG"
ls -lh "$PROBE_DIR"
```

Do not run PartField yet. Return these three artifacts for review:

```text
metadata.json
foreground_mask.png
projection_overlay.png
```

Immediate stop conditions: intended chair is not the selected foreground;
overlay is mirrored/rotated/strongly offset; any automatic invariant fails;
or one-image time makes 36-image smoke unaffordable.

## Gate 2: same-mesh PartField interface

Run only after Gate 1 is reviewed as PASS. Create the official PartField
environment in no-card mode if possible; its released environment is kept
separate because it pins PyTorch 2.4 while Pixal3D pins PyTorch 2.6.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
VALIDATION_ROOT=/root/autodl-tmp/pixal3d_validation
cd "$VALIDATION_ROOT/repos/PartField"
conda env create -f environment.yml
```

Then on A800 run exactly one raw mesh, with mesh preprocessing disabled and
the official correspondence-mode vertex representation enabled:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate partfield

REPO=/root/autodl-tmp/workspace/DiTF-official
VALIDATION_ROOT=/root/autodl-tmp/pixal3d_validation
PROBE_DIR="$REPO/analysis/pixal3d_assets/chair/2008_006733"
PARTFIELD="$VALIDATION_ROOT/repos/PartField"

cd "$PARTFIELD"
/usr/bin/time -v python partfield_inference.py \
  -c configs/final/demo.yaml \
  --opts \
  continue_ckpt "$PARTFIELD/model/model_objaverse.ckpt" \
  result_name partfield_features/pixal_probe_chair_2008_006733 \
  dataset.data_path "$PROBE_DIR" \
  preprocess_mesh False \
  vertex_feature True \
  > "$PROBE_DIR/partfield.log" 2>&1

FEATURES="$PARTFIELD/exp_results/partfield_features/pixal_probe_chair_2008_006733/part_feat_raw_mesh_0_batch.npy"
python "$REPO/verify_partfield_vertex_features.py" \
  --probe_dir "$PROBE_DIR" \
  --partfield_features "$FEATURES"

tail -n 100 "$PROBE_DIR/partfield.log"
ls -lh "$PROBE_DIR/vertex_features.npy" "$PROBE_DIR/metadata.json"
```

Gate 3 manifests and batch commands are intentionally withheld until the
one-image foreground/projection and PartField row contracts pass. This
prevents paying for 36 reconstructions when the deployment interface itself
is invalid.
