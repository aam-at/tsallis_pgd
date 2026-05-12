#!/usr/bin/env bash
# Download all pretrained model checkpoints for the segmentation attacks project.
#
# Sources:
#   - https://github.com/open-mmlab/mmsegmentation         (Standard segmentation checkpoints & configs)
#   - https://github.com/JIA-Lab-research/Robust-Semantic-Segmentation (DDCAT models + ResNet init)
#   - https://github.com/nmndeep/revisiting-at             (Robust ImageNet backbones)
#   - https://github.com/nmndeep/robust-segmentation      (PIRAT segmentation models)
#
# Requirements: wget, unzip, uvx (gdown is invoked via `uvx`)
#
# Usage:
#   ./scripts/download_models.sh [MODELS_DIR]
#   Default MODELS_DIR: <project_root>/models

set -euo pipefail
shopt -s globstar nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${1:-${PROJECT_ROOT}/models}"

echo "============================================="
echo " Downloading models to: $MODELS_DIR"
echo "============================================="

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

download() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    echo "  [SKIP] $(basename "$dest")"
    return 0
  fi
  echo "  [GET]  $(basename "$dest")"
  wget -q --show-progress --no-check-certificate -O "$dest" "$url" || {
    echo "  [FAIL] $(basename "$dest")"
    rm -f "$dest"
    return 1
  }
}

download_gdrive() {
  local file_id="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    echo "  [SKIP] $(basename "$dest")"
    return 0
  fi
  echo "  [GET]  $(basename "$dest") (Google Drive)"
  uvx --from "gdown==5.2.1" gdown --fuzzy "https://drive.google.com/uc?id=${file_id}" -O "$dest" || {
    echo "  [FAIL] $(basename "$dest")"
    rm -f "$dest"
    return 1
  }
}

download_nextcloud() {
  local share_url="$1"
  local dest="$2"
  download "${share_url}/download" "$dest"
}

section() {
  echo ""
  echo "---------------------------------------------"
  echo " $1"
  echo "---------------------------------------------"
}

iter_checkpoint_files() {
  local root_dir="$1"
  local checkpoint_file

  for checkpoint_file in "$root_dir"/**/*.pth; do
    [[ -f "$checkpoint_file" ]] || continue
    printf '%s\0' "$checkpoint_file"
  done
}

copy_ddcat_checkpoints() {
  local extracted_root="$1"
  local checkpoint_file rel_path dest copied=0

  while IFS= read -r -d '' checkpoint_file; do
    rel_path="${checkpoint_file#"$extracted_root"/}"

    if [[ "$rel_path" =~ (^|.*/)(cityscapes|voc2012)/(deeplabv3|pspnet)/(ddcat|no_defense|sat)/([^/]+\.pth)$ ]]; then
      dest="$DDCAT_DIR/${BASH_REMATCH[2]}/${BASH_REMATCH[3]}/${BASH_REMATCH[4]}/${BASH_REMATCH[5]}"
      if [[ ! -f "$dest" ]]; then
        cp -- "$checkpoint_file" "$dest"
        echo "  [COPY] ${BASH_REMATCH[2]}/${BASH_REMATCH[3]}/${BASH_REMATCH[4]}/${BASH_REMATCH[5]}"
        copied=$((copied + 1))
      fi
    fi
  done < <(iter_checkpoint_files "$extracted_root")

  if [[ "$copied" -eq 0 ]]; then
    echo "  [WARN] No DDCAT checkpoints were copied from the extracted archive."
  fi
}

print_model_summary() {
  local root_dir="$1"
  local model_file

  for model_file in \
    "$root_dir"/*.pth "$root_dir"/*.pt \
    "$root_dir"/*/*.pth "$root_dir"/*/*.pt \
    "$root_dir"/*/*/*.pth "$root_dir"/*/*/*.pt \
    "$root_dir"/*/*/*/*.pth "$root_dir"/*/*/*/*.pt; do
    [[ -f "$model_file" ]] || continue
    echo "  ${model_file#"$root_dir"/}"
  done | sort
}

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------

for cmd in wget unzip; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' is required but not found." >&2
    exit 1
  fi
done

if ! command -v uvx &>/dev/null; then
  echo "ERROR: 'uvx' is required (gdown is invoked via 'uvx')." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------------

mkdir -p "$MODELS_DIR"/{init_model,init_robust_model,mmsegmentation,pretrain_pirat}
mkdir -p "$MODELS_DIR"/pretrain_ddcat/cityscapes/deeplabv3/{ddcat,no_defense,sat}
mkdir -p "$MODELS_DIR"/pretrain_ddcat/cityscapes/pspnet/{ddcat,no_defense,sat}
mkdir -p "$MODELS_DIR"/pretrain_ddcat/voc2012/deeplabv3/{ddcat,no_defense,sat}
mkdir -p "$MODELS_DIR"/pretrain_ddcat/voc2012/pspnet/{ddcat,no_defense,sat}

# ===========================================================================
# 1. mmsegmentation/ — Standard segmentation checkpoints
#    Source: https://github.com/open-mmlab/mmsegmentation
# ===========================================================================
section "mmsegmentation/ (OpenMMLab checkpoints)"

MMSEG_DIR="$MODELS_DIR/mmsegmentation"
OPENMMLAB="https://download.openmmlab.com/mmsegmentation/v0.5"

# Note: config files are resolved from the installed mmseg package
# (`mmseg/.mim/configs`) at runtime and are not downloaded here.

# --- Checkpoints (.pth) ---

# DeepLabV3+
download "$OPENMMLAB/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug/deeplabv3plus_r50-d8_512x512_40k_voc12aug_20200613_161759-e1b43aa9.pth" \
  "$MMSEG_DIR/deeplabv3plus_r50-d8_512x512_40k_voc12aug_20200613_161759-e1b43aa9.pth"

download "$OPENMMLAB/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth" \
  "$MMSEG_DIR/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth"

download "$OPENMMLAB/deeplabv3plus/deeplabv3plus_r50b-d8_512x1024_80k_cityscapes/deeplabv3plus_r50b-d8_512x1024_80k_cityscapes_20201225_213645-a97e4e43.pth" \
  "$MMSEG_DIR/deeplabv3plus_r50b-d8_512x1024_80k_cityscapes_20201225_213645-a97e4e43.pth"

# FCN HRNet
download "$OPENMMLAB/hrnet/fcn_hr48_512x1024_160k_cityscapes/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth" \
  "$MMSEG_DIR/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth"

download "$OPENMMLAB/hrnet/fcn_hr48_512x512_40k_voc12aug/fcn_hr48_512x512_40k_voc12aug_20200613_222111-1b0f18bc.pth" \
  "$MMSEG_DIR/fcn_hr48_512x512_40k_voc12aug_20200613_222111-1b0f18bc.pth"

# PSPNet
download "$OPENMMLAB/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth" \
  "$MMSEG_DIR/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"

download "$OPENMMLAB/pspnet/pspnet_r50-d8_512x512_40k_voc12aug/pspnet_r50-d8_512x512_40k_voc12aug_20200613_161222-ae9c1b8c.pth" \
  "$MMSEG_DIR/pspnet_r50-d8_512x512_40k_voc12aug_20200613_161222-ae9c1b8c.pth"

# SegFormer
download "$OPENMMLAB/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth" \
  "$MMSEG_DIR/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth"

download "$OPENMMLAB/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth" \
  "$MMSEG_DIR/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth"

# ===========================================================================
# 2. pretrain_ddcat/ + init_model/ — DDCAT models & ImageNet ResNet backbones
#    Source: https://github.com/JIA-Lab-research/Robust-Semantic-Segmentation
# ===========================================================================
section "init_model/ (ImageNet ResNet backbones)"

INIT_DIR="$MODELS_DIR/init_model"

# Individual ResNet checkpoint files from Google Drive
# (downloading individually avoids re-traversing the entire folder tree)
download_gdrive "1w5pRmLJXvmQQA5PtCbHhZc_uC4o0YbmA" "$INIT_DIR/resnet50_v2.pth"
download_gdrive "1V-sfLnqSuwTPgNMrqUp4HDZpqtoBH4xm" "$INIT_DIR/resnet101_v2.pth"
download_gdrive "1pVzjWA1C-y4TniEkc06ygRO6e4Mec0wW" "$INIT_DIR/resnet152_v2.pth"

section "pretrain_ddcat/ (DDCAT adversarial training models)"

DDCAT_DIR="$MODELS_DIR/pretrain_ddcat"
DDCAT_ARCHIVE="$MODELS_DIR/pretrained_model.zip"

# Check if any DDCAT checkpoint is missing
DDCAT_COMPLETE=true
for ds in cityscapes voc2012; do
  for arch in deeplabv3 pspnet; do
    for defense in ddcat no_defense sat; do
      if [[ "$ds" == "cityscapes" ]]; then
        epoch=400
      else
        epoch=50
      fi
      if [[ ! -f "$DDCAT_DIR/$ds/$arch/$defense/train_epoch_${epoch}.pth" ]]; then
        DDCAT_COMPLETE=false
        break 3
      fi
    done
  done
done

if [[ "$DDCAT_COMPLETE" == true ]]; then
  echo "  [SKIP] All DDCAT checkpoints already exist"
else
  # This file must be downloaded manually from Google Drive (gdown fails on it).
  if [[ ! -f "$DDCAT_ARCHIVE" ]]; then
    echo ""
    echo "  [MANUAL] DDCAT archive must be downloaded manually."
    echo "           1. Open: https://drive.google.com/uc?id=120xLY_pGZlm3tqaLxTLVp99e06muBjJC"
    echo "           2. Save the file to: $DDCAT_ARCHIVE"
    echo "           3. Re-run this script."
    echo ""
  fi

  if [[ -f "$DDCAT_ARCHIVE" ]]; then
    echo "  [UNZIP] Extracting DDCAT archive..."
    # The archive may extract with a top-level directory; handle both cases.
    TMPDIR_DDCAT=$(mktemp -d)
    if UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -o "$DDCAT_ARCHIVE" -d "$TMPDIR_DDCAT"; then
      :
    else
      UNZIP_STATUS=$?
      if ! IFS= read -r -d '' _ < <(iter_checkpoint_files "$TMPDIR_DDCAT"); then
        echo "  [FAIL] Failed to extract DDCAT archive (unzip exit code: $UNZIP_STATUS)." >&2
        rm -rf "$TMPDIR_DDCAT"
        exit "$UNZIP_STATUS"
      fi
      echo "  [WARN] unzip reported exit code $UNZIP_STATUS, but checkpoint files were extracted; continuing."
    fi
    copy_ddcat_checkpoints "$TMPDIR_DDCAT"

    rm -rf "$TMPDIR_DDCAT"
    echo "  [DONE] DDCAT archive extracted"
  fi
fi

# ===========================================================================
# 3. init_robust_model/ — Robust ImageNet backbones
#    Source: https://github.com/nmndeep/revisiting-at
#    Hosted on NextCloud (Uni Tuebingen)
# ===========================================================================
section "init_robust_model/ (Robust ImageNet backbones)"

ROBUST_DIR="$MODELS_DIR/init_robust_model"

# ViT-S (single .pt file)
download \
  "https://nc.mlcloud.uni-tuebingen.de/public.php/dav/files/XLLnoCnJxp74Zqn/weights_vit_s.pt" \
  "$ROBUST_DIR/vit_s_robust.pt"

# ConvNext-S-CvSt (direct robust checkpoint file)
download \
  "https://nc.mlcloud.uni-tuebingen.de/public.php/dav/files/m3bAwNg4CJY4jrp/convnext_s_cvst_robust.pt" \
  "$ROBUST_DIR/convnext_s_cvst_robust.pt"

# ConvNext-T-CvSt (direct robust checkpoint file)
download \
  "https://nc.mlcloud.uni-tuebingen.de/public.php/dav/files/BFLoMrMdn8iBk7Y/convnext_tiny_cvst_robust.pt" \
  "$ROBUST_DIR/convnext_tiny_cvst_robust.pt"

# ===========================================================================
# 4. pretrain_pirat/ — PIRAT robust segmentation models
#    Source: https://github.com/nmndeep/robust-segmentation
#    Hosted on NextCloud (Uni Tuebingen)
# ===========================================================================
section "pretrain_pirat/ (PIRAT robust segmentation models)"

PIRAT_DIR="$MODELS_DIR/pretrain_pirat"

# Segmenter ViT-S (ADE20K)
download_nextcloud \
  "https://nc.mlcloud.uni-tuebingen.de/index.php/s/XF6Woa9G3eiGPig" \
  "$PIRAT_DIR/Segmenter_vit_s_robust.pth"

# UperNet ConvNeXt-T (PASCAL-VOC)
download_nextcloud \
  "https://nc.mlcloud.uni-tuebingen.de/index.php/s/zSFgoAngcm47FZm" \
  "$PIRAT_DIR/UperNet_ConvNext_T_VOC.pth"

# UperNet ConvNeXt-S (PASCAL-VOC)
download_nextcloud \
  "https://nc.mlcloud.uni-tuebingen.de/index.php/s/MBXnMd5QKztmZaa" \
  "$PIRAT_DIR/UperNet_ConvNext_S_VOC.pth"

# UperNet ConvNeXt-T (ADE20K)
download_nextcloud \
  "https://nc.mlcloud.uni-tuebingen.de/index.php/s/ACMQRiyfyXboXwT" \
  "$PIRAT_DIR/UperNet_ConvNext_T_ADE.pth"

# UperNet ConvNeXt-S (ADE20K)
download_nextcloud \
  "https://nc.mlcloud.uni-tuebingen.de/index.php/s/Smogk2BWbfMxkyo" \
  "$PIRAT_DIR/UperNet_ConvNext_S_ADE.pth"

# ===========================================================================
# Summary
# ===========================================================================
section "Download complete"

echo ""
echo "Model directory structure:"
echo ""
if command -v tree &>/dev/null; then
  tree -L 3 "$MODELS_DIR" --filelimit 20
else
  print_model_summary "$MODELS_DIR"
fi

# Check for missing files
echo ""
MISSING=0
for f in \
  "init_model/resnet50_v2.pth" \
  "init_model/resnet101_v2.pth" \
  "init_model/resnet152_v2.pth" \
  "init_robust_model/vit_s_robust.pt" \
  "init_robust_model/convnext_s_cvst_robust.pt" \
  "init_robust_model/convnext_tiny_cvst_robust.pt" \
  "pretrain_pirat/Segmenter_vit_s_robust.pth" \
  "pretrain_pirat/UperNet_ConvNext_T_VOC.pth" \
  "pretrain_pirat/UperNet_ConvNext_S_VOC.pth" \
  "pretrain_pirat/UperNet_ConvNext_T_ADE.pth" \
  "pretrain_pirat/UperNet_ConvNext_S_ADE.pth"; do
  if [[ ! -f "$MODELS_DIR/$f" ]]; then
    echo "  [MISSING] $f"
    MISSING=$((MISSING + 1))
  fi
done

if [[ "$MISSING" -eq 0 ]]; then
  echo "  All model files present!"
else
  echo ""
  echo "  $MISSING file(s) missing. See warnings above."
fi
