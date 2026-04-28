# TsallisPGD

**TsallisPGD: Adaptive Gradient Weighting for Adversarial Attacks on Semantic Segmentation**

Alexander Matyasko, Xin Lou, Indriyati Atmosukarto, Wei Zhang — *Singapore Institute of Technology*
IJCNN 2026 (camera-ready) · [arXiv (coming soon)](https://arxiv.org/abs/TBD)

This repository contains the reference implementation of **TsallisPGD** along with the full evaluation harness used in the paper, covering Cityscapes, Pascal VOC, and ADE20K against both standard and adversarially trained segmentation models.

## Abstract

Attacking semantic segmentation models is significantly harder than image classification models because an attacker must flip thousands of pixel predictions simultaneously. Standard pixel-wise cross-entropy (CE) is ill-suited to this setting: it tends to overemphasize already-misclassified pixels, which slows optimization and overstates model robustness. To address these issues, we introduce TsallisPGD, an adversarial attack built on the Tsallis cross-entropy, a generalization of CE parameterized by $q$, which adaptively reshapes the gradient landscape by controlling gradient concentration across pixels. By varying $q$, we steer the attack toward pixels at different confidence levels. We first show that no single fixed-$q$ is universally optimal, as its effectiveness depends on the dataset, model architecture, and perturbation budget. Motivated by this, we propose a dynamic $q$-schedule that sweeps $q$ during optimization. Extensive experiments on Cityscapes, Pascal VOC, and ADE20K show that TsallisPGD, using a single validation-selected schedule, achieves the best average attack rank across all evaluated settings and improves over CEPGD, SegPGD, CosPGD, JSPGD, and MaskedPGD in reducing accuracy and mIoU on both standard and robust models.

## Overview

Adversarial attacks on semantic segmentation must flip thousands of pixel predictions at once, and the standard pixel-wise cross-entropy (CE) objective wastes gradient mass on already-misclassified pixels — leading to slow optimization and overestimated robustness.

**TsallisPGD** replaces pixel-wise CE with the **Tsallis cross-entropy**, a generalization of CE controlled by a parameter $q$. The Tsallis objective induces a confidence-dependent reweighting in gradient space — each pixel is effectively scaled by $p_y^{\,1-q}$, where $p_y$ is the predicted probability of the ground-truth class. By tuning $q$, the attack can target pixels at different confidence levels:

- $q = 1$ recovers standard cross-entropy.
- $q < 1$ emphasizes high-confidence, correctly classified pixels and down-weights already-broken pixels.

Because no single fixed $q$ is optimal across datasets, architectures, and perturbation budgets, the paper introduces a **dynamic linear $q$-schedule** that sweeps $q$ over the course of the attack. A single validation-selected schedule ($q: -2 \to 1$) is used unchanged across all datasets, models, and budgets in the main benchmark.

Across 21 settings spanning Cityscapes, Pascal VOC, and ADE20K, TsallisPGD achieves the best average attack rank in both pixel accuracy and mIoU among CEPGD, SegPGD, CosPGD, JSPGD, and MaskedPGD, with the largest gains on adversarially trained models.

## Repository Status

Research code released alongside the camera-ready paper. The Hydra-based attack and evaluation harness has been used to produce all results reported in the paper. Linux + CUDA only.

## Setup

The project is managed with [Pixi](https://pixi.sh). Install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then bootstrap the project:

```bash
git clone https://github.com/aam-at/tsallis_pgd.git
cd tsallis_pgd
pixi run setup
```

`pixi run setup` installs Python dependencies (including a CUDA build of `mmcv`) and downloads the pretrained checkpoints needed to reproduce the paper.

### Data

Downloads default to `$HOME/data` (override with `DATA_ROOT`):

```bash
pixi run download-data-pascal-voc-aug   # Pascal VOC + augmented split
pixi run download-data-ade20k           # ADE20K
pixi run download-data-cityscapes       # Cityscapes (prompts for credentials)
```

### Requirements

- Linux x86-64 with an NVIDIA GPU
- CUDA 13.0 toolchain (provisioned by Pixi)
- Python 3.13 (provisioned by Pixi)

## Running TsallisPGD

The main entry point is `experiments/test.py`, a Hydra application. The TsallisPGD attack used in the paper is the dynamic-schedule variant `auto_multi_pgd_tsallis_ce_adaptive`, which combines APGD step-size selection, a 300-iteration budget, a single random start, and the multi-`ε` trick.

### TsallisPGD on Pascal VOC (UPerNet ConvNeXt-T, PIR-AT, $\epsilon_\infty = 8/255$)

```bash
pixi run python experiments/test.py \
  -cn test_voc2012 \
  model=convnext_t_cvst_robust_voc2012 \
  attack=auto_multi_pgd_tsallis_ce_adaptive \
  attack.base_epsilon=0.031372549 \
  attack.base_iterations=300 \
  task_name=tsallispgd_voc_demo
```

### TsallisPGD on ADE20K (UPerNet ConvNeXt-T, PIR-AT, $\epsilon_\infty = 8/255$)

```bash
pixi run python experiments/test.py \
  -cn test_ade20k \
  model=convnext_t_cvst_robust_ade20k \
  attack=auto_multi_pgd_tsallis_ce_adaptive \
  attack.base_epsilon=0.031372549 \
  attack.base_iterations=300 \
  task_name=tsallispgd_ade20k_demo
```

### TsallisPGD on Cityscapes (PSPNet ResNet-50, DDCAT, $\epsilon_\infty = 0.5/255$)

```bash
pixi run python experiments/test.py \
  -cn test_cityscapes \
  model=pspnet_cityscapes_ddcat \
  attack=auto_multi_pgd_tsallis_ce_adaptive \
  attack.base_epsilon=0.001960784 \
  attack.base_iterations=300 \
  data.normalize=false \
  data.crop_size=449 \
  data.info.ignore_index=255 \
  task_name=tsallispgd_cityscapes_demo
```

Outputs (logs, metrics, and optional adversarial images/predictions) are written to `runs/{task_name}/...`.

### Selecting the schedule

The default `auto_multi_pgd_tsallis_ce_adaptive` config uses the validation-selected linear sweep $q: -2 \to 1$. To override the endpoints:

```bash
... attack=auto_multi_pgd_tsallis_ce_adaptive \
    loss.params.q_start=-3 \
    loss.params.q_end=0.5
```

For the **fixed-$q$** ablation in the paper, use `attack=auto_multi_pgd_tsallis_ce` and set `loss.params.q=<value>` (e.g. $-2$, $-1$, $0$, $0.5$, $1$).

## Reproducing the Paper

The paper benchmarks five baselines plus TsallisPGD. The available attack configs are:

| Attack               | Hydra config                          |
| -------------------- | ------------------------------------- |
| CEPGD                | `auto_multi_pgd_ce`                   |
| SegPGD               | `auto_multi_pgd_ce_annealed`          |
| CosPGD               | `auto_multi_pgd_ce_cossim`            |
| JSPGD                | `auto_multi_pgd_js`                   |
| MaskedPGD            | `auto_multi_pgd_masked_ce`            |
| TsallisPGD           | `auto_multi_pgd_tsallis_ce_adaptive`  |
| TsallisPGD (fixed-q) | `auto_multi_pgd_tsallis_ce`           |

Per-benchmark driver scripts wrap the full sweep over models, attacks, and perturbation budgets used in the paper:

```bash
cd experiments
./test_ddcat_cityscapes.sh    # Cityscapes, PSPNet (DDCAT-robust)
./test_pirat_voc2012.sh       # Pascal VOC, UPerNet (PIR-AT)
./test_pirat_ade20k.sh        # ADE20K, UPerNet (PIR-AT) / Segmenter (PIR-AT)
./test_mmseg_voc2012.sh       # Pascal VOC, clean MMSeg models
./test_mmseg_cityscapes.sh    # Cityscapes, clean MMSeg models
```

Each script accepts the list of attacks to dispatch via its arguments; see `experiments/attack_utils.sh` for the helpers used to build commands. Result aggregation lives in `scripts/parse_attack_results.py`.

## Repository Layout

- `experiments/test.py` — main Hydra entry point for running attacks
- `experiments/test_*.sh` — per-benchmark driver scripts used in the paper
- `experiments/data/`, `experiments/models/` — dataset wrappers and model loaders
- `configs/test.yaml`, `configs/test_{cityscapes,voc2012,ade20k}.yaml` — top-level Hydra configs
- `configs/attack/` — PGD/APGD attack variants, including TsallisPGD
- `configs/loss/` — pixel-wise objectives (CE, masked CE, JS, fixed Tsallis, adaptive Tsallis schedules)
- `configs/model/` — model definitions for clean and adversarially trained checkpoints
- `packages/segmentation_attacks/` — core attack and loss implementations
- `scripts/` — data/model download utilities and result parsing
- `docs/` — figures and supporting assets

## Citation

If you use this code or build on TsallisPGD, please cite:

```bibtex
@inproceedings{matyasko2026tsallispgd,
  title     = {{TsallisPGD}: Adaptive Gradient Weighting for Adversarial Attacks on Semantic Segmentation},
  author    = {Matyasko, Alexander and Lou, Xin and Atmosukarto, Indriyati and Zhang, Wei},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026}
}
```

## Acknowledgments

This research is supported by the National Research Foundation, Singapore, under its AI Singapore Programme (AISG Award No: AISG4-GC-2023-006-1B); the Ministry of Education, Singapore, under its Academic Research Tier-1 Grant (Award No: 1091/R-MA124-R205-0002); and A*STAR under its MTC Individual Research Grant (Award No: M23M6c0113) and MTC Programmatic Grant (Award No: M23L9b0052).

The evaluation harness builds on prior work in segmentation adversarial robustness — SegPGD, CosPGD, and the JSPGD/MaskedPGD/PIR-AT line of work — and integrates pretrained checkpoints from those releases. We thank the authors for making their code and models available.
