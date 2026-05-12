# TsallisPGD: Adaptive Gradient Weighting for Adversarial Attacks on Semantic Segmentation

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![Pixi](https://img.shields.io/badge/env-pixi-2f80ed)
![IJCNN 2026](https://img.shields.io/badge/paper-IJCNN%202026-purple)
![License MIT](https://img.shields.io/badge/license-MIT-green)

**Alexander Matyasko, Xin Lou, Indriyati Atmosukarto, and Wei Zhang**<br>
Singapore Institute of Technology<br>
IJCNN 2026 (camera-ready)<br>
Paper: arXiv coming soon

### Abstract

Attacking semantic segmentation models is significantly harder than attacking image classification models because the attacker must change thousands of pixel predictions simultaneously. Standard pixel-wise cross-entropy (CE) is not well matched to this setting: it tends to overemphasize already-misclassified pixels, slowing optimization and overstating model robustness. We introduce **TsallisPGD**, an adversarial attack based on the Tsallis cross-entropy, a generalization of CE parameterized by `q`. The objective adaptively reshapes the gradient landscape by controlling how gradient mass is distributed across pixels. By varying `q`, the attack can target pixels at different confidence levels. We show that no single fixed `q` is universally optimal across datasets, model architectures, and perturbation budgets. Motivated by this, we use a dynamic linear `q` schedule that sweeps from `q = -2` to `q = 1` during optimization. Across Cityscapes, Pascal VOC, and ADE20K, TsallisPGD achieves the best average attack rank among CEPGD, SegPGD, CosPGD, JSPGD, and MaskedPGD in reducing pixel accuracy and mIoU on both standard and adversarially trained segmentation models.

### TsallisPGD Attack

TsallisPGD replaces pixel-wise CE with Tsallis cross-entropy. In gradient space, each pixel is effectively weighted by `p_y ** (1 - q)`, where `p_y` is the predicted probability of the ground-truth class.

- `q = 1` recovers standard cross-entropy.
- `q < 1` emphasizes high-confidence, correctly classified pixels and down-weights pixels that are already broken.
- The main paper configuration uses a linear schedule from `q = -2` to `q = 1`.

The adaptive schedule is the default because fixed `q` values behave differently across datasets, architectures, and perturbation budgets. Sweeping from `q = -2` to `q = 1` starts by strongly suppressing gradients from low-confidence or already-broken pixels, then gradually returns toward CE-like behavior as the attack progresses. This schedule was selected on validation experiments and then kept fixed for the paper evaluations.

The main attack config is:

```text
configs/attack/auto_multi_pgd_tsallis_ce_adaptive.yaml
```

It combines APGD step-size selection, a 300-iteration budget, one random start, the multi-epsilon trick, and the validation-selected adaptive Tsallis schedule.

![Per-pixel input-gradient visualization comparing CE and Tsallis CE losses](docs/assets/ce_vs_tsallis_input_gradients.png)

**Figure.** Per-pixel input-gradient visualization for a semantic segmentation model under standard cross-entropy (CE) and Tsallis cross-entropy objectives. The top row shows the input image, the ground-truth class confidence map $p_y$, a binary map of pixels already misclassified by the clean model, and the CE input-gradient norm $\lVert \nabla_x L_i \rVert_2$. The bottom row shows Tsallis CE input-gradient norms $\lVert \nabla_x L_{q,i} \rVert_2$ for $q \in \{-2, -1, 0, 0.5\}$. Gradient maps are independently normalized per panel; therefore color indicates relative spatial importance within each loss, rather than absolute gradient magnitude across losses. As $q$ decreases, Tsallis CE suppresses gradients on low-confidence or already-misclassified pixels and concentrates gradient mass around pixels where the model assigns higher probability to the ground-truth class.

---

### Evaluated Semantic Segmentation Models

This repository contains the evaluation harness used for the TsallisPGD paper. It covers standard MMSegmentation models, DDCAT robust models, and PIR-AT robust models. The PIR-AT checkpoints and evaluation conventions follow the public [Robust-Segmentation](https://github.com/nmndeep/Robust-Segmentation) repository.

| Model family                          | Dataset                | Training / source                                  | Config examples                                                   |
| ------------------------------------- | ---------------------- | -------------------------------------------------- | ----------------------------------------------------------------- |
| PSPNet / DeepLabV3+ / FCN / SegFormer | Cityscapes, Pascal VOC | Standard MMSegmentation checkpoints                | `configs/model/*cityscapes.yaml`, `configs/model/*voc2012.yaml`   |
| PSPNet ResNet-50                      | Cityscapes, Pascal VOC | DDCAT robust checkpoints                           | `pspnet_cityscapes_ddcat`, `pspnet_voc2012_ddcat`                 |
| UPerNet ConvNeXt-T/S CvSt             | Pascal VOC, ADE20K     | PIR-AT robust checkpoints from Robust-Segmentation | `convnext_t_cvst_robust_voc2012`, `convnext_t_cvst_robust_ade20k` |
| Segmenter ViT-S                       | ADE20K                 | PIR-AT robust checkpoint from Robust-Segmentation  | `segment_vit_s_robust_ade20k`                                     |

Clean baseline metrics and attacked metrics are reported in the paper tables and can be regenerated with the scripts under `experiments/`. Checkpoint download locations are encoded in [`scripts/download_models.sh`](scripts/download_models.sh). Dataset locations and model checkpoint roots are configured through Hydra in `configs/paths/default.yaml`.

### Available attacks

| Attack                        | Hydra config                         | Description                                            |
| ----------------------------- | ------------------------------------ | ------------------------------------------------------ |
| CEPGD                         | `auto_multi_pgd_ce`                  | Standard pixel-wise cross-entropy objective            |
| SegPGD                        | `auto_multi_pgd_ce_annealed`         | CE with annealed emphasis on currently correct pixels  |
| CosPGD                        | `auto_multi_pgd_ce_cossim`           | CE gradients combined with cosine-similarity weighting |
| JSPGD                         | `auto_multi_pgd_js`                  | Jensen-Shannon based segmentation attack objective     |
| MaskedPGD                     | `auto_multi_pgd_masked_ce`           | CE restricted to selected pixel subsets                |
| TsallisPGD                    | `auto_multi_pgd_tsallis_ce_adaptive` | Proposed adaptive Tsallis CE schedule                  |
| TsallisPGD fixed-`q` ablation | `auto_multi_pgd_tsallis_ce`          | Fixed-parameter Tsallis CE ablation                    |

---

### Setup

The project is managed with [Pixi](https://pixi.sh). Install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then clone and bootstrap the repository:

```bash
git clone https://github.com/aam-at/tsallis_pgd.git
cd tsallis_pgd
pixi install
pixi run setup
```

`pixi install` creates the Pixi environment. `pixi run setup` installs the Python dependencies, builds a CUDA-enabled `mmcv`, and downloads pretrained segmentation checkpoints.

Before running experiments interactively, start a Pixi shell from the repository root:

```bash
pixi shell
```

This activates the correct Python, CUDA, compiler, and library paths for the current terminal session. Run the experiment commands below from inside this shell.

Requirements:

- Linux x86-64
- NVIDIA GPU
- CUDA 13.0 toolchain, provisioned by Pixi
- Python 3.13, provisioned by Pixi

#### Troubleshooting

- `mmcv` build failures usually indicate a CUDA/compiler mismatch. Run `pixi run setup` from a clean shell so Pixi's CUDA toolchain is first on `PATH`.
- If model downloads fail partway through, re-run `pixi run download-models`; existing checkpoints are skipped.
- Cityscapes download requires valid Cityscapes credentials and acceptance of the dataset terms.

#### Data

Dataset downloads default to `$HOME/data`. Override with `DATA_ROOT` or pass an explicit root to the download script.

```bash
pixi run download-data-pascal-voc-aug
pixi run download-data-ade20k
pixi run download-data-cityscapes
```

Cityscapes download requires Cityscapes account credentials.

---

### Quick Start Demos

The main entry point is the Hydra application `experiments/test.py`.

Run these commands from inside `pixi shell`.

#### Pascal VOC, UPerNet ConvNeXt-T PIR-AT, `epsilon = 8/255`

```bash
python experiments/test.py \
  -cn test_voc2012 \
  model=convnext_t_cvst_robust_voc2012 \
  attack=auto_multi_pgd_tsallis_ce_adaptive \
  attack.base_epsilon=0.031372549 \
  attack.base_iterations=300 \
  task_name=tsallispgd_voc_demo
```

#### ADE20K, UPerNet ConvNeXt-T PIR-AT, `epsilon = 8/255`

```bash
python experiments/test.py \
  -cn test_ade20k \
  model=convnext_t_cvst_robust_ade20k \
  attack=auto_multi_pgd_tsallis_ce_adaptive \
  attack.base_epsilon=0.031372549 \
  attack.base_iterations=300 \
  task_name=tsallispgd_ade20k_demo
```

#### Cityscapes, PSPNet ResNet-50 DDCAT, `epsilon = 0.5/255`

```bash
python experiments/test.py \
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

Outputs are written under `runs/{task_name}/...`.

To override the adaptive schedule:

```bash
python experiments/test.py \
  attack=auto_multi_pgd_tsallis_ce_adaptive \
  loss.params.q_start=-3 \
  loss.params.q_end=0.5
```

For fixed-`q` ablations, use `attack=auto_multi_pgd_tsallis_ce` and set `loss.params.q`.

---

### Reproducing the Paper

Per-benchmark shell scripts wrap the sweeps over models, attacks, and perturbation budgets used in the paper.

Run these commands from inside `pixi shell`.

```bash
cd experiments
./test_ddcat_cityscapes.sh
./test_pirat_voc2012.sh
./test_pirat_ade20k.sh
./test_mmseg_voc2012.sh
./test_mmseg_cityscapes.sh
```

Each script accepts the list of attacks to dispatch via its arguments. Shared command construction lives in `experiments/attack_utils.sh`. Result aggregation lives in `scripts/parse_attack_results.py`.

---

### Project Structure

```text
.
├── configs/                       # Hydra configs for data, models, losses, and attacks
│   ├── attack/                    # PGD/APGD variants, including TsallisPGD
│   ├── loss/                      # CE, masked CE, JS, fixed Tsallis, adaptive Tsallis
│   └── model/                     # Clean and adversarially trained segmentation models
├── experiments/
│   ├── test.py                    # Main Hydra entry point
│   ├── test_*.sh                  # Paper reproduction sweeps
│   ├── data/                      # Dataset wrappers and pipelines
│   └── models/                    # Model loaders and wrappers
├── packages/segmentation_attacks/
│   └── src/segmentation_attacks/  # Core attack implementations
├── scripts/                       # Download utilities, visualizations, result parsing
└── docs/assets/                   # README and paper-supporting figures
```

---

#### Citation

If you use this code or build on TsallisPGD, please cite:

```bibtex
@inproceedings{matyasko2026tsallispgd,
  title     = {{TsallisPGD}: Adaptive Gradient Weighting for Adversarial Attacks on Semantic Segmentation},
  author    = {Matyasko, Alexander and Lou, Xin and Atmosukarto, Indriyati and Zhang, Wei},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026}
}
```

##### Acknowledgements

This research is supported by the National Research Foundation, Singapore, under its AI Singapore Programme (AISG Award No: AISG4-GC-2023-006-1B); the Ministry of Education, Singapore, under its Academic Research Tier-1 Grant (Award No: 1091/R-MA124-R205-0002); and A\*STAR under its MTC Individual Research Grant (Award No: M23M6c0113) and MTC Programmatic Grant (Award No: M23L9b0052).

The evaluation harness builds on prior work in semantic segmentation robustness, including SegPGD, CosPGD, DDCAT, and the JSPGD/MaskedPGD/PIR-AT line of work. It also integrates pretrained checkpoints and conventions from [Robust-Segmentation](https://github.com/nmndeep/Robust-Segmentation), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Robust-Semantic-Segmentation](https://github.com/JIA-Lab-research/Robust-Semantic-Segmentation), and [Revisiting-AT](https://github.com/nmndeep/revisiting-at). We thank the authors for making their code and models available.
