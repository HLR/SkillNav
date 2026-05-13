# SkillNav

Official code release for **Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents**.

[Paper (arXiv)](https://arxiv.org/abs/2508.07642) · [Project page](https://hlr.github.io/SkillNav/) · [Annotations (Google Drive)](https://drive.google.com/drive/folders/1ZYCi2yJAHqk96Ptud6sqQRtWJalIyig7?usp=sharing)

---

## Repository layout

```
SkillNav/
├── skillnav/
│   └── backbones/
│       ├── scalevln/                       # SkillNav built on ScaleVLN (ViT-B/16)
│       │   ├── maps_nav_src/               # working dir for train/test
│       │   │   ├── moe/                    # skill-based agents + VLM router
│       │   │   ├── models/                 # transformer / VLN-BERT backbone
│       │   │   ├── r2r/                    # navigation env, agent loop, parser
│       │   │   ├── prompts/                # router / reordering / data prompts
│       │   │   ├── evaluation/             # offline eval (NavNuances etc.)
│       │   │   ├── scripts/                # train / test bash scripts
│       │   │   └── utils/
│       │   └── datasets/                   # features, annotations, ckpts
│       └── srdf/                           # SkillNav built on VLN-SRDF (InternViT-6B)
│           ├── map_nav_src/                # same layout as scalevln/maps_nav_src
│           └── datasets/
├── assets/                                 # paper figures (PDF + PNG)
│   ├── figures/                            # rendered figures used by the page
│   └── source/                             # editable PDF sources
├── docs/                                   # extra documentation
├── static/                                 # project-page CSS / JS
├── index.html                              # project page
├── pyproject.toml
├── requirements.txt
└── README.md
```

> The inner directory names `maps_nav_src/` (ScaleVLN) and `map_nav_src/` (SRDF) are
> kept verbatim from the upstream baselines so their internal bare imports
> (`from utils.x`, `from moe.y`, …) keep working without rewriting any source file.

Two backbone variants are kept side-by-side because they require different feature
extractors and pretrained checkpoints (ScaleVLN-Aug vs. SRDF-Aug).

---

## 1. Matterport3D Simulator

We use the **latest** version of the [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator) (not v0.1).
Python 3.9 is recommended.

```bash
# system deps
sudo apt-get update
sudo apt-get install -y libjsoncpp-dev libepoxy-dev libglm-dev libopencv-dev \
                        libegl1 libegl1-mesa-dev libgl1-mesa-dev libtiff-dev \
                        libosmesa6 libosmesa6-dev libglew-dev

# conda packages
conda create -n skillnav python=3.9 -y && conda activate skillnav
conda install -c conda-forge cmake gdal libtiff libstdcxx-ng -y

# build the simulator (EGL backend)
cd Matterport3DSimulator
mkdir -p build && cd build
cmake -DEGL_RENDERING=ON -DPYTHON_EXECUTABLE="$(which python)" ..
make -j

# expose to PYTHONPATH
export PYTHONPATH=$(realpath ..):$PYTHONPATH
```

---

## 2. Install SkillNav

```bash
git clone https://github.com/HLR/SkillNav.git
cd SkillNav
pip install -r requirements.txt
pip install -e .              # editable install of the skillnav package
```

The router uses a VLM served via `vLLM`. If you plan to run the router locally, make sure
`vllm`, `transformers>=4.45`, and a compatible CUDA stack are installed (see
`requirements.txt`).

---

## 3. Data

### R2R skill-specific annotations

Download from the [Google Drive folder](https://drive.google.com/drive/folders/1ZYCi2yJAHqk96Ptud6sqQRtWJalIyig7?usp=sharing) and place under each backbone's annotations folder:

```
skillnav/backbones/scalevln/datasets/R2R/annotations/
skillnav/backbones/srdf/datasets/R2R/annotations/
```

### Pretrained features and checkpoints

| Backbone | Features                                                   | Init checkpoint              |
| -------- | ---------------------------------------------------------- | ---------------------------- |
| ScaleVLN | ViT-B/16 (same as [ScaleVLN](https://github.com/wz0919/ScaleVLN)) | ScaleVLN-pretrained ViT-B/16 |
| SRDF     | InternViT-6B (same as [VLN-SRDF](https://github.com/wz0919/VLN-SRDF)) | SRDF-pretrained checkpoint   |

Drop them under each backbone's `datasets/R2R/features/` and
`datasets/R2R/trained_models/` directories. The bash scripts under each backbone's
`scripts/` directory reference these paths directly.

---

## 4. Train

Each skill specialist is trained on its own skill-specific augmentation split.

```bash
# ScaleVLN backbone
cd skillnav/backbones/scalevln/maps_nav_src
bash scripts/train_r2r_b16_mix_vertical.sh        # Vertical Movement (VM)
bash scripts/train_r2r_b16_mix_direction.sh       # Directional Adjustment (DA)
bash scripts/train_r2r_b16_mix_landmark.sh        # Landmark Detection (LD)
bash scripts/train_r2r_b16_mix_region.sh          # Area & Region Identification (AR)
bash scripts/train_r2r_b16_mix_stop.sh            # Stop & Pause (SP)
bash scripts/train_r2r_b16_mix_temporal.sh        # Temporal Reordering data
```

```bash
# SRDF (InternViT-6B) backbone
cd skillnav/backbones/srdf/map_nav_src
bash scripts/train_r2r_internvit6b_mix_vertical.sh
# …same five skills…
```

---

## 5. Test

End-to-end evaluation uses the VLM-based action router (top-1 routing).

### Start the router server (vLLM)

```bash
cd skillnav/backbones/scalevln/maps_nav_src/moe
python vLLM_API.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8000
```

Supported routers: `Qwen2.5-VL-7B-Instruct`, `GLM-4.1V-9B-Thinking`, GPT-4o (via API).

### Run navigation

```bash
# R2R Val-Unseen / Test-Unseen
cd skillnav/backbones/scalevln/maps_nav_src
bash scripts/test_r2r_b16_moe-top1.sh

# GSA-R2R
bash scripts/test_gsa-r2r_b16_moe-top1.sh

# NavNuances per-skill eval
bash scripts/test_navnuance_b16_mix.sh
```

For the SRDF backbone, use the analogous scripts under
`skillnav/backbones/srdf/map_nav_src/scripts/`.

---

## 6. Baselines

SkillNav builds on two open-source VLN baselines &mdash; the upstream repos are:

* [ScaleVLN](https://github.com/wz0919/ScaleVLN) (Wang et al., 2023)
* [VLN-SRDF](https://github.com/wz0919/VLN-SRDF) (Wang et al., 2024)

The novel SkillNav code &mdash; the skill specialists, the temporal reordering module,
the VLM action router, the skill-specific synthetic data prompts &mdash; lives under each
backbone's `moe/` and `prompts/` directories.

---

## 7. Citation

```bibtex
@misc{ma2025breakingbuildingupmixture,
  title  = {Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents},
  author = {Tianyi Ma and Yue Zhang and Zehao Wang and Parisa Kordjamshidi},
  year   = {2025},
  eprint = {2508.07642},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url    = {https://arxiv.org/abs/2508.07642}
}
```
