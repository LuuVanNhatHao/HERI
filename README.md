# HERI

**Hypothesis-Mediated Evidence Reinterpretation for Grounded Multimodal Manipulation Verification**

HERI detects multimodal manipulation, identifies manipulation types, and localizes image regions and text spans. This repository provides the model, data adapters, training and evaluation pipelines, experiment configurations, and the reproducible SAMM-20K annotation manifest.

## Method overview

```text
Image + text
    -> SigLIP2 / DINOv3 / DeBERTa-v3-large
    -> visual-semantic, visual-forensic, cross-modal and contextual-text evidence
    -> saliency/diversity selection and reliability estimation
    -> hypothesis-evidence graph with shared reasoning weights
    -> semantic and hypothesis-relation refresh between two reasoning cycles
    -> manipulation probabilities, Noisy-OR authenticity, image/text grounding
```

The evidence families are VS, VF, XA and TC. Graph budgeting selects reasoning nodes while retaining dense features for grounding. Reliability is estimated once per forward pass; relation neighborhoods and support strengths are refreshed from the evolved states. Prediction and grounding heads share the refined hypothesis representations.

DGM4 uses FS/FA hypotheses for image manipulation and TS/TA for text manipulation. SAMM uses SWAP and ATTRIBUTE, each applicable to both modalities. Annotation-availability masks define supervision and evaluation populations. Alignment calibration averages the best image similarity at each valid text position against a detached global cosine target.

## Installation

Use Python 3.10 or later in a compatible environment:

```bash
git clone https://github.com/LuuVanNhatHao/HERI.git
cd HERI
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -B scripts/heri.py --help
```

The production requirements pin PyTorch for CUDA 12.8. Training requires sufficient GPU memory and access to the pretrained backbones: `google/siglip2-so400m-patch14-384`, `facebook/dinov3-vitl16-pretrain-lvd1689m`, and `microsoft/deberta-v3-large`. The native HGT operator uses PyTorch; GAT controls use the declared `torch-geometric` dependency. The launcher suppresses Python bytecode generation.

## Datasets and SAMM-20K

| Protocol | Training | Validation | Test | Hypotheses |
| --- | ---: | ---: | ---: | --- |
| DGM4 | Original release split | Original release split | Original release split | FS, FA, TS, TA |
| SAMM-20K | 20,000 | 20,000 | 50,000 | SWAP, ATTRIBUTE |

SAMM-20K denotes the 20,000-example **training subset**. Selection seed 777 is independent of the model-training seeds. Validation and test retain the original release order and populations.

The repository includes:

- [SAMM-20K manifest](data/SAMM-20K/manifest.json): all selected training IDs, selection recipe, source identities, class counts and reconstructed annotation hashes.
- [Original SAMM annotation snapshot](data/sources/SAMM): the unchanged `SAMM-without-CAP` training, validation and test JSONs, pinned to publisher revision `9ff6eafbe04f13e9ad8df6bc09a63f9f7a24f3f4`.
- [Annotation license](data/sources/SAMM/LICENSE) and [source attribution](data/sources/SAMM/NOTICE).

Reconstruct the runtime annotations from the included originals:

```bash
python -B scripts/heri.py reconstruct-samm --source-root data/sources/SAMM --output-root data/SAMM
```

The command verifies source hashes, the selected ID list, split sizes and output hashes against the published manifest before writing. It refuses to replace different existing annotations. Obtain image files separately from the [official SAMM release](https://huggingface.co/datasets/SJJ0854/SAMM), and obtain DGM4 separately under its release terms.

```text
data/
  SAMM-20K/manifest.json              # published selection and identities
  sources/SAMM/{train,val,test}.json  # published original annotations
  SAMM/{train,val,test}.json          # reconstructed runtime annotations
  SAMM/images/{orig_output,swap_jpg,emotion_jpg}/...
  DGM4/{train,val,test}.json
  DGM4/images/...
```

Only the manifest, original SAMM JSONs and their attribution are distributed. Reconstructed annotations, images and generated run products remain local. The upstream dataset card declares Apache-2.0 for SAMM; that annotation license does not assign a license to the HERI implementation.

## Training and evaluation

Run from the repository root:

```bash
python -B scripts/heri.py prepare-data config/datasets/dgm4/prepare.yaml
python -B scripts/heri.py train config/datasets/dgm4/heri/train.yaml
python -B scripts/heri.py evaluate config/datasets/dgm4/heri/evaluate.yaml

python -B scripts/heri.py prepare-data config/datasets/samm/prepare.yaml
python -B scripts/heri.py train config/datasets/samm/heri/train.yaml
python -B scripts/heri.py evaluate config/datasets/samm/heri/evaluate.yaml
```

Set dataset locations and checkpoint resolution in the corresponding YAML. B0 global-fusion and B1 cross-attention baselines use entries under `config/datasets/*/baselines/`. Data preparation audits annotations and paths; training and evaluation measure the declared dataset hashes at startup.

The reference configuration uses hidden size 512, VS/VF/XA/TC budgets 96/96/64/64, semantic TopK 8 and two shared reasoning cycles. Training uses 20 epochs, batch size 16, AdamW with weight decay 0.01, module learning rate `1e-4` and encoder learning rate `2e-6`. Encoders freeze for epochs 1-3 and fully unfreeze from epoch 4; 500 warm-up steps precede global linear decay. The reference seed is 7, and matched studies use 7/21/101. Decisions use fixed 0.5; validation CF1 selects checkpoints, with early stopping disabled.

## Controlled experiments

| Configuration under `config/experiments/` | Study type | Comparison |
| --- | --- | --- |
| `dgm4/reasoning_controls.yaml` | `scientific` | Backbone/query controls and reasoning mechanisms |
| `dgm4/progressive.yaml` | `scientific` | Cumulative architectural construction |
| `dgm4/components.yaml` | `scientific` | Component and attention controls |
| `dgm4/matched_components.yaml` | `scientific` | Matched-seed component controls |
| `dgm4/cycle_study.yaml` | `cycle` | Cycle depth and relation refresh |
| `dgm4/ablation.yaml` | `ablation` | Focused component removal |
| `dgm4/multiseed.yaml` | `multiseed` | Repeated reference runs |
| `samm/matched_multiseed.yaml` | `samm` | Matched SAMM models and controls |
| `samm/key_ablations.yaml` | `ablation` | SAMM component removal |
| `samm/evidence_sufficiency_ablation.yaml` | `ablation` | Sufficiency-objective comparison |

Inspect resolved jobs before launching training:

```bash
python -B scripts/heri.py study scientific config/experiments/dgm4/progressive.yaml --dry-run
python -B scripts/heri.py study scientific config/experiments/dgm4/components.yaml --dry-run
python -B scripts/heri.py study samm config/experiments/samm/matched_multiseed.yaml --dry-run
```

Remove `--dry-run` to execute. Study options support variant/seed selection, training or evaluation phases, and source/configuration-checked resume where applicable; consult `study --help` and the [configuration index](config/README.md).

Each component intervention preserves unrelated base settings. Progressive constructions change several mechanisms cumulatively: pooled evidence, routed pooling, one-pass hypothesis graphs, constrained Noisy-OR prediction and full iterative reasoning. Resolved designs and configuration deltas are stored with each study. Removing hypothesis nodes also changes the readout, so that contrast does not isolate node count alone.

## Outputs and reproducibility

Evaluation exports per-example predictions, metrics and a summary. Metrics cover authenticity accuracy/AUC/EER, attribution mAP/CF1/OF1, original-coordinate image IoU, native-token/original-word text F1, and hypothesis-specific EDC/UDR/VSR diagnostics where applicable. Availability masks and metric populations remain explicit in exported definitions.

Generated run products are written beneath `artifacts/`. Checkpoints carry bound configuration snapshots and metadata; completed evaluations seal hashes of their inputs and outputs. Inspect a study with:

```bash
python -B scripts/heri.py audit-study path/to/manifest.json --output path/to/audit.json
```

The audit reports missing, inconsistent or internally consistent artifact chains. Scientific aggregation requires consistent artifacts and retains incomplete jobs in the planned population. This checks saved-run consistency without recomputing benchmark metrics.

## Source layout

```text
src/heri/data/         Dataset adapters, geometry, token alignment and reconstruction
src/heri/evidence/     Four evidence families and dense feature stores
src/heri/routing/      Reliability estimation and selection
src/heri/graph/        Typed relations, HGT and iterative reasoning
src/heri/models/       Encoders, HERI, baselines and prediction heads
src/heri/losses/       Masked learning objectives
src/heri/training/     Optimization, schedules, precision and checkpoints
src/heri/evaluation/   Inference and result export
src/heri/metrics/      Task and evidence-support metrics
src/heri/experiments/  Study designs, orchestration and provenance
config/               Dataset and experiment configurations
scripts/heri.py        Command-line entry point
data/                 Published SAMM annotations and subset manifest
```

## SAMM attribution

SAMM was introduced by Jinjie Shen, Yaxiong Wang, Lechao Chen, Pu Nan and Zhun Zhong in *Beyond Artificial Misalignment: Detecting and Grounding Semantic-Coordinated Multimodal Manipulation* (ACM Multimedia 2025). See the [paper](https://arxiv.org/abs/2509.12653), [official dataset](https://huggingface.co/datasets/SJJ0854/SAMM) and [official implementation](https://github.com/shen8424/SAMM-RamDG-CAP).
