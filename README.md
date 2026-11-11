# HERI

**Hypothesis-Mediated Evidence Reinterpretation for Grounded Multimodal Manipulation Verification**

HERI addresses three related tasks: image-text authenticity detection, manipulation-type attribution, and localization of manipulated image regions and text tokens. Candidate manipulation types are represented as hypothesis nodes. Local evidence and hypothesis states interact through two graph-reasoning cycles with shared parameters, allowing the model to update the relations that determine evidence support.

**DGM4 is the primary benchmark** for model comparisons, reasoning studies and component analyses. **SAMM-20K is a supplementary benchmark** with separately configured training and evaluation splits.

## Method

The model extracts four evidence families: visual-semantic (VS), visual-forensic (VF), cross-modal alignment (XA), and contextual text (TC). Learned saliency and feature diversity determine which evidence nodes enter the graph. Dense features remain available to the grounding heads.

| Component | Role |
| --- | --- |
| SigLIP2, DINOv3 and DeBERTa-v3-large | Encode global and local image-text representations |
| Evidence selection and reliability | Select graph nodes and estimate family-specific evidence reliability |
| Hypothesis-evidence graph | Represent structural, semantic, support and feedback relations |
| Shared HGT reasoning | Update evidence and hypothesis states; refresh semantic and hypothesis relations between cycles |
| Prediction heads | Produce manipulation probabilities and hypothesis-conditioned image/text grounding |

Reliability is computed once per forward pass and remains fixed across reasoning cycles. Relation neighborhoods and support strengths depend on the evolving states. Refined hypothesis representations are shared by attribution and grounding.

For manipulation probabilities $p_k$, the authenticity score is

$$
p_{\mathrm{fake}} = 1 - \prod_{k=1}^{K}(1-p_k).
$$

Image predictions aggregate individually active image hypotheses at threshold 0.5. Text predictions combine hypothesis-conditioned token scores using relative probability weights. Annotation-availability masks determine which localization targets contribute to supervision and evaluation.

Training combines attribution, authenticity and localization losses with per-modality evidence sufficiency, alignment calibration and router regularization. Alignment calibration applies Smooth L1 to the mean text-to-image maximum similarities and a detached global image-text cosine target, excluding padded positions.

## Environment

The reference dependencies are Python 3.10, PyTorch 2.8.0 with CUDA 12.8, and Transformers 4.57.6. Exact constraints are listed in [requirements.txt](requirements.txt) and [requirements/runtime.txt](requirements/runtime.txt).

```bash
git clone https://github.com/LuuVanNhatHao/HERI.git
cd HERI
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -B scripts/heri.py --help
```

The configured encoders are `google/siglip2-so400m-patch14-384`, `facebook/dinov3-vitl16-pretrain-lvd1689m`, and `microsoft/deberta-v3-large`. Their pretrained weights must be accessible in the training environment. HGT uses native PyTorch operations; GAT controls use `torch-geometric`. The launcher disables Python bytecode generation.

## Data protocols

| Benchmark | Role | Train | Validation | Test | Manipulation hypotheses |
| --- | --- | ---: | ---: | ---: | --- |
| DGM4 | Primary | 208,184 | 22,126 | 50,705 | Face swap, face attribute, text swap, text attribute |
| SAMM-20K | Supplementary | 20,000 | 20,000 | 50,000 | SWAP, ATTRIBUTE |

### Primary benchmark: DGM4

Obtain the annotations and images from the [official DGM4 release](https://huggingface.co/datasets/rshaojimmy/DGM4). The main model configuration is [DGM4 training](config/datasets/dgm4/heri/train.yaml); the corresponding study manifests are under `config/experiments/dgm4/`.

Image grounding applies to face-swap/face-attribute hypotheses and text grounding to text-swap/text-attribute hypotheses. Dataset paths, ontology, preprocessing and annotation hashes are specified in `config/datasets/dgm4/`.

### Supplementary benchmark: SAMM-20K

SAMM evaluates the SWAP and ATTRIBUTE hypotheses, both applicable to image and text grounding. Its configurations are under `config/datasets/samm/`, with supplementary study manifests under `config/experiments/samm/`.

SAMM-20K denotes the 20,000-example **training subset** drawn from the original 190,970-row training split. Selection uses seed 777; selected integer IDs are sorted. Validation and test preserve the original release populations and row order. The selection seed is independent of model-training seeds.

The [manifest](data/SAMM-20K/manifest.json) contains all selected training IDs, the selection and serialization rules, source-file identities, split sizes, class counts and expected annotation hashes. The [included original annotations](data/sources/SAMM) are unchanged `SAMM-without-CAP` files from publisher revision `9ff6eafbe04f13e9ad8df6bc09a63f9f7a24f3f4`.

```bash
python -B scripts/heri.py reconstruct-samm --source-root data/sources/SAMM --output-root data/SAMM
```

Reconstruction uses the manifest IDs and verifies source hashes, split sizes, ID order and output hashes before writing runtime annotations. Existing annotations with different contents are preserved. Images for this supplementary benchmark must be obtained separately from the [official SAMM release](https://huggingface.co/datasets/SJJ0854/SAMM).

```text
data/
  DGM4/{train,val,test}.json
  DGM4/images/...
  SAMM-20K/manifest.json
  sources/SAMM/{train,val,test}.json
  SAMM/{train,val,test}.json
  SAMM/images/{orig_output,swap_jpg,emotion_jpg}/...
```

The original SAMM annotation files are distributed under their [Apache-2.0 license](data/sources/SAMM/LICENSE), with [upstream attribution](data/sources/SAMM/NOTICE). Reconstructed annotations, images, model weights and run outputs are stored locally.

## Training and evaluation

Commands are executed from the repository root. Configure data locations and checkpoint resolution in the relevant YAML before running.

For the primary benchmark:

```bash
python -B scripts/heri.py prepare-data config/datasets/dgm4/prepare.yaml
python -B scripts/heri.py train config/datasets/dgm4/heri/train.yaml
python -B scripts/heri.py evaluate config/datasets/dgm4/heri/evaluate.yaml
```

For the supplementary benchmark, after reconstructing annotations and preparing images:

```bash
python -B scripts/heri.py prepare-data config/datasets/samm/prepare.yaml
python -B scripts/heri.py train config/datasets/samm/heri/train.yaml
python -B scripts/heri.py evaluate config/datasets/samm/heri/evaluate.yaml
```

B0 global fusion and B1 cross-attention use the corresponding entries under `config/datasets/*/baselines/`. All runs verify configured annotation hashes at startup.

| Setting | Reference configuration |
| --- | --- |
| Hidden dimension | 512 |
| VS / VF / XA / TC graph budgets | 96 / 96 / 64 / 64 |
| Semantic neighbors | 8 |
| Reasoning | Two cycles, shared parameters |
| Training | 20 epochs, batch size 16 |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rates | New modules: `1e-4`; encoders: `2e-6` |
| Encoder adaptation | Frozen in epochs 1-3; fully unfrozen from epoch 4 |
| Learning-rate schedule | 500 warm-up steps, then linear decay across both stages |
| Seeds | Reference: 7; matched studies: 7, 21, 101 |
| Decisions | Fixed threshold 0.5 |
| Checkpoint selection | DGM4: `cf1_fixed`; SAMM: `samm_official_cf1` |

Early stopping is disabled in the reference protocol. Checkpoint selection uses validation data; the selected checkpoint is evaluated on the test split.

## Experimental design

The main study is defined under `config/experiments/dgm4/`; supplementary studies are defined under `config/experiments/samm/`. Manifests specify the models, seeds and comparisons to execute. Their resolved configurations record the operators, readouts, objectives and changes between experimental conditions.

| Configuration under `config/experiments/` | CLI study type | Comparison |
| --- | --- | --- |
| `dgm4/reasoning_controls.yaml` | `scientific` | Backbone/query controls and relation-refresh mechanisms |
| `dgm4/progressive.yaml` | `scientific` | Cumulative architectural construction |
| `dgm4/components.yaml` | `scientific` | Component and attention controls |
| `dgm4/matched_components.yaml` | `scientific` | Three-seed component comparisons |
| `dgm4/cycle_study.yaml` | `cycle` | Reasoning depth and relation refresh |
| `dgm4/ablation.yaml` | `ablation` | Focused component removal |
| `dgm4/multiseed.yaml` | `multiseed` | Repeated reference runs |
| `samm/matched_multiseed.yaml` | `samm` | Matched SAMM model and component comparisons |
| `samm/key_ablations.yaml` | `ablation` | SAMM component removal |
| `samm/evidence_sufficiency_ablation.yaml` | `ablation` | Evidence-sufficiency objectives |

Materialize and inspect jobs without loading pretrained models:

```bash
python -B scripts/heri.py study scientific config/experiments/dgm4/progressive.yaml --dry-run
python -B scripts/heri.py study scientific config/experiments/dgm4/components.yaml --dry-run
python -B scripts/heri.py study samm config/experiments/samm/matched_multiseed.yaml --dry-run
```

Remove `--dry-run` to execute. Available variant, seed, phase and resume options depend on the study type; see `study --help` and the [configuration index](config/README.md).

Component interventions preserve unrelated base settings. Progressive construction combines several architectural changes: evidence pooling, reliability routing, hypothesis-graph reasoning, constrained Noisy-OR prediction and iterative refinement. Removing hypothesis nodes also changes the prediction readout; its effect therefore includes both changes.

## Evaluation conventions

| Task | Reported measures |
| --- | --- |
| Authenticity | Accuracy, ROC-AUC and EER |
| Manipulation attribution | mAP, class-level F1 (CF1) and overall F1 (OF1) |
| Image grounding | IoU in original-image coordinates, with explicit annotation populations |
| Text grounding | Native-subword F1 and original-word F1, reported separately |
| Evidence support | Evidence-decision consistency (EDC), unsupported-decision rate (UDR) and verified support at rank 1 (VSR@1), where applicable |

CF1 is the harmonic mean of class-averaged precision and recall; it differs from the arithmetic mean of per-class F1 scores. OF1 is the micro-averaged F1 across class decisions. For SAMM, released-protocol attribution uses `(p_swap, p_attribute, p_fake)` with targets `000` for original, `101` for swap and `011` for attribute examples. This projection defines `samm_official_cf1` used for checkpoint selection.

Grounding exports retain annotation availability, word coverage and truncation information. Missing predictions for original words are treated as negative. Hypothesis-specific evidence-support diagnostics are reported only for models that produce the required support representations.

## Reproducibility and run records

Generated files are written beneath `artifacts/`. A completed run records its effective configurations, source identity, observed dataset hashes, checkpoint metadata, predictions and evaluation summary. Checkpoint metadata binds the weight file to an immutable configuration snapshot. Evaluation records contain hashes linking the files used and produced by that evaluation.

```bash
python -B scripts/heri.py audit-study path/to/manifest.json --output path/to/audit.json
```

The audit reports `missing`, `inconsistent` or `internally_consistent` artifact chains. Scientific aggregation requires consistent records and retains incomplete jobs in the planned population. These checks verify agreement between saved files and recorded run identifiers. Benchmark metrics are computed by the evaluation pipeline.

## Code organization

| Location | Responsibility |
| --- | --- |
| [Model composition](src/heri/models/heri/model.py) | Connect encoders, evidence, routing, reasoning and prediction |
| [Data adapters](src/heri/data) | Parse annotations, align tokens and transform geometry |
| [Evidence](src/heri/evidence), [routing](src/heri/routing) | Construct dense evidence, select nodes and estimate reliability |
| [Graph reasoning](src/heri/graph) | Build typed relations and perform shared iterative updates |
| [Prediction heads](src/heri/models/heads), [losses](src/heri/losses) | Compute predictions and masked training objectives |
| [Training](src/heri/training) | Manage optimization, precision, schedules and resume |
| [Evaluation](src/heri/evaluation), [metrics](src/heri/metrics) | Perform inference, compute measures and export outputs |
| [Experiments](src/heri/experiments) | Define comparisons, orchestrate runs and inspect provenance |
| [CLI](scripts/heri.py), [configurations](config) | Provide executable entry points and experiment settings |

## Dataset attribution

DGM4 data, methodology and citation information are provided by the [official project](https://github.com/rshaojimmy/MultiModal-DeepFake) and [dataset release](https://huggingface.co/datasets/rshaojimmy/DGM4).

SAMM was introduced by Jinjie Shen, Yaxiong Wang, Lechao Chen, Pu Nan and Zhun Zhong in *Beyond Artificial Misalignment: Detecting and Grounding Semantic-Coordinated Multimodal Manipulation* (ACM Multimedia 2025). See the [paper](https://arxiv.org/abs/2509.12653), [dataset](https://huggingface.co/datasets/SJJ0854/SAMM) and [official implementation](https://github.com/shen8424/SAMM-RamDG-CAP).
