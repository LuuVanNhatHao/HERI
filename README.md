# HERA

Hypothesis-Evidence Routing for Attribution-First Multimodal Manipulation
Detection.

HERA is a research implementation for detecting manipulated image-text pairs,
identifying the manipulation type, and grounding the supporting evidence. The
same core architecture is instantiated on two protocols:

- **DGM4**, with four atomic hypotheses: face swap (`FS`), face attribute
  manipulation (`FA`), text swap (`TS`), and text attribute manipulation
  (`TA`).
- **SAMM-20K**, with two dataset-native hypotheses: image-text swap (`SWAP`)
  and attribute manipulation (`ATTRIBUTE`). Both hypotheses may use visual and
  textual evidence.

This README documents the method, implementation contract, and reproducible
end-to-end commands. Experimental numbers are intentionally kept out of the
repository overview; generated CSV and JSON artifacts are the source of record
for a particular run.

## Core idea

Conventional multimodal detectors often fuse an image and its text into one
global representation and predict `real` or `fake` directly. HERA instead
treats manipulation detection as structured hypothesis testing:

```text
image + text
     |
     v
pretrained multimodal, visual, and language encoders
     |
     v
four typed evidence families
     |
     v
one-time reliability routing
     |
     v
typed hypothesis-evidence graph
     |
     v
state-dependent multi-cycle reasoning
     |
     +--> atomic manipulation attribution
     +--> hypothesis-conditioned image grounding
     +--> hypothesis-conditioned text grounding
     `--> Real/Fake probability derived from attribution
```

The binary decision is therefore downstream of the manipulation hypotheses,
not an independent shortcut. This makes attribution part of the decision
mechanism rather than an auxiliary explanation attached after prediction.

## Architecture and mechanisms

### 1. Complementary pretrained encoders

The default implementation combines three pretrained backbones:

| Backbone | Role in HERA |
| --- | --- |
| SigLIP2 (`google/siglip2-so400m-patch14-384`) | Global image-text semantics and aligned local evidence |
| DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) | Visual/forensic evidence and the dense image patch store used for grounding |
| DeBERTa-v3-large (`microsoft/deberta-v3-large`) | Contextual text evidence and the dense token store used for grounding |

Backbone outputs are projected to a shared hidden dimension while preserving
their source identity.

### 2. Typed evidence decomposition

HERA constructs four evidence families before graph reasoning:

- **Visual semantic (`VS`)**: image regions relevant to semantic content.
- **Visual forensic (`VF`)**: image regions useful for manipulation artifacts.
- **Text alignment (`XA`)**: textual evidence used for image-text consistency.
- **Text context (`TC`)**: contextual token evidence used for linguistic and
  attribute-level reasoning.

Each family has its own projection, node budget, validity mask, and provenance
metadata. Dense DINOv3 patches and DeBERTa tokens are retained separately so
grounding is not limited to the reduced graph-node budget.

### 3. Reliability-aware routing

A learned router estimates a soft reliability value for every valid evidence
node and scales that node before graph construction. The router runs once per
forward pass. Its outputs are reusable routing diagnostics, not calibrated
epistemic probabilities.

The canonical system uses soft routing. The controlled `no_router` ablation
replaces learned reliability with uniform evidence flow while leaving the rest
of the pipeline unchanged.

### 4. Typed hypothesis-evidence graph

Evidence nodes and learned hypothesis nodes form a heterogeneous graph. HERA
uses relation-aware Heterogeneous Graph Transformer layers to preserve the
semantics of evidence types and relations. Graph construction includes spatial,
cross-modal similarity, word-alignment, hypothesis-support, and feedback
relations where applicable.

The canonical reasoner performs two shared-weight reasoning cycles:

1. Build the graph from routed evidence and learned hypothesis states, then run
   typed message passing.
2. Reconstruct the refined evidence state, recompute state-dependent semantic
   relations, rebuild the graph, and run the same reasoner again.

Reliability is not re-estimated between cycles. This isolates iterative
hypothesis-evidence refinement from repeated router computation. The hypothesis
states are learned supervised latent states; they are not natural-language
chains of thought.

### 5. Dataset-native hypothesis banks

The model topology is shared, but the hypothesis ontology is explicit in each
dataset configuration:

| Protocol | Hypothesis order | Image evidence | Text evidence |
| --- | --- | --- | --- |
| DGM4 | `FS, FA, TS, TA` | `FS, FA` | `TS, TA` |
| SAMM-20K | `SWAP, ATTRIBUTE` | both hypotheses | both hypotheses |

This is an ontology adapter, not a dataset-specific replacement architecture.
In particular, SAMM does not inherit the DGM4 assumption that each hypothesis
belongs to exactly one modality.

### 6. Attribution-first authenticity

Each final hypothesis state produces one atomic attribution logit. The fake
probability is derived with a numerically stable Noisy-OR:

```text
P(fake) = 1 - product_h (1 - P(h))
P(real) = product_h (1 - P(h))
```

Canonical HERA contains no independent Real/Fake classifier. During training,
the target-consistent binary objective prevents the aggregate fake loss from
pushing known-absent atomic hypotheses upward on manipulated examples.

### 7. Hypothesis-conditioned grounding

- **Image grounding** scores the full DINOv3 patch store with hypothesis
  queries and predicts normalized bounding boxes using patch attention,
  geometric priors, and the refined hypothesis state.
- **Text grounding** scores the full valid DeBERTa token store with
  hypothesis-specific queries and maps subword predictions back to original
  words for evaluation.

Text heads are aggregated without target labels. The canonical
`presence_gated` rule first forms a relative mixture of the relevant
hypothesis-conditioned token probabilities, then scales it by the predicted
probability that at least one text hypothesis is present:

```text
relative token mixture  = sum_h softmax(log P(h) / temperature) * P(token | h)
text presence           = 1 - product_h (1 - P(h))
final token probability = text presence * relative token mixture
```

This retains hypothesis selection while suppressing token false positives on
samples for which no text manipulation is predicted. `relative_soft` and
`token_noisy_or` remain available as explicit diagnostic interventions.

Image grounding uses the analogous ontology-aware, target-free abstention
rule. It emits the official empty box when no configured image hypothesis is
present according to a validation-only threshold:

```text
image presence = 1 - product_{h in image hypotheses} (1 - P(h))
predicted box  = empty if image presence < validation threshold else routed box
```

DGM4 therefore aggregates `FS/FA`, while SAMM aggregates
`SWAP/ATTRIBUTE`; neither the dataset name nor the number of classes is
hard-coded. The threshold is tuned on validation image-presence labels and
then frozen for test. An archived SAMM checkpoint may reuse its saved
validation binary threshold because every SAMM manipulation hypothesis is
image-grounded.

### 8. Multi-task learning and staged adaptation

The training objective combines atomic attribution, attribution-derived binary
calibration, image grounding, text grounding, evidence sufficiency,
cross-modal alignment, and router regularization. Grounding availability masks
exclude only unavailable annotations from the corresponding loss and metric;
the samples remain in detection and attribution evaluation.

Canonical training uses two stages:

- **Stage A, epochs 1-3**: pretrained backbones are frozen and new HERA modules
  are optimized.
- **Stage B, from epoch 4**: all backbones are unfrozen with lower per-backbone
  learning rates; optimizer state for already trained parameters is preserved.

Runs support automatic recovery from `last.pt`. Validation thresholds are
tuned on the validation split and stored with the selected checkpoint. DGM4
selects checkpoints by validation `CF1`; SAMM selects by validation three-class
macro-F1. The default maximum is 20 epochs with patience-three early stopping
after the minimum backbone-adaptation period.

## Evaluation contract

Evaluation writes a summary CSV, prediction CSV, metrics JSON, thresholds, and
text classification reports into an isolated run directory. The main metric
families are:

| Task | Outputs |
| --- | --- |
| Binary detection | accuracy, ROC-AUC, EER, Real/Fake class report |
| Atomic attribution | mAP, class-wise F1, class-averaged F1 (`CF1`), overall F1 (`OF1`) |
| Dataset-native SAMM classification | accuracy and macro/weighted F1 over `orig`, `swap_manipulation`, and `attribute_manipulation` |
| Image grounding | mean IoU and IoU-threshold metrics, including positive-only views |
| Text grounding | token/word precision, recall, F1, coverage, and positive-only views |
| Routing analysis | reliability and routed-evidence statistics by family, class, and dataset |

Thresholds must be tuned on validation data and reused unchanged for test
evaluation. Prediction artifacts record sample IDs, availability masks,
protocol metadata, and routing summaries so analyses can be audited without
re-running inference.

## Repository layout

```text
HERA_project/
|-- config/
|   |-- datasets/             # Canonical DGM4 and SAMM entry points
|   |-- experiments/          # Ablation, cycle, and multi-seed manifests
|   |-- diagnostics/          # Component inspection and server checks
|   `-- benchmarks/           # Auxiliary binary-benchmark adapters
|-- data/                     # Local datasets; large assets are not committed
|-- docs/                     # Scientific contract and generated reports
|-- scripts/                  # Thin CLI entry points
|-- src/
|   |-- data/                 # Schemas, adapters, collation, preflight
|   |-- evidence/             # Typed evidence extraction
|   |-- routing/              # Reliability estimation and routing
|   |-- graph/                # Graph building and iterative HGT reasoning
|   |-- models/               # HERA, baselines, encoders, and heads
|   |-- losses/               # Multi-task objectives
|   |-- training/             # Staged training, resume, checkpoints
|   |-- evaluation/           # Protocol, metrics, summaries, diagnostics
|   `-- experiments/          # Reproducible experiment orchestration
|-- tests/                    # Unit and integration contracts
|-- data_unzip.py             # Preserved transfer/extraction helper
|-- data_unzip_hera.py        # Preserved transfer/extraction helper
`-- requirements.txt
```

The two `data_unzip*.py` helpers are retained byte-for-byte for existing data
transfer workflows. They are not part of the canonical preparation pipeline.
See [`config/README.md`](config/README.md) for the complete configuration map.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins CUDA 12.8 PyTorch wheels suitable for the Blackwell
production environment. Install a compatible official PyTorch build instead
when using a different CUDA runtime or GPU generation. The first model run may
download pretrained weights from Hugging Face.

All commands below are run from the repository root. Dataset paths are
repository-relative and default to `data/DGM4` and `data/SAMM`.

## Data layout and validation

The repository does not redistribute either dataset. Obtain each release under
its original license and create the following layouts.

### DGM4

```text
data/DGM4/
|-- train.json
|-- val.json
|-- test.json
`-- images/
```

Run the frozen split, schema, path, image, box, and text-position checks:

```bash
python scripts/prepare_data.py config/datasets/dgm4/prepare.yaml
```

### SAMM-20K

```text
data/SAMM/
|-- train.json
|-- val.json
|-- test.json
`-- images/
    |-- orig_output/
    |-- swap_jpg/
    `-- emotion_jpg/
```

Validate the frozen protocol hashes, annotation schema, image paths, clipped
boxes, and text localization:

```bash
python scripts/prepare_data.py config/datasets/samm/prepare.yaml
```

Invalid or unavailable grounding annotations are represented by explicit masks;
they are not silently converted into negative labels or removed from the
classification protocol.

## DGM4: end-to-end

### Preflight and smoke tests

```bash
python scripts/prepare_data.py config/datasets/dgm4/prepare.yaml
python scripts/hera_smoke.py config/datasets/dgm4/hera/smoke.yaml
python scripts/model_smoke.py config/datasets/dgm4/baselines/b1/smoke.yaml
```

### Canonical HERA

```bash
python scripts/train.py config/datasets/dgm4/hera/train.yaml
python scripts/evaluate.py config/datasets/dgm4/hera/evaluate.yaml
```

### Same-backbone B1

```bash
python scripts/train.py config/datasets/dgm4/baselines/b1/train.yaml
python scripts/evaluate.py config/datasets/dgm4/baselines/b1/evaluate.yaml
```

The lighter global-fusion B0 control is available at
`config/datasets/dgm4/baselines/b0/` and uses the same `model_smoke.py`,
`train.py`, and `evaluate.py` entry points.

## SAMM-20K: end-to-end

### Preflight and smoke tests

```bash
python scripts/prepare_data.py config/datasets/samm/prepare.yaml
python scripts/hera_smoke.py config/datasets/samm/hera/smoke.yaml
python scripts/model_smoke.py config/datasets/samm/baselines/b1/smoke.yaml
```

### Canonical HERA

```bash
python scripts/train.py config/datasets/samm/hera/train.yaml
python scripts/evaluate.py config/datasets/samm/hera/evaluate.yaml
```

### Same-backbone B1

```bash
python scripts/train.py config/datasets/samm/baselines/b1/train.yaml
python scripts/evaluate.py config/datasets/samm/baselines/b1/evaluate.yaml
```

The normal evaluator discovers the latest compatible run under the configured
training output root. To evaluate the archived seed-42 checkpoint at the
server path encoded by the dedicated config:

```bash
python scripts/evaluate.py config/datasets/samm/hera/evaluate_seed42.yaml
```

To reproduce the target-free text-grounding aggregation diagnostic on the same
checkpoint:

```bash
python scripts/evaluate.py config/datasets/samm/hera/text_grounding_diagnostic.yaml
```

The archived checkpoint paths in these two specialized YAML files are
environment-specific; override them in a local derived config when the files
are stored elsewhere.

## Controlled experiments

Experiment runners materialize effective per-job configs, validate them, and
execute jobs sequentially. Inspect a suite without starting a GPU process by
adding `--dry-run`.

### DGM4 ablations

```bash
python scripts/run_hera_ablation.py config/experiments/dgm4/ablation.yaml --dry-run
python scripts/run_hera_ablation.py config/experiments/dgm4/ablation.yaml
```

The registered suite tests routing, evidence typing, graph operator,
hypothesis feedback, grounding conditioning, and the attribution-derived binary
decision.

### DGM4 cycle-depth study

```bash
python scripts/run_hera_cycle_study.py config/experiments/dgm4/cycle_study.yaml --dry-run
python scripts/run_hera_cycle_study.py config/experiments/dgm4/cycle_study.yaml
```

### DGM4 canonical multi-seed study

```bash
python scripts/run_hera_multiseed.py config/experiments/dgm4/multiseed.yaml --dry-run
python scripts/run_hera_multiseed.py config/experiments/dgm4/multiseed.yaml
```

The broader matched scientific study can be materialized without GPU work, or
materialized and run in one invocation:

```bash
python scripts/run_scientific_study.py --suite config/experiments/dgm4/scientific_v3.yaml
python scripts/run_scientific_study.py --suite config/experiments/dgm4/scientific_v3.yaml --run
```

### SAMM key ablations

```bash
python scripts/run_hera_ablation.py config/experiments/samm/key_ablations.yaml --dry-run
python scripts/run_hera_ablation.py config/experiments/samm/key_ablations.yaml
```

### SAMM matched multi-seed study

```bash
python scripts/run_samm_matched_multiseed.py config/experiments/samm/matched_multiseed.yaml --dry-run
python scripts/run_samm_matched_multiseed.py config/experiments/samm/matched_multiseed.yaml
```

This final suite runs canonical HERA, same-backbone B1, `no_router`, and
`generic_grounding` sequentially for the prespecified seeds. By default it
deletes a job's large checkpoints only after validating its complete test
summary and prediction CSV. Pass `--keep-checkpoints` to retain them.

### Cross-dataset evidence routing

After producing HERA prediction CSVs for both datasets:

```bash
python scripts/compare_evidence_routing.py \
  --input DGM4=outputs/phase7_final_clean_eval/<dgm4-run>/predictions/test_predictions.csv \
  --input SAMM=outputs/samm/hera_eval/<samm-run>/predictions/test_predictions.csv \
  --reference DGM4 \
  --comparison SAMM \
  --output outputs/routing_dgm4_vs_samm
```

This analysis compares how the same router redistributes evidence when the
dataset ontology and cross-modal consistency structure change.

## Resume, checkpoints, and artifacts

- Training writes `last.pt` for automatic recovery and `best.pt` for model
  selection.
- Re-running the same training config resumes the latest compatible incomplete
  run when `training.auto_resume.enabled` is true.
- Evaluation resolves the selected checkpoint and its validation thresholds;
  explicit paths take precedence in specialized configs.
- Each run stores its resolved configuration, environment/protocol metadata,
  metrics, reports, and prediction artifacts in an isolated output directory.
- Multi-seed retention logic never removes a checkpoint before validating the
  expected evaluation artifacts.

For long experiments, archive `best.pt`, its SHA256 sidecar, the threshold JSON,
the resolved config, and the evaluation summary together.

## Verification

Run the complete repository test suite after code or configuration changes:

```bash
python -m pytest -q
```

Useful non-training checks include:

```bash
python -m compileall -q src scripts tests
python scripts/run_hera_ablation.py config/experiments/samm/key_ablations.yaml --dry-run
python scripts/run_samm_matched_multiseed.py config/experiments/samm/matched_multiseed.yaml --dry-run
```

The test suite covers configuration inheritance, data contracts, target-free
forward paths, ontology adaptation, Noisy-OR behavior, grounding aggregation,
training-stage transitions, checkpoint resume, experiment materialization, and
the canonical repository layout.

## Reproducibility boundaries

- A comparison is meaningful only when dataset release, frozen splits,
  preprocessing, backbone revisions, seed policy, threshold tuning, and metric
  definitions match.
- Grounding metrics can change with tokenizer alignment, annotation clipping,
  image resizing, and availability policy; preserve the emitted protocol
  metadata with reported results.
- Router reliability scores quantify learned evidence utility within this
  model. They should not be interpreted as calibrated uncertainty.
- Grounding predictions identify supervised localization evidence; they do not
  by themselves prove causal faithfulness.
- Auxiliary configs under `config/benchmarks/` are independent binary
  benchmark adapters and are not part of the DGM4/SAMM attribution protocol.

The implementation contract is documented in
[`docs/HERA_DGM_CONSTITUTION.md`](docs/HERA_DGM_CONSTITUTION.md). Dataset terms,
licenses, and redistribution restrictions remain those of the original dataset
providers.
