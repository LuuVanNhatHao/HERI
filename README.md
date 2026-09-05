# HERA-DGM

HERA-DGM is governed by `docs/HERA_DGM_CONSTITUTION.md` and its scientific-v3 amendment.

## Scientific evaluation v3

The current evaluator corrects CF1, includes all samples in grounding, and reports
original-word truncation coverage. Historical CF1/grounding scores are not
interchangeable with v3 outputs. Legacy weights can be evaluated explicitly;
legacy checkpoint-selection state cannot resume into v3 training.

The matched core suite adds B1+noisy-OR and learned-query controls alongside
static-cycle and full HERA. Results remain pending until validated server runs
are available.

```bash
# Prepare 15 matched jobs, without starting GPU training.
python scripts/run_scientific_study.py
```

Use the new isolated scientific-v3 manifest for revised paper experiments.
The older phase/cycle commands below remain available as historical workflows.

## Final protocol (`hera_final_v2`)

The final implementation intentionally breaks compatibility with earlier
checkpoints. It never supplies DGM4 `mtcnn_boxes` to a model, predicts image
boxes directly from the full DINOv3 patch grid, routes FS/FA and TS/TA
grounding at inference using predicted atomic probabilities only, applies
edge confidence inside HGT attention/messages, and performs exactly two graph
reasoning cycles with a target-independent graph rebuild between cycles.
Legacy checkpoints must not be reused; train/evaluate under the isolated
`outputs/phase7_final_clean*` roots.

Phase 1 implements only DGM4 dataset infrastructure:

- schema validation;
- label mapping;
- local image path resolution;
- dataset preflight;
- image resize/padding and bbox transforms;
- `fake_text_pos` word-to-subword alignment utilities;
- `DGM4Dataset`, collate, and DataLoader factory;
- data tests and smoke inspection scripts.

Canonical commands:

```bash
python scripts/prepare_data.py config/prepare_dgm4.yaml
python scripts/inspect_batch.py config/prepare_dgm4.yaml
```

All runtime paths, parameters, and settings must come from YAML.

Phase 2 adds controlled strong-backbone baselines only:

- B0 global fusion baseline;
- B1 cross-attention baseline;
- real SigLIP2, DINOv3, and DeBERTa wrapper code;
- shared losses, metrics, threshold tuning, training, checkpoint, and evaluation infrastructure.

Canonical Phase-2 commands:

```bash
python scripts/model_smoke.py config/smoke_baseline_b0.yaml
python scripts/model_smoke.py config/smoke_baseline_b1.yaml
python scripts/train.py config/train_baseline_b0.yaml
python scripts/train.py config/train_baseline_b1.yaml
python scripts/evaluate.py config/evaluate_baseline_b0.yaml
python scripts/evaluate.py config/evaluate_baseline_b1.yaml
```

Local resource-dependent checks are recorded in
`outputs/phase2/deferred_server_validation.json`.

## CPU/DataLoader tuning on GPU servers

Canonical server training uses 16 single-threaded DataLoader workers with two
prefetched batches per worker. This matches a 32-vCPU host while preventing
PyTorch and Hugging Face tokenizer thread pools from multiplying inside each
worker. Pinned batches are transferred to CUDA non-blockingly.

Measure the host-side input pipeline before changing the canonical worker
count:

```bash
python scripts/dataloader_benchmark.py config/train_hera.yaml \
  --workers 8 12 16 24 \
  --warmup-batches 10 \
  --batches 100
```

Use the reported `best_num_workers` only when it improves end-to-end training
`samples/s`; the complete HERA full-fine-tune remains primarily GPU-bound.

## Controlled HERA ablations

The production ablation suite reuses the canonical DGM4 split, two-stage
fine-tuning protocol, optimizer, scheduler, thresholds, and final test
evaluation. It contains six HERA-only experiments; B0 and B1 are deliberately
excluded because they are reported as standalone baselines:

- `no_router`;
- `homogeneous_evidence`;
- `gatv2`;
- `no_hypothesis_feedback`;
- `generic_grounding`;
- `independent_binary`.

Validate and materialize all effective configs on a CPU machine without
loading any backbone or starting CUDA:

```bash
python scripts/run_hera_ablation.py \
  config/experiments/hera_ablation.yaml \
  --dry-run
```

On the training server, verify the real installed PyG graph operators with
small synthetic CPU tensors before committing GPU time. This test does not
download or load SigLIP2, DINOv3, DeBERTa, or DGM4:

```bash
python -m pytest -q \
  tests/test_hera_ablation_config.py \
  tests/test_hera_ablation_components.py
```

Run the complete suite sequentially on the GPU server:

```bash
python scripts/run_hera_ablation.py config/experiments/hera_ablation.yaml
```

Run or resume one variant, then evaluate it separately if needed:

```bash
python scripts/run_hera_ablation.py \
  config/experiments/hera_ablation.yaml \
  --variants no_router \
  --phase train

python scripts/run_hera_ablation.py \
  config/experiments/hera_ablation.yaml \
  --variants no_router \
  --phase evaluate
```

Every variant owns an isolated directory under `outputs/hera_ablation/`.
Training auto-resume therefore cannot select another variant's checkpoint.
Evaluation rows are combined into `outputs/hera_ablation/ablation_summary.csv`.
Each ablation has a 12-epoch maximum with the same early-stopping policy.
Rerunning the same command resumes an interrupted variant from its latest
`last.pt`, skips variants with a valid `training_complete.json`, and skips an
evaluation whose `summary.csv` already exists. Checkpoints are written after
each epoch, so an interruption in the middle of an epoch repeats that epoch.

## Controlled graph-cycle study

The cycle-depth study is separate from the six component ablations. It holds the
DGM4 split, HERA modules, backbones, losses, staged full fine-tuning, batch
size, and threshold protocol fixed while changing only graph pass depth and
inter-pass reconstruction:

- `single_pass`: one graph pass and no inter-pass rebuild;
- `two_cycle_static`: two passes over one frozen topology and frozen edge
  attributes while node states continue to update;
- `full_hera`: canonical two-pass state-dependent graph reconstruction;
- `three_cycle_extension`: three-pass state-dependent saturation probe.

Materialize and validate all four complete configurations without loading a
model or starting CUDA:

```bash
python scripts/run_hera_cycle_study.py \
  config/experiments/hera_cycle_study.yaml \
  --dry-run
```

Before committing to the three-cycle run, test its exact Stage-B memory path on
the GPU server using the generated configuration:

```bash
python scripts/full_ft_capacity_test.py \
  outputs/hera_cycle_study/three_cycle_extension/configs/train.yaml \
  --batch-size 16 --batches 2
```

If that exact test reports OOM, change `micro_batch_size` and
`gradient_accumulation_steps` once in the suite manifest so the effective batch
is preserved, then regenerate all four configs. Do not reduce only the
three-cycle variant, because that would break the matched-protocol control.

Run all variants sequentially, or resume one selected variant:

```bash
python scripts/run_hera_cycle_study.py \
  config/experiments/hera_cycle_study.yaml \
  --prune-last-after-evaluation

python scripts/run_hera_cycle_study.py \
  config/experiments/hera_cycle_study.yaml \
  --variants three_cycle_extension
```

Each variant has an isolated auto-resume root and the canonical 20-epoch
ceiling with the same early-stopping policy. Cross-variant checkpoint loading is rejected. The
study result table is written to
`outputs/hera_cycle_study/cycle_study_summary.csv`; CF1 deltas are computed
from the measured single-pass row and are never embedded as assumed numbers.
The router is not rerun between graph passes, so reports must describe this as
state-dependent graph/evidence reconstruction rather than reliability
re-estimation.
On storage-limited servers, `--prune-last-after-evaluation` removes only the
resume checkpoint after a valid test summary has been written and preserves
`best.pt`. Without this explicit flag, the runner never deletes checkpoints.

## Canonical HERA multi-seed runs

The final reproducibility suite runs canonical HERA only (no B0/B1 and no
ablation variants) under three configured seeds. It inherits the complete
architecture, staged full-fine-tuning, optimization, batch-size,
early-stopping, and threshold protocol from `config/train_hera.yaml` and
`config/evaluate_hera.yaml`.

Validate all generated production configs without loading a model or GPU:

```bash
python scripts/run_hera_multiseed.py \
  config/experiments/hera_multiseed.yaml \
  --dry-run
```

Run all three seeds sequentially, including each seed's test evaluation:

```bash
python scripts/run_hera_multiseed.py config/experiments/hera_multiseed.yaml
```

Resume or target a subset without changing seed identity:

```bash
python scripts/run_hera_multiseed.py \
  config/experiments/hera_multiseed.yaml \
  --seeds 7 21 101
```

Each seed has an isolated auto-resume root under `outputs/hera_multiseed/`.
Rerunning the command resumes from that seed's `last.pt`; a seed with an
existing test `summary.csv` is skipped even if its checkpoints were later
deleted. Final per-seed results and paper-ready mean/sample-standard-deviation
tables are written to `outputs/hera_multiseed/seed_results.csv` and
`outputs/hera_multiseed/multiseed_statistics.csv`.

## Binary multimodal benchmarks

MultiBanFakeDetect and VLDBench use a separate task-aware delivery. It reuses
the production HERA backbone, evidence extraction, reliability routing, HGT,
and iterative hypothesis reasoning, but interprets the four graph states as
latent evidence slots. It does not instantiate or supervise DGM4 attribution
and grounding heads. Dataset-native auxiliary signals (`subtype` for
MultiBanFakeDetect and `text_label` for VLDBench) are independently gated by
`task.signals` and matching loss weights; both are off in the canonical YAMLs.
The canonical MultiBanFakeDetect configuration is a dataset-scale compact
delivery: `google/siglip2-base-patch16-384`,
`facebook/dinov3-vits16-pretrain-lvd1689m`, and the 13M
`csebuetnlp/banglabert_small` contextual encoder. It preserves 384-pixel input
geometry and uses the pinned official Bengali normalization pipeline. The
normalizer is applied once when each dataset split is materialized, before both
contextual tokenization and multimodal reasoning. Its isolated
`multibanfakedetect_compact_v1` output root cannot auto-resume an older
large-backbone run.
VLDBench uses `answerdotai/ModernBERT-large` as its contextual encoder with a
2,048-token ceiling. Documents shorter than that ceiling remain unpadded until
batch collation. Longer documents use a deterministic headline + article lead
+ article tail policy rather than prefix-only truncation. The preparation audit
records the original length distribution, truncation rate, and retained-token
ratio. This delivery writes to the isolated
`vldbench_modernbert_long_v1` output root and cannot resume the earlier
DeBERTa/512-token run.

Prepare and audit the real releases before training:

```bash
python scripts/prepare_binary_benchmark.py config/benchmark_multibanfakedetect.yaml
python scripts/prepare_binary_benchmark.py config/benchmark_vldbench.yaml
```

VLDBench ships as one Parquet split, so preparation creates a deterministic,
image-hash-grouped 70/15/15 manifest. MultiBanFakeDetect preserves its official
80/10/10 protocol for prior-work comparability and records its released
cross-split duplicates in the audit report instead of silently changing it.

On the GPU server, test the exact full-fine-tune workload without writing a
checkpoint, then train/evaluate:

```bash
python scripts/binary_benchmark_capacity_test.py \
  config/benchmark_multibanfakedetect.yaml \
  --batch-size 16 --batches 4 --gradient-accumulation-steps 1

python scripts/train_binary_benchmark.py config/benchmark_multibanfakedetect.yaml
python scripts/evaluate_binary_benchmark.py config/benchmark_multibanfakedetect.yaml

python scripts/binary_benchmark_capacity_test.py \
  config/benchmark_vldbench.yaml \
  --batch-size 1 --batches 16 --gradient-accumulation-steps 16

python scripts/train_binary_benchmark.py config/benchmark_vldbench.yaml
python scripts/evaluate_binary_benchmark.py config/benchmark_vldbench.yaml
```

Each YAML owns an output root and auto-resumes from its newest `last.pt`.
Stage A freezes every backbone for three epochs; Stage B fully unfreezes all
three backbones. The first Stage-B backward pass writes a gradient-coverage
audit proving that the full backbones, rather than only their last blocks,
receive gradients. Evaluation checkpoints are selected only from Stage B, so
a frozen-warmup epoch can never be reported as the final benchmark model.
