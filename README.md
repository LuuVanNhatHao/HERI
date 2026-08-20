# HERA-DGM

HERA-DGM is governed by the frozen Phase 0 Constitution in `docs/HERA_DGM_CONSTITUTION.md`.

## Final protocol (`hera_final_v1`)

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

## Binary multimodal benchmarks

MultiBanFakeDetect and VLDBench use a separate task-aware delivery. It reuses
the production HERA backbone, evidence extraction, reliability routing, HGT,
and iterative hypothesis reasoning, but interprets the four graph states as
latent evidence slots. It does not instantiate or supervise DGM4 attribution
and grounding heads. Dataset-native auxiliary signals (`subtype` for
MultiBanFakeDetect and `text_label` for VLDBench) are independently gated by
`task.signals` and matching loss weights; both are off in the canonical YAMLs.

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
  --batch-size 4 --batches 2 --gradient-accumulation-steps 1

python scripts/train_binary_benchmark.py config/benchmark_multibanfakedetect.yaml
python scripts/evaluate_binary_benchmark.py config/benchmark_multibanfakedetect.yaml

python scripts/binary_benchmark_capacity_test.py \
  config/benchmark_vldbench.yaml \
  --batch-size 2 --batches 2 --gradient-accumulation-steps 1

python scripts/train_binary_benchmark.py config/benchmark_vldbench.yaml
python scripts/evaluate_binary_benchmark.py config/benchmark_vldbench.yaml
```

Each YAML owns an output root and auto-resumes from its newest `last.pt`.
Stage A freezes every backbone for three epochs; Stage B fully unfreezes all
three backbones. The first Stage-B backward pass writes a gradient-coverage
audit proving that the full backbones, rather than only their last blocks,
receive gradients.
