# HERA-DGM

HERA-DGM is governed by the frozen Phase 0 Constitution in `docs/HERA_DGM_CONSTITUTION.md`.

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

The production ablation suite reuses the canonical DGM4 split, three-stage
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
Rerunning the same command resumes an interrupted variant from its latest
`last.pt`, skips variants with a valid `training_complete.json`, and skips an
evaluation whose `summary.csv` already exists. Checkpoints are written after
each epoch, so an interruption in the middle of an epoch repeats that epoch.
