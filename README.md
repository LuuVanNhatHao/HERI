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
