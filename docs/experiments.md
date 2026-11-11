# Experiment workflow

Every result starts from a resolved YAML configuration, observed annotation identity and compatible source/checkpoint identity. The [configuration index](../config/README.md) maps datasets and study manifests; the [implementation reference](method/paper_contract.md) defines the formulas and decisions being compared.

## Prepare and verify data

Run `prepare-data` before training. It checks annotation structure and paths. SAMM reconstruction is a separate annotation-only step because its frozen subset uses a fixed ID and serialization recipe. Local official SAMM source annotations are kept under `data/sources/SAMM/`.

```bash
python -B scripts/heri.py reconstruct-samm --source-root data/sources/SAMM --output-root data/SAMM
python -B scripts/heri.py prepare-data config/datasets/samm/prepare.yaml
```

Canonical loaders measure actual annotation SHA-256 and reject mismatches. Configured hashes and observed hashes remain separate artifact fields. Localization annotation availability is explicit; missing boxes or word annotations do not remove samples from classification.

## Train and evaluate one model

```bash
python -B scripts/heri.py train config/datasets/dgm4/heri/train.yaml
python -B scripts/heri.py evaluate config/datasets/dgm4/heri/evaluate.yaml
```

The evaluator resolves the checkpoint according to YAML. Use the same dataset, model, seeds and objective identities as training. Resume restores optimizer/scaler/scheduler state and loader/sampler random streams; it rejects incompatible seed, architecture and protocol state. Checkpoints are written atomically.

The selected checkpoint maximizes validation CF1 under fixed 0.5 decisions. SAMM uses its explicitly projected three-label released classification metric. The 20-epoch schedule continues across encoder unfreezing; a resumed stage retains the same global learning-rate clock.

## Run controlled studies

```bash
python -B scripts/heri.py study scientific config/experiments/dgm4/reasoning_controls.yaml --dry-run
python -B scripts/heri.py study cycle config/experiments/dgm4/cycle_study.yaml --dry-run
python -B scripts/heri.py study ablation config/experiments/dgm4/ablation.yaml --dry-run
python -B scripts/heri.py study multiseed config/experiments/dgm4/multiseed.yaml --dry-run
python -B scripts/heri.py study samm config/experiments/samm/matched_multiseed.yaml --dry-run
```

`--dry-run` materializes configurations and pending summaries without training. Remove it on the training machine to execute. Scientific suites support variant/seed selection and per-job resume. Matched seeds are 7, 21 and 101; seed 777 belongs only to SAMM subset construction.

The seven reasoning conditions are single pass, static two-cycle, semantic-only refresh, hypothesis-only refresh, initial-hypothesis control, full HERI and a three-cycle extension. Additional manifests compare components, attention and progressive construction. Their operators, readouts and combined changes are defined in [graph reasoning](method/graph_reasoning.md). Progressive transitions are cumulative architectural comparisons; they should not be interpreted as single-factor removals.

## Inspect artifacts

New outputs live under `artifacts/experiments/`; preflight reports live under `artifacts/data/`. An evaluated run records effective configuration, source/environment fingerprint, observed data hashes, checkpoint identity, prediction CSV, metrics JSON, summary CSV and threshold policy. Studies retain manifests and per-job state linking aggregate rows to source runs.

Each checkpoint has a `<checkpoint>.metadata.json` sidecar containing its
SHA-256, UUID, immutable configuration snapshot and digest, source fingerprint
and alignment-objective identity. Evaluation writes per-row identities and
seals the configuration, environment, prediction, summary, checkpoint and
annotation hashes in `run_evidence.json`.

Primary outputs include binary accuracy/AUC/EER, attribution mAP/CF1/OF1, original-frame image IoU and native-token/original-word text F1. CF1 is the harmonic mean of class-mean precision and recall. Token and word metrics keep separate names and coverage. EDC/UDR use predicted-positive valid hypotheses; VSR@1 additionally requires a true-positive hypothesis and correctly localized confident top-1 support. Baselines have no hypothesis-specific EDC/UDR/VSR; empty diagnostic populations return null.

Aggregation includes completed jobs with compatible data, configuration and evaluation identities. Missing jobs remain pending and are listed in the study report.

## Experimental provenance

A study configuration defines an experiment; a materialized manifest defines
its planned jobs. Neither is evidence that training or evaluation completed.
Associate a reported result with an executed design using its resolved training
and evaluation configurations, source identity, observed dataset hashes,
checkpoint identifier and file digest, predictions, and evaluation summary.
Record the selected checkpoint and seed for every aggregate observation.

The four intermediate progressive designs are `evidence_decomposition`,
`reliability_router`, `hypothesis_evidence_graph` and
`constrained_single_pass`. Original training artifacts for these four stages
are unavailable. The configurations specify executable designs; their
measurements must come from new runs. They do not establish equivalence with
previously reported table entries. A display name or matching score cannot
replace the missing experimental evidence.

Keep imported observations linked to their original artifacts. Verify their
checkpoint/configuration/evaluation relationship before including them in
an aggregate. Record unavailable artifacts as missing evidence rather than
filling result fields from a manuscript table. Statistical comparisons use
matched seeds and the full planned job population; report incomplete pairs.

Audit a materialized study manifest before interpreting its results:

```bash
python -B scripts/heri.py audit-study path/to/manifest.json --output path/to/audit.json
```

The audit checks consistency among resolved training/evaluation configurations,
checkpoint and metadata digests, annotation/prediction identifiers and summary
identity. It reports each job as `missing`, `inconsistent` or
`internally_consistent`. The last status establishes agreement among the
available artifacts; it does not establish original-run identity or recompute
benchmark performance. Scientific aggregation requires this artifact
consistency, and incomplete jobs retain their place in the study denominator.
Scientific manifests declare `evidence_policy: artifact_consistency`; reports
are written to `study_results.json`. Metadata-only archives without this policy
are reported as `unverified`, even when their summaries are complete.

Materialized studies also record their model designs and the mechanism/configuration
changes between adjacent designs. Use those differences when interpreting a
contrast. Save the audit with the study artifacts so readers can inspect the
evidence behind each aggregate observation.

Audits compare scientific settings with the immutable checkpoint snapshot;
resume bookkeeping may change the run-level configuration without replacing
that training identity. Seed intervals describe variation on one fixed test
split; the reported intervals are exploratory and unadjusted for multiple
comparisons.

## Verify the implementation

```bash
python -B scripts/heri.py validate
```

The bounded suite checks formulas, masks, graph refresh, gradients, resume and synthetic end-to-end evaluation on CPU. [Verification documentation](validation/local_checks.md) describes test coverage and the requirements for full experiments.
