# Verification

The verification suite connects the mathematical definitions in the
[implementation reference](../method/paper_contract.md) to executable tests.
It checks individual operations and their composition through prediction,
supervision and evaluation.

## Run the bounded suite

From the repository root:

```bash
python -B scripts/heri.py validate
```

The command uses CPU FP32, one compute thread and synthetic samples. Pretrained
model downloads and CUDA are disabled. The launcher suppresses Python bytecode
and pytest caching, removes temporary test files after execution and writes
JUnit results and a JSON summary to `artifacts/validation/latest/`.

Individual groups can be selected with `--section`:

| Group | Coverage |
| --- | --- |
| `equations` | Evidence selection, graph attention, grounding, objectives and diagnostic metrics |
| `runtime` | Checkpoint identity, random streams, atomic writes, precision and schedule continuity |
| `integration` | Connected forward/backward and evaluation paths for both dataset ontologies |
| `config` | Experiment manifests, CLI options and repository interfaces |

For component-specific regressions, run a test module or directory directly:

```bash
python -B -m pytest tests/unit/graph -q
```

See [test organization](../../tests/README.md) for module responsibilities.

## Equation coverage

- **Decision and supervision (Eq.10, Eq.23-28).** Stable Noisy-OR is checked at
  extreme logits with finite gradients. Binary supervision detaches absent
  hypotheses on manipulated samples. Grounding losses honor hypothesis,
  modality, annotation and position masks; empty supervision populations return
  differentiable zero.
- **Evidence and graph reasoning (Eq.12-16).** Tests check learned saliency
  gradients, greedy saliency/diversity selection, character-span correspondence,
  graph isolation, ontology and hand-computed confidence-aware attention.
  Refresh tests inspect actual edges for semantic-only, hypothesis-only and
  initial-hypothesis controls.
- **Grounding (Eq.18-22).** Tests cover masked softmax, empty stores, sigmoid
  center-size boxes, probability weighting, active-hypothesis image selection,
  letterbox removal and restoration to original image coordinates. Empty
  predictions on manipulated images receive IoU zero.
- **Image-loss reduction (Eq.26).** Patch BCE is averaged within each valid
  image/hypothesis pair, then pairs are averaged equally. Unequal valid patch
  counts are checked explicitly.
- **Alignment and reliability (Eq.29-30).** Hand-computed cases use unequal
  visual/text position counts and the printed visual-count denominator.
  Calibration tests check the pooled-node regularizer and detached teacher.
- **Verified evidence (Eq.31-33).** Tests check EDC's unweighted mean of clipped
  support ratios, UDR's support threshold and the confidence/location
  requirements of top-1 verified support.

## Integration coverage

Two synthetic examples per dataset ontology exercise evidence selection,
reliability routing, native HGT, shared two-cycle reasoning, grounding heads,
the complete objective and backward pass. Feature widths and node budgets are
reduced while the production operators remain in use. Component removals,
progressive readouts and the backbone-matched B1 baseline are also exercised.

Target perturbation checks ensure inference predictions do not depend on
supervision labels. Graph selection preserves dense positions used by the
grounding heads. Evaluation tests check original-frame geometry, native and
released SAMM outputs, per-cycle hypothesis probabilities and CSV export.
Localization availability masks remain separate from original-word metadata
completeness when aggregating classification results.

The SAMM annotation recipe is checked on a small source fixture for fixed
sampling RNG, sorted training IDs, validation order, field order, UTF-8,
compact JSON and newline conventions. The reconstruction command additionally
verifies the expected hashes of all generated split files.

## Runtime and experiment integrity

Resume checks reject incompatible architecture, protocol and seed identities.
Separate sampler/worker random streams reproduce the next epoch's order.
Interrupted checkpoint writes preserve the previous file. Precision checks
verify unscaling before gradient clipping; stage transitions preserve Adam
state and the global learning-rate schedule.

Experiment tests validate fixed thresholds, registered controls and matched
seeds 7/21/101. Dry-run materialization validates full effective configurations
without loading a model. Aggregate reports retain incomplete jobs as pending.

## Full experimental validation

Synthetic verification establishes the tested implementation behavior.
Benchmark performance, resource measurements and statistical comparisons
require the configured pretrained backbones, complete datasets and full
training/evaluation runs. Optional GAT controls additionally require
`torch-geometric`; canonical HGT uses native PyTorch attention.
