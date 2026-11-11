# HERI verification

Run the bounded offline checks from the repository root:

```console
python -B scripts/heri.py validate
```

The validation command selects hand-computed equation cases, small runtime counterexamples and two-sample end-to-end integration checks. It loads no pretrained weights and writes its report under `artifacts/validation/latest/`; scratch files are removed after the run.

| Location | Responsibility |
| --- | --- |
| `contracts/` | Mathematical invariants, experiment settings, run identity and repository interfaces |
| `integration/` | Data flow between HERI, dataset adapters, losses, evaluation and experiment orchestration |
| `unit/data/` | Annotation parsing, image geometry, token alignment and collation |
| `unit/evidence/`, `unit/routing/`, `unit/graph/` | Evidence construction, selection, reliability and reasoning |
| `unit/models/`, `unit/losses/` | Heads, encoder contracts and optimization objectives |
| `unit/metrics/`, `unit/evaluation/`, `unit/training/` | Metrics, exports, precision, scheduling and checkpoints |
| `support/` | Shared synthetic fixtures with explicit imports |

For a specific component, use `python -B -m pytest tests/unit/graph -q`. The full collection includes component regressions and equation checks; the validation command selects a bounded subset for routine verification. Use `-B` for direct Python invocations so they do not create bytecode directories.

The equation-to-source map is in [the implementation reference](../docs/method/paper_contract.md). Synthetic checks verify formulas and data flow; dataset-level results are produced by the training and evaluation commands.
