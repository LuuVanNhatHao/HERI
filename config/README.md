# Configuration index

YAML files describe dataset identity, architecture, optimization and evaluation. Paths distinguish single-run entries from controlled experiment manifests.

```text
config/
├── datasets/
│   ├── dgm4/
│   │   ├── prepare.yaml
│   │   ├── heri/{train,evaluate}.yaml
│   │   └── baselines/{b0,b1}/{train,evaluate}.yaml
│   └── samm/
│       ├── prepare.yaml
│       ├── heri/{train,evaluate}.yaml
│       └── baselines/b1/{train,evaluate}.yaml
└── experiments/
    ├── dgm4/           Reasoning, component, progressive and matched-seed studies
    └── samm/           Matched controls and sufficiency comparisons
```

SAMM inherits the shared DGM4 model/runtime contract and overrides paths, hashes, ontology, text length, output location and selection metric. Keep inherited settings in their owning file instead of copying whole configurations. Generated effective YAML belongs with run artifacts and must not be edited in place.

## Single-run entries

```bash
python -B scripts/heri.py prepare-data config/datasets/dgm4/prepare.yaml
python -B scripts/heri.py train config/datasets/dgm4/heri/train.yaml
python -B scripts/heri.py evaluate config/datasets/dgm4/heri/evaluate.yaml
```

Substitute `samm` for the second protocol. Use `baselines/b0` or `baselines/b1` where provided. Output roots are under `artifacts/`; checkpoint resolution and annotation paths remain explicit in YAML.

Losses use `per_modality_v1`: a sufficiency unit is a positive hypothesis with an applicable, annotated modality and at least one local evidence node. The named `any_modality_or` control is separate and cannot share a checkpoint identity.

Primary thresholds are 0.5. An image prediction averages only hypotheses whose individual probability reaches 0.5; an empty active set emits an empty box. Text aggregation follows Eq.21 (`relative_soft`). DGM4 selects by `cf1_fixed`; SAMM selects by released three-label `samm_official_cf1`.

Training seed 7 and matched seeds 7/21/101 are independent of SAMM subset seed 777. The schedule is 20 epochs, batch 16, frozen encoders for epochs 1–3 and full adaptation from epoch 4. AdamW uses encoder LR `2e-6`, module LR `1e-4`, weight decay 0.01, 500 warm-up steps and one global linear-decay clock. Early stopping is disabled.

## Study entries

| Manifest under `config/experiments/` | CLI study type | Scope |
| --- | --- | --- |
| `dgm4/reasoning_controls.yaml` | `scientific` | Matched backbone/query and seven reasoning controls |
| `dgm4/progressive.yaml` | `scientific` | Table 7 progressive construction |
| `dgm4/components.yaml` | `scientific` | Component and attention controls |
| `dgm4/matched_components.yaml` | `scientific` | Three-seed component controls |
| `dgm4/cycle_study.yaml` | `cycle` | Focused cycle/refresh study |
| `dgm4/ablation.yaml` | `ablation` | Focused component removals |
| `dgm4/multiseed.yaml` | `multiseed` | Matched-seed orchestration |
| `samm/matched_multiseed.yaml` | `samm` | B1, static, full, router and grounding controls |
| `samm/key_ablations.yaml` | `ablation` | Focused SAMM ablations |
| `samm/evidence_sufficiency_ablation.yaml` | `ablation` | Sufficiency-protocol comparison |

```bash
python -B scripts/heri.py study scientific config/experiments/dgm4/reasoning_controls.yaml --dry-run
python -B scripts/heri.py study samm config/experiments/samm/matched_multiseed.yaml --dry-run
```

Remove `--dry-run` to execute. Scientific studies additionally support `--variants`, `--seeds` and `--resume`. Resume targets a job's own checkpoint; source fingerprint and effective configuration hash must agree. Incomplete jobs remain pending in aggregate reports.

The intermediate readouts and control operators are defined in [graph reasoning](../docs/method/graph_reasoning.md). The [implementation reference](../docs/method/paper_contract.md) defines equations, masks and experimental identity; [experiment documentation](../docs/experiments.md) describes artifacts and comparisons.
