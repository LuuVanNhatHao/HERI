# HERI

**Hypothesis-Mediated Evidence Reinterpretation for Grounded Multimodal Manipulation Verification**

Source code for HERI, a hypothesis-mediated framework for authenticity detection, manipulation attribution and image/text grounding on DGM4 and SAMM. The implementation combines four evidence families, reliability-aware graph reasoning and hypothesis-conditioned prediction heads.

The [implementation reference](docs/method/paper_contract.md) maps the accompanying manuscript's equations to source modules. [Architecture](docs/architecture.md) and [experiment documentation](docs/experiments.md) describe the data flow and reproduction workflow.

## Setup

Use a Python 3.10-compatible environment:

```bash
git clone https://github.com/LuuVanNhatHao/HERI.git
cd HERI
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The pinned PyTorch wheels target CUDA 12.8. Full experiments require sufficient GPU memory, pretrained-model access and separately obtained datasets. The launcher also works without installing the local package:

```bash
python -B scripts/heri.py --help
```

All supported operations use this launcher. It disables Python bytecode generation, including subprocesses. Local validation also disables pytest caching and cleans temporary files.

## End-to-end workflow

1. Place datasets in the structure below, under their original terms.
2. Run data preparation to verify annotations, paths and frozen dataset identity.
3. Train the selected model and evaluate its compatible selected checkpoint.
4. Use study manifests to run matched controls and aggregate completed results.

```text
data/
├── DGM4/
│   ├── {train,val,test}.json
│   └── images/...
└── SAMM/
    ├── {train,val,test}.json
    └── images/{orig_output,swap_jpg,emotion_jpg}/...
```

For frozen SAMM-20K annotations, first reconstruct the Appendix A subset from the official release:

```bash
python -B scripts/heri.py reconstruct-samm --source-root data/sources/SAMM --output-root data/SAMM
```

This annotation-only recipe checks expected hashes before writing and refuses to replace different existing files. It does not copy images. Dataset selection seed 777 is independent of training seeds.

Prepare, train and evaluate:

```bash
python -B scripts/heri.py prepare-data config/datasets/dgm4/prepare.yaml
python -B scripts/heri.py train config/datasets/dgm4/heri/train.yaml
python -B scripts/heri.py evaluate config/datasets/dgm4/heri/evaluate.yaml

python -B scripts/heri.py prepare-data config/datasets/samm/prepare.yaml
python -B scripts/heri.py train config/datasets/samm/heri/train.yaml
python -B scripts/heri.py evaluate config/datasets/samm/heri/evaluate.yaml
```

Dataset locations, run directories and checkpoint resolution are explicit in the effective YAML. B0/B1 use corresponding entries under `config/datasets/*/baselines/`. Training and evaluation verify observed annotation hashes; incompatible architecture, protocol and seed identities are rejected.

## Verification

```bash
python -B scripts/heri.py validate
python -B scripts/heri.py validate --section equations
```

The bounded suite runs CPU formula checks and two synthetic samples per ontology. It exercises the real evidence selector, router, HGT, grounding heads, loss, backward pass and evaluation exports without downloading pretrained weights or launching full training. Reports go to `artifacts/validation/latest/`. See [test coverage and execution requirements](docs/validation/local_checks.md).

## Experiments

Inspect effective jobs before launching workloads:

```bash
python -B scripts/heri.py study scientific config/experiments/dgm4/reasoning_controls.yaml --dry-run
python -B scripts/heri.py study scientific config/experiments/dgm4/progressive.yaml --dry-run
python -B scripts/heri.py study scientific config/experiments/dgm4/components.yaml --dry-run
python -B scripts/heri.py study scientific config/experiments/dgm4/matched_components.yaml --dry-run
python -B scripts/heri.py study samm config/experiments/samm/matched_multiseed.yaml --dry-run
```

Remove `--dry-run` to execute on the training machine. [Experiment documentation](docs/experiments.md) connects manifests, controls, artifacts and result identity; the [configuration index](config/README.md) lists entry points.

The canonical configuration uses hidden size 512, VS/VF/XA/TC budgets 96/96/64/64, semantic TopK 8 and two shared-weight reasoning cycles. Training uses seed 7, 20 epochs, batch 16, AdamW weight decay 0.01, module LR `1e-4` and encoder LR `2e-6`. Encoders freeze for epochs 1–3 and fully unfreeze from epoch 4. The global schedule has 500 warm-up steps followed by linear decay. Matched studies use seeds 7/21/101. Decisions use fixed 0.5; validation CF1 selects checkpoints and early stopping is disabled.

## Repository map

```text
config/                 Dataset entries and controlled experiment manifests
src/heri/               Importable implementation package
scripts/heri.py         Unified command-line entry point
tests/
├── unit/               Data, model, graph, loss, metric and training checks
├── integration/        Connected pipeline and adapter checks
├── contracts/          Paper, configuration, runtime and experiment contracts
└── support/            Shared synthetic fixtures
docs/
├── architecture.md     Module ownership, data flow and traceability
├── experiments.md      Training, evaluation and study workflow
├── method/             Equation map and graph/control definitions
└── validation/         Verification scope and execution records
artifacts/              Generated checkpoints, predictions and validation reports
data/                   Local datasets; excluded from version control
```

Start with [architecture and code navigation](docs/architecture.md) to trace a prediction from the data adapter through reasoning to exported metrics.
