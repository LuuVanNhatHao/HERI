# Generated artifacts

- `data/`: dataset preflight audits.
- `experiments/`: HERI training, checkpoints, evaluations and controlled studies.
- `validation/latest/`: the latest bounded CPU validation report and summary. Temporary files are removed after the check.

Generated content is ignored by Git. Each new run records the effective config, source identity, data hashes and evaluation protocol. See [the experiment guide](../docs/experiments.md).
