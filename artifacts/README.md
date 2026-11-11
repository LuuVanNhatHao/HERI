# Generated artifacts

- `data/`: dataset preflight audits.
- `experiments/`: HERI training, checkpoints, evaluations and controlled studies.
- `validation/latest/`: the latest bounded CPU validation report and summary. Temporary files are removed after the check.

Generated content is ignored by Git. Each new run records the effective config, source identity, data hashes and evaluation protocol. See [the experiment guide](../docs/experiments.md).

Keep study manifests, design contrasts and artifact-consistency audit reports
with the corresponding runs. An audit status of `internally_consistent` refers
to the available configuration/checkpoint/evaluation evidence; it does not
assert the identity of an unavailable original experiment.

Retain checkpoint `.metadata.json` sidecars and evaluation `run_evidence.json`
seals with their source files. Scientific aggregation writes `study_results.json`
after validating the manifest's artifact-consistency policy.
