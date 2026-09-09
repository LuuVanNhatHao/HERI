# Configuration layout

Runtime behavior is configured through YAML. The hierarchy makes the dataset
protocol, model role, and experiment scope visible from each path.

```text
config/
|-- datasets/
|   |-- dgm4/
|   |   |-- prepare.yaml
|   |   |-- prepare_server_validation.yaml
|   |   |-- hera/{train,evaluate,smoke}.yaml
|   |   `-- baselines/{b0,b1}/{train,evaluate,smoke}.yaml
|   `-- samm/
|       |-- prepare.yaml
|       |-- hera/{train,evaluate,smoke}.yaml
|       |-- hera/evaluate_seed42.yaml
|       |-- hera/text_grounding_diagnostic.yaml
|       `-- baselines/b1/{train,evaluate,smoke}.yaml
|-- experiments/
|   |-- dgm4/
|   `-- samm/
|-- diagnostics/
|   |-- inspect/
|   |-- smoke/
|   `-- server_validation.yaml
`-- benchmarks/
```

Files under `datasets/` are the canonical end-to-end entry points. SAMM
configs inherit the corresponding DGM4 model/runtime contract and override the
dataset paths, hashes, ontology, text length, output roots, and selection
metric. This keeps the architecture shared while making dataset-specific
semantics explicit.

Experiment manifests under `experiments/` materialize isolated effective YAML
files before launching generic scripts. Generated effective configs are stored
with their run artifacts and must not be edited in place.

The `benchmarks/` directory contains auxiliary binary-benchmark adapters. It is
separate from the two attribution-and-grounding protocols under `datasets/`.
