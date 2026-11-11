# Graph reasoning and controlled variants

This document specifies evidence selection, confidence-aware graph reasoning
and the readouts used in the controlled experiments. Equation references follow
the accompanying HERI manuscript.

## Canonical equations

- Eq.12: separate learned affine saliency/reliability scorers on projected
  family features. Selected features receive a differentiable saliency gate;
  reliability is computed on the ungated projected features, once per forward.
- Eq.13: initialize with maximum saliency and greedily maximize the declared
  saliency/cosine-diversity objective at eta=0.5. Exact ties prefer source index.
- Spatial intra-family kNN excludes self, is symmetrized and deduplicated.
  Textual sequential edges join predecessor/successor in retained original
  token order. VS/VF correspondence uses one nearest counterpart; semantic
  VS/XA relations use eight neighbors and their exact reverse edges.
- Token spans use fast-tokenizer offsets in the common character frame
  reconstructed as `' '.join(words)`. Tokens are joined on span overlap;
  nonoverlapping source tokens use the nearest interval (first source-order
  destination breaks ties). Word-only fixtures have an explicit whole-word
  fallback. Padding and special tokens are excluded.
- Eq.14 confidence includes normalized cosine and fixed reliability.
  Structural confidence uses the geometric mean for two evidence endpoints;
  with one evidence endpoint, it uses that endpoint's reliability.
- Eq.15–16 use bias-free Q/K/V/O and relation transforms, per-relation incoming
  softmax, confidence log-prior and direct message gating, residual and LayerNorm.
  Canonical graph dropout is zero. There is no learned edge bias or relation
  priority. Sensitive similarity/confidence/softmax operations use FP32 with
  epsilon 1e-8. Native PyTorch segment softmax permits actual tiny CPU execution
  without changing the operator or requiring PyG.
- Eq.29 averages over valid text positions. Each text position contributes
  its maximum initial cosine similarity over valid image positions; the
  denominator is the number of valid text positions. SmoothL1 calibrates this
  statistic against the detached native-global image/text cosine. Padding is
  excluded, valid zero/negative maxima remain in the average, and samples with
  an empty modality are excluded. An empty batch domain returns differentiable
  zero.
- Eq.30 combines a pooled valid-node calibration BCE, mean-reliability penalty,
  and population-standard-deviation floor as a sum. Its selection pseudo-target
  is detached; learned saliency still receives gradients through feature gates.

## Independent refresh controls

`graph.reasoning.refresh_semantic` and `refresh_hypothesis` select which
relation families are updated after the first cycle. Structural edges and
reliability are retained. `hypothesis_state=initial` uses H0 solely for refreshed
support/feedback strengths; current node states still propagate through the
second shared operator. The metadata names are `semantic_only`,
`hypothesis_only`, `initial_hypothesis`, `state_dependent`, `static`, and `none`.

## Control architectures

- Removing hypothesis graph nodes leaves zero hypothesis nodes and incident
  edges. Each updated evidence family is mean-pooled; the four family means
  are concatenated and passed through a learned class-specific linear readout
  to produce the states required by the common prediction/grounding heads.
  This readout has no hypothesis nodes or hypothesis-conditioned message passing.
  The contrast therefore combines removal of hypothesis-node reasoning with
  a different readout. Its performance difference cannot be attributed solely
  to the presence of graph nodes.
- The progressive `operator=pooled` stages use the same family-mean readout
  while omitting all graph edges and HGT computation.
- Homogeneous evidence retains the exact canonical confidence-aware attention
  implementation while collapsing evidence types/relations. Homogeneous GAT
  uses shared graph attention without type/relation embeddings; its optional
  PyG dependency is explicit.
- The edge-residual diagnostic adds a learned two-attribute edge residual to
  standard attention. Similarity-conditioned attention retains normalized
  cosine confidence but removes reliability from attention confidence; the
  already routed node representations remain unchanged. Standard attention
  omits both confidence factors. These isolate the attention pathway.
- Removing forensic graph evidence retains the dense DINO store and the same
  image-grounding task, while exposing zero VF nodes to the graph/router.

## Progressive construction

The progressive study compares the following executable designs. The pooled
stages concatenate the four masked family means and use a learned linear map
to construct class-specific states for the shared prediction and grounding
heads. The graph stages read out the refined hypothesis nodes.

| Variant | Evidence and readout | Binary decision | Sufficiency objective | Reasoning |
| --- | --- | --- | --- | --- |
| `b0` | Global-feature fusion baseline | Independent head | None | None |
| `b1` | Cross-attention baseline | Independent head | None | Cross-attention |
| `evidence_decomposition` | Four evidence families; pooled readout; reliability router disabled | Independent head | Disabled | No graph operator |
| `reliability_router` | Same pooled readout; continuous reliability gates and router regularization | Independent head | Disabled | No graph operator |
| `hypothesis_evidence_graph` | Typed evidence/hypothesis graph; hypothesis-node readout | Independent head | Disabled | One HGT cycle |
| `constrained_single_pass` | Same typed graph and readout | Noisy-OR | Enabled | One HGT cycle |
| `full_hera` | Typed graph and hypothesis-node readout | Noisy-OR | Enabled | Two shared cycles with relation refresh |

These transitions are cumulative design comparisons. Introducing the graph
also changes the readout, and `constrained_single_pass` introduces Noisy-OR and
the sufficiency objective together. Use the separate `independent_binary` and
`no_evidence_sufficiency` controls to examine those mechanisms individually
under the full architecture. Likewise, report the no-hypothesis-node control
as a graph-and-readout contrast.

Use the same data, pretrained encoders, optimization horizon, threshold policy
and matched training seeds for a comparison. Inspect the resolved configurations
and mechanism definitions alongside the results; a variant name alone does
not identify an experimental run. [Experiment provenance](../experiments.md#experimental-provenance)
describes the artifacts needed to associate a result with its executed design.

Component interventions change only their declared settings and preserve the
base model's other loss weights, reasoning depth and refresh policy. Conflicting
study labels or stacked conditions are rejected instead of silently replacing
the base design.
