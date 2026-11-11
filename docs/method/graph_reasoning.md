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
- Eq.29 is implemented literally: maxima are summed over valid text nodes
  and divided by the number of valid image nodes. The printed denominator
  is retained when the visual and textual position counts differ.
  Empty sets return differentiable zero.
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
