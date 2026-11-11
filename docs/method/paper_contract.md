# Implementation reference

Equation and page references below refer to the accompanying HERI manuscript.
This reference connects the mathematical definitions to their implementation,
including numerical reductions, validity masks and experimental settings.
The Python package is `heri`; `scripts/heri.py` provides the command-line interface.

## Equation-to-code map

| Paper reference | Executable definition | Implementation |
| --- | --- | --- |
| Eq.1-2, pp.3-5 | Dataset-native hypothesis bank; four evidence families | [src/heri/graph/schema.py](../../src/heri/graph/schema.py), [src/heri/evidence/extractor.py](../../src/heri/evidence/extractor.py) |
| Eq.3-7, pp.5-6 | Two cycles, shared operator; refreshed semantic neighborhoods and hypothesis support | [src/heri/graph/reasoner.py](../../src/heri/graph/reasoner.py), [src/heri/graph/builder.py](../../src/heri/graph/builder.py) |
| Eq.8-9, p.6 | Dense hypothesis-conditioned grounding and atomic probabilities | [src/heri/models/heads/hypothesis_grounding.py](../../src/heri/models/heads/hypothesis_grounding.py), [src/heri/models/heads/attribution.py](../../src/heri/models/heads/attribution.py) |
| Eq.10, p.6 | `p_fake = 1 - product(1-p_k)` | [src/heri/models/heads/noisy_or.py](../../src/heri/models/heads/noisy_or.py) |
| Eq.11, p.6 | Seven weighted objective terms | [src/heri/losses/heri_components.py](../../src/heri/losses/heri_components.py) |
| Eq.12, p.17 | Independent sigmoid-linear saliency and reliability scorers | [src/heri/evidence/budgeting.py](../../src/heri/evidence/budgeting.py), [src/heri/routing/reliability.py](../../src/heri/routing/reliability.py) |
| Eq.13, p.18 | Highest-saliency start; greedy `eta*s + (1-eta)*(1-max_cosine)`; eta=0.5 | [src/heri/evidence/budgeting.py](../../src/heri/evidence/budgeting.py) |
| Eq.14, p.18 | Semantic confidence `((1+cos)/2)*sqrt(r_i*r_j)`; support/feedback confidence `r_i*(1+cos(e,h))/2` | [src/heri/routing/router.py](../../src/heri/routing/router.py), [src/heri/graph/builder.py](../../src/heri/graph/builder.py) |
| Eq.15-16, p.18 | Type Q/K/V/O, relation projections, log-confidence attention prior and direct confidence message gate | [src/heri/graph/hgt.py](../../src/heri/graph/hgt.py) |
| Eq.17-20, p.18 | Modality-shared bilinear dense scores; valid-position softmax; shared `[h;pooled]` box regressor; sigmoid cxcywh | [src/heri/models/heads/hypothesis_grounding.py](../../src/heri/models/heads/hypothesis_grounding.py) |
| Eq.21, p.18 | Relative probability-weighted mixture over applicable hypotheses | [src/heri/evaluation/hypothesis_grounding.py](../../src/heri/evaluation/hypothesis_grounding.py) |
| Eq.22, p.19 | Image aggregation uses individual `p_k>=0.5`; empty if none; clip valid image and undo padding/resize | [src/heri/evaluation/hypothesis_grounding.py](../../src/heri/evaluation/hypothesis_grounding.py), [src/heri/evaluation/evaluator.py](../../src/heri/evaluation/evaluator.py) |
| Eq.23-26, p.19 | Attribution/binary BCE; text BCE over valid entries; image SmoothL1+GIoU+0.25 patch BCE | [src/heri/losses/heri_components.py](../../src/heri/losses/heri_components.py), [src/heri/losses/box_grounding.py](../../src/heri/losses/box_grounding.py) |
| Eq.27-28, p.19 | Per-positive-hypothesis, per-applicable-modality sufficiency with annotation/position masks | [src/heri/losses/heri_components.py](../../src/heri/losses/heri_components.py) |
| Eq.29, p.19 | Mean over valid text positions of their maximum image similarity; stopped native-global teacher | [src/heri/evidence/cross_modal.py](../../src/heri/evidence/cross_modal.py) |
| Eq.30, pp.19-20 | Detached selection-derived reliability target in [0.5,1], mean=0.75 and std floor=0.05 | [src/heri/routing/router.py](../../src/heri/routing/router.py) |
| Eq.31-32, p.20 | Predicted-positive/applicable/annotation-available domain; EDC is mean clipped support ratio, UDR is support below 0.5 | [src/heri/metrics/consistency.py](../../src/heri/metrics/consistency.py) |
| Eq.33, p.22 | VSR@1 additionally requires true-positive hypothesis and correctly localized confident top-1 support | [src/heri/metrics/consistency.py](../../src/heri/metrics/consistency.py), [src/heri/evaluation/evaluator.py](../../src/heri/evaluation/evaluator.py) |

## Exact choices and edge cases

- Hidden size 512; node budgets VS/VF/XA/TC=96/96/64/64. Saliency gates selected
  projected features continuously. Reliability is measured once and retained.
- Semantic TopK=8. Visual intra-family spatial KNN=4, immediate sequential text
  neighbors, VS-VF nearest counterparts, XA-TC character-span overlap with
  nearest-span fallback. Padding and special tokens never form evidence edges.
- DGM4 FS/FA connect to VS/VF; TS/TA connect to XA/TC. Both SAMM hypotheses
  connect to all four families. Support and reverse feedback share confidence.
- Structural edges with one evidence endpoint use that endpoint's reliability;
  two-evidence structural edges use geometric-mean reliability. The paper
  requires reliability conditioning but does not print this structural formula.
- Relation attention is normalized over incoming neighbors within each relation
  and head, then relation contributions are summed. Sensitive operations use
  FP32 and epsilon 1e-8. All-masked softmax rows return zero without NaNs.
- Box parameters are sigmoid center/width/height. Internally, xyxy may extend
  beyond the canvas before inference clipping. SmoothL1 uses normalized cxcywh;
  GIoU and patch overlap use xyxy. A patch is annotated positive when its
  intersection area with the target box is strictly positive.
- Eq.26 first averages patch BCE within each valid image/hypothesis pair,
  then averages pairs equally. Unequal valid patch counts from letterboxing
  do not change the relative weight of those pairs.
- Text canonical aggregation is Eq.21 relative weighting. `presence_gated` and
  `token_noisy_or` are separately configured diagnostic controls.
- EDC is `mean(min(1,s/(p+1e-8)))`; UDR is `mean(s<0.5)` on Eq.31's domain.
  They do not use true manipulation labels. Empty domains return null.
- VSR@1 uses true-positive hypotheses, valid annotations and applicable modality;
  the top valid patch/token must score at least 0.5 and match its annotation.
  Empty domains return null. B0/B1 have no hypothesis-specific EDC/UDR/VSR.
- Image prediction selection and suppression never consume target labels.
  Annotation availability affects supervised losses and evaluation populations.

## Implementation conventions

1. **Eq.29 normalization.** For each valid text position, take the maximum
   initial similarity over valid image positions, then average these maxima
   over the valid text positions. The denominator is `|V_T|`. Padding contributes
   neither to the maximum nor to the text-position count; a valid zero or
   negative maximum still contributes to the average. SmoothL1 compares this
   local statistic with the detached cosine of the native image/text global
   features. Losses are averaged across samples with both valid modalities;
   samples without a valid image/text pair are excluded, and an entirely empty
   domain returns differentiable zero. The objective is identified by
   `loss.alignment.aggregation: valid_text_mean`; resume and evaluation compare
   this field so a checkpoint trained with a different reduction is not silently
   reused under the current objective.
2. **Text evaluation unit.** Text grounding is evaluated at both subword and
   original-word resolution. Artifacts retain both native-token and original-word
   F1, their definitions, visible-word coverage and truncation false negatives.
   These fields must not be exchanged under a common unlabeled table heading.
   Original-word metadata completeness is exported separately from annotation
   availability, so masked localization annotations do not invalidate an
   otherwise complete classification run.
3. **Control readouts.** Progressive stages define cumulative architectures,
   while component controls test specific contrasts. The
   [experimental designs](graph_reasoning.md#progressive-construction) state
   their operators, readouts and combined changes. Removing hypothesis nodes
   also replaces the hypothesis readout; that contrast measures both changes.
4. **Grounding conventions.** HERI uses its configured preprocessing and
   tokenization. Released SAMM classification projection is explicit. All-sample
   image denominators and inverse geometry are implemented, but equal metric
   names alone do not establish identical preprocessing to external systems.

## Training and experimental identity

- Canonical seed 7; matched seeds 7/21/101 for both datasets. SAMM subset seed
  777 is independent. `python -B scripts/heri.py reconstruct-samm` implements the printed
  ID sampling/order/UTF-8 recipe and verifies the three frozen output hashes
  before writing. Full training: 20 epochs, batch 16, AdamW weight decay
  0.01, modules LR 1e-4, each encoder LR 2e-6, 500 warm-up steps and linear decay.
- Encoders freeze for epochs 1-3, fully unfreeze from epoch 4. Adam state and
  one global learning-rate clock continue across the stage boundary.
- The manuscript's reference hardware is one RTX PRO 5000 with 47.8 GB VRAM.
  CPU verification uses reduced synthetic features.
- Binary, attribution and image-hypothesis decisions use fixed 0.5. Validation
  CF1 selects checkpoints; early stopping is disabled for paper experiments.
- Noisy-OR binary loss detaches absent hypotheses on fake samples only.
- Resume validates both model and sampler seeds, architecture and protocol;
  checkpoint state includes loader/sampler RNG. Checkpoints write atomically.
- Every dataset load measures annotation SHA-256. Canonical loading rejects
  mismatches; artifacts distinguish observed hashes from configured identities.
- Each reported experiment is linked to its checkpoint, effective configuration,
  observed data hashes and predictions. The
  [verification guide](../validation/local_checks.md) describes the synthetic
  tests used to check the implementation before full experiments.
