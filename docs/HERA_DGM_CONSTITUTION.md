# HERA-DGM — PHASE 0 FINAL

> **Final implementation amendment (`hera_final_v2`).** This amendment
> supersedes every legacy MTCNN-candidate, one-cycle, frozen-edge, and
> target-selected grounding statement below. The model never consumes
> `mtcnn_boxes`; image grounding is dense DINOv3 box regression; inference
> grounding is routed only by predicted atomic probabilities; edge confidence
> is part of HGT attention/messages; and canonical reasoning uses exactly two
> cycles with a target-independent graph rebuild.
> **Scientific-v3 amendment.** For corrected-protocol experiments, this
> amendment supersedes legacy metric and experiment rules below. CF1 means the
> harmonic mean of class-mean precision and recall; macro-F1 is a separate field.
> Fixed-0.5 validation CF1 selects checkpoints in the scientific suite. Grounding
> reports all samples, explicit positive-only diagnostics, and original-word
> truncation coverage. Training labels may enter atomic loss, the target-consistent
> noisy-OR gradient rule, and applicable grounding losses, but never inference
> evidence selection/graph routing. Router scores are computed once; dense raw
> stores remain available to grounding. Local-support consistency is not causal
> faithfulness. Registered learned-query and B1+noisy-OR controls are authorized
> experimental alternatives, not canonical architecture changes. The scientific
> suite matches 20 epochs and seeds 7/21/101 with early stopping disabled; R1 scores
> are archival until corrected-protocol evaluation.

## Scientific & Engineering Constitution

**Project:** HERA-DGM  
**Full name:** Hypothesis–Evidence Routing for Attribution-First Multimodal Manipulation Detection  
**Constitution version:** Phase 0 Final v1.0  
**Primary benchmark:** DGM⁴  
**Status:** FROZEN before Phase 1 implementation

---

# 1. Purpose of This Constitution

This document is the authoritative scientific and engineering specification for HERA-DGM.

It defines:

- scientific thesis;
- novelty boundaries;
- dataset contract;
- supervision contract;
- architecture;
- model configuration;
- graph mechanism;
- training strategy;
- threshold tuning;
- validation and testing behavior;
- evaluation protocol;
- repository organization;
- runtime configuration;
- logging and progress requirements;
- reproducibility requirements;
- experimental baselines;
- ablation requirements;
- scientific kill criteria.

Any implementation that violates a requirement marked **MUST**, **MUST NOT**, **MANDATORY**, or **FROZEN** is not the canonical HERA-DGM implementation.

If an implementation decision conflicts with this Constitution, the Constitution takes precedence.

---

# 2. Scientific Problem

DGM⁴ jointly studies:

- multimodal manipulation detection;
- manipulation-type attribution;
- manipulated-image grounding;
- manipulated-text grounding.

HERA-DGM does **not** claim that previous methods completely ignore interactions among these tasks.

It also does **not** claim that:

- evidence graphs are new;
- graph reasoning for misinformation is new;
- detection-grounding interaction is new;
- uncertainty modeling is new.

The scientific question is deliberately narrower:

> **Can multimodal manipulation detection become more faithful and more generalizable when authenticity is forced to arise from localized manipulation-specific evidence, instead of being predicted directly from a fused multimodal representation?**

The intended dependency is:

\[
\boxed{
Localized\ Evidence
\rightarrow
Manipulation\ Attribution
\rightarrow
Authenticity
}
\]

---

# 3. Core HERA-DGM Thesis

HERA-DGM is an **evidence-centric, attribution-first** framework.

The four atomic manipulation hypotheses are:

\[
H=\{FS,FA,TS,TA\}
\]

where:

- `FS` = Face Swap;
- `FA` = Face Attribute;
- `TS` = Text Swap;
- `TA` = Text Attribute.

The canonical inference process is:

\[
E
\rightarrow
\{FS,FA,TS,TA\}
\rightarrow
Fake/Real.
\]

Authenticity is therefore a consequence of evidence-supported manipulation attribution.

---

# 4. Critical No-Shortcut Rule

The final canonical HERA-DGM model:

**MUST NOT contain an independent binary classifier that directly maps a global/fused representation to Real/Fake.**

Forbidden canonical path:

```text
multimodal representation
        ↓
binary MLP
        ↓
Fake / Real
```

Required path:

```text
localized evidence
        ↓
FS / FA / TS / TA
        ↓
derived authenticity
```

A direct binary head may exist **only as an ablation or controlled baseline**.

---

# 5. Novelty Position

The novelty of HERA-DGM is a system-level scientific formulation, not one isolated neural block.

## 5.1 Attribution-First Decision

Existing DGM⁴-style systems commonly learn multiple correlated outputs.

HERA-DGM instead imposes:

\[
Evidence
\rightarrow
Attribution
\rightarrow
Verdict.
\]

The final Fake probability is derived from manipulation hypotheses.

## 5.2 Evidence-Supported Hypotheses

Each manipulation prediction must be associated with localized supporting evidence.

Examples:

\[
E^{visual\_forensic}\rightarrow FS
\]

\[
E^{visual\_forensic}\rightarrow FA
\]

\[
E^{text}\rightarrow TS
\]

\[
E^{text}\rightarrow TA.
\]

Grounding is therefore part of the reasoning process rather than only another parallel output.

## 5.3 Heterogeneous Evidence Separation

HERA distinguishes:

\[
E^{semantic}
\]

from:

\[
E^{forensic}.
\]

### Semantic evidence

Captures:

- image-text alignment;
- semantic contradiction;
- contextual inconsistency;
- entity/event mismatch.

### Modality-native forensic evidence

Captures:

- local visual manipulation traces;
- abnormal facial-region evidence;
- contextual text anomalies;
- sparse manipulated-word evidence.

Core principle:

> **Do not collapse semantic and forensic evidence into one representation before determining which manipulation hypothesis they support.**

## 5.4 Uncertainty as Routing

Uncertainty is used primarily to determine whether evidence should participate in reasoning.

Preferred interpretation:

\[
uncertainty
\rightarrow
evidence\ reliability
\rightarrow
routing.
\]

Not merely:

\[
uncertainty
\rightarrow
final\ scalar\ fusion\ weight.
\]

---

# 6. Claims That MUST NOT Be Used

The paper and repository documentation must not claim:

> We are the first to connect grounding and detection.

> Evidence graph reasoning is new.

> Graph-based misinformation verification has not been studied.

> Previous DGM⁴ methods treat every sub-task independently.

> HGT itself is the main novelty.

> SigLIP2, DINOv3, or DeBERTa usage is the novelty.

> Uncertainty estimation itself is new.

> No previous work models agreement between detection and grounding.

The contribution must be described as the **specific evidence → attribution → authenticity formulation and its empirical consequences**.

---

# 7. Preferred Contribution Statement

Canonical formulation:

> Existing multimodal manipulation detectors learn authenticity, manipulation attribution, and grounding as strongly correlated predictive objectives. HERA-DGM introduces an attribution-first alternative in which localized heterogeneous evidence is explicitly routed toward manipulation hypotheses, while authenticity is derived from the resulting evidence-supported attribution.

Secondary contribution:

> HERA-DGM preserves cross-modal semantic evidence and modality-native forensic evidence as distinct information sources until hypothesis-level reasoning.

Experimental contribution:

> HERA-DGM evaluates whether evidence-supported attribution improves IID performance, cross-generator generalization, and prediction faithfulness under evidence intervention.

---

# 8. Dataset Root

The canonical dataset root is permanently defined as:

```yaml
dataset:
  root: data/DGM4
```

Source code MUST NOT contain user-specific absolute dataset paths.

Forbidden examples:

```text
/home/user/...
/workspace/...
/mnt/...
C:\...
```

All paths must be derived from:

```text
data/DGM4
```

using `pathlib.Path`.

---

# 9. Frozen Dataset Manifest

Canonical local DGM⁴ release:

| Split | Rows |
|---|---:|
| Train | 208,184 |
| Validation | 22,126 |
| Test | 50,705 |
| Total | **281,015** |

Additional frozen statistics:

```text
Unique physical images: 206,362
Real/orig rows:          128,441
Manipulated rows:        152,574
Unique pristine IDs:      77,426
```

The official train/validation/test splits MUST remain unchanged for the primary benchmark.

---

# 10. Dataset SHA-256 Contract

Canonical annotation hashes:

```text
data/DGM4/train.json
dafd4fe101d3de63b433759fae79b14b465294f7be73f26c9f3bb13b8de87704
```

```text
data/DGM4/val.json
d4377ecc19cb6528e36c77401482cbfffaa57977faee2c6124e3b1bdda045520
```

```text
data/DGM4/test.json
b58b29c4cf0f40cdf791189181ad57821f9359e358e93683cc702db3a41c0a18
```

Canonical experiments SHOULD verify these hashes during dataset preflight.

---

# 11. Dataset Integrity

Audit results are frozen:

```text
Unique images checked:        206,362
Full PIL decodes successful:  206,362
Decode failures:              0
Warnings:                     0
Image format:                 JPEG
Image mode:                   RGB

Invalid fake_image_box:       0
Invalid mtcnn_boxes:          0
Invalid fake_text_pos rows:   0
Missing annotation images:    0
```

No silent data repair is permitted.

Unexpected corruption or schema violations must produce a visible failure or explicitly logged exclusion.

---

# 12. Local Path Mapping

Annotation manipulation path:

```text
DGM4/manipulation/{method}/{file}.jpg
```

maps to:

```text
data/DGM4/images/manipulation/{method}/{method}/{file}.jpg
```

Annotation original path:

```text
DGM4/origin/{source}/{folder}/{file}.jpg
```

maps to:

```text
data/DGM4/images/origin/{source}/{source}/{folder}/{file}.jpg
```

The loader must implement this mapping through a centralized path resolver.

---

# 13. Atomic Label Contract

Canonical label order:

```text
[FS, FA, TS, TA]
```

Mappings:

```text
orig
→ [0,0,0,0]

face_swap
→ [1,0,0,0]

face_attribute
→ [0,1,0,0]

text_swap
→ [0,0,1,0]

text_attribute
→ [0,0,0,1]

face_swap&text_swap
→ [1,0,1,0]

face_attribute&text_swap
→ [0,1,1,0]

face_swap&text_attribute
→ [1,0,0,1]

face_attribute&text_attribute
→ [0,1,0,1]
```

This mapping must be centralized in one source file.

Individual model modules MUST NOT independently parse `fake_cls`.

---

# 14. Label Distribution

Atomic operation counts:

```text
FS: 66,722
FA: 56,411
TS: 43,546
TA: 18,588
```

Mixed manipulation:

```text
32,693 samples
```

Text-only manipulations:

```text
29,441
```

All 29,441 text-only rows use origin/pristine images.

---

# 15. Text Grounding Contract

`fake_text_pos` semantics are frozen as:

- 0-based;
- word-level;
- indexed over `text.split()`;
- text already has ASCII punctuation cleaned.

Audit:

```text
Rows with fake_text_pos:        62,134
Marked word positions:         553,514
Min position:                        0
Max position:                       29
Valid 0-based rows:            62,134 / 62,134
```

Canonical tokenizer workflow:

```python
words = text.split()

encoding = tokenizer(
    words,
    is_split_into_words=True,
    ...
)
```

Grounding supervision must be mapped through:

```python
encoding.word_ids()
```

A subword is positive if:

```python
word_id in fake_text_pos
```

Direct comparison between DeBERTa subword indices and `fake_text_pos` is forbidden.

---

# 16. Text Manipulation Structural Prior

Text Swap and Text Attribute have materially different evidence structures.

Approximate behavior:

```text
Text Swap:
~12 manipulated words on average

Text Attribute:
~1.5 manipulated words on average
```

Therefore:

```text
TS → distributed / span-level evidence

TA → sparse / local evidence
```

Hypothesis-specific aggregation may exploit this distinction.

This is an inductive bias, not a headline novelty claim.

---

# 17. Image Grounding Contract

Rows with image manipulation:

```text
123,133
```

Every one contains `fake_image_box`.

All:

- are in image bounds;
- are non-degenerate.

For all 123,133 manipulated-image rows:

\[
\max IoU(fake\_image\_box,mtcnn\_boxes)=1.0.
\]

This correspondence is a descriptive release audit only and grants no
permission to expose MTCNN candidates to a model.

---

# 18. Anti-Leakage Rule for Image Grounding

`fake_image_box`:

**MAY** be used as:

- training supervision;
- validation ground truth;
- test ground truth.

`fake_image_box`:

**MUST NOT** be supplied as an input feature.

It must not be used to reveal which MTCNN face candidate is manipulated at inference time.

`mtcnn_boxes` MUST NOT be used as model input, candidate generation, routing,
grounding selection, or evaluation-time side information. It is unused release
metadata that may only be inspected during a data audit.

---

# 19. Dense Image Evidence

The complete target-independent DINOv3 patch grid is the image-localization
source. A graph-budgeted subset supports reasoning, while the unbudgeted patch
store remains available to the grounding head. FS/FA-conditioned dense
heatmaps and continuous normalized boxes are predicted directly. The legacy
`face` graph type and all of its relations are absent from the final schema, so
no MTCNN-derived parameter, node, or proposal can enter training or inference.

---

# 20. Generator Contract

Face Swap generators:

```text
InfoSwap
SimSwap
```

Face Attribute generators:

```text
HFGI
StyleCLIP
```

Audit confirms 100% generator ↔ atomic-class consistency.

This mapping may be used for controlled cross-generator evaluation.

---

# 21. Image Preprocessing

Canonical image target:

```text
384 × 384
```

Images MUST preserve aspect ratio.

Required flow:

```text
RGB decode
    ↓
aspect-ratio preserving resize
    ↓
padding
    ↓
384 × 384 canvas
```

The same deterministic geometric transformation must be applied to:

- `fake_image_box` when used as supervision.

Naive geometric stretching to 384×384 is forbidden.

Aggressive random crops that can remove manipulated evidence are forbidden in canonical grounding training.

---

# 22. Main Semantic Multimodal Encoder

Canonical semantic backbone:

```text
google/siglip2-so400m-patch14-384
```

Responsibilities:

- cross-modal semantics;
- image-text alignment;
- semantic inconsistency;
- local semantic image representations;
- text representations aligned with image semantics.

It is not the final manipulation detector by itself.

---

# 23. Dense Visual Encoder

Canonical dense visual backbone:

```text
facebook/dinov3-vitl16-pretrain-lvd1689m
```

Responsibilities:

- dense visual features;
- local visual evidence;
- local FS/FA manipulation representations;
- patch-context representations;
- modality-native visual forensic clues.

The DINOv3 branch is not an independent Fake/Real classifier.

---

# 24. Contextual Text Encoder

Canonical contextual language model:

```text
microsoft/deberta-v3-large
```

Responsibilities:

- contextual word/token representations;
- Text Swap evidence;
- sparse Text Attribute evidence;
- language-native contextual anomaly detection.

---

# 25. Transformers Compatibility Rule

All model wrappers using Hugging Face Transformers MUST be designed for reasonable compatibility across newer and older supported `transformers` APIs.

Repository MUST include:

```text
src/compat/transformers_compat.py
```

Compatibility code must be centralized.

Preferred strategy:

```text
detect supported API/capability
        ↓
use modern implementation
        ↓
safe fallback where possible
```

Compatibility concerns include:

- model loading kwargs;
- processors;
- tokenizers;
- dtype arguments;
- attention implementation;
- model output fields;
- optional parameters;
- fast-tokenizer behavior.

MUST NOT:

- edit `site-packages`;
- monkey-patch Transformers globally;
- silently discard unsupported parameters;
- scatter version checks throughout the repository.

Unsupported versions must fail early with an explicit diagnostic.

---

# 26. Heterogeneous Evidence Families

At minimum HERA models:

### Visual semantic evidence

From SigLIP2 image features.

### Visual forensic evidence

From DINOv3 dense features.

### Text alignment evidence

From SigLIP2 text representations.

### Text contextual evidence

From DeBERTa.

### Manipulation hypothesis nodes

```text
H_FS
H_FA
H_TS
H_TA
```

---

# 27. Evidence Selection

The graph should not indiscriminately contain all dense image patches.

Initial tunable ranges:

```yaml
model:
  evidence:
    visual_top_k: 32
    text_top_k: 32
```

Reasonable search region:

```text
visual Top-K: 32–64
text Top-K:   16–32
```

Exact values belong in YAML.

They are hyperparameters, not novelty claims.

---

# 28. Evidence Reliability Router

Each evidence candidate receives reliability:

\[
r_i\in[0,1].
\]

The router may use uncertainty, anomaly score, context, or learned reliability estimation.

Primary function:

```text
candidate evidence
        ↓
reliability
        ↓
routing / suppression
        ↓
graph
```

It should prevent uncertain evidence from indiscriminately contributing to every manipulation hypothesis.

---

# 29. Graph Backbone

Canonical graph mechanism:

```text
Heterogeneous Graph Transformer (HGT)
```

Initial configuration:

```yaml
model:
  graph:
    type: hgt
    layers: 2
    hidden_dim: 512
    heads: 8
    dropout: 0.1
```

A hidden dimension of 768 may be evaluated.

Three graph layers may be evaluated.

Depth beyond three requires empirical justification.

HGT is an implementation mechanism, not the novelty itself.

Every edge stores `[relation_strength, confidence]`. In `hera_final_v2`,
confidence contributes an additive log gate to the per-destination attention
logit and directly multiplies the transmitted value message. Adding edge
attributes only as a post-HGT residual is forbidden.

---

# 30. Required Graph Baseline

GATv2 must be supported as a controlled graph baseline.

This allows comparison between:

```text
HGT
vs
GATv2
```

under equivalent evidence and backbone settings.

---

# 31. Canonical Relations

Initial relation families may include:

```text
visual_semantic ↔ visual_forensic

visual_semantic ↔ text_alignment

text_alignment ↔ text_context

visual_forensic → FS

visual_forensic → FA

text_context → TS

text_context → TA

cross_modal_semantic → FS

cross_modal_semantic → FA

cross_modal_semantic → TS

cross_modal_semantic → TA

hypothesis → evidence
```

New relation types must have explicit scientific justification.

---

# 32. Hypothesis-Specific Reasoning

## FS

Primary signals:

- local visual forensic evidence;
- visual context;
- optional semantic evidence.

## FA

Primary signals:

- fine-grained local visual evidence;
- facial-context anomaly.

## TS

Primary signals:

- distributed contextual text evidence;
- semantic mismatch;
- span-level evidence.

## TA

Primary signals:

- sparse contextual evidence;
- high-confidence local token evidence;
- semantic evidence where useful.

---

# 33. Bidirectional Reasoning

HERA may use:

\[
Evidence\rightarrow Hypothesis
\]

followed by:

\[
Hypothesis\rightarrow Evidence.
\]

The first stage asks:

> Which evidence supports this hypothesis?

The second asks:

> Given this hypothesis, which evidence should be refined or emphasized?

Canonical refinement depth:

```text
exactly 2 cycles with graph rebuild between cycles
```

Deep iterative reasoning should not be introduced without evidence that it improves the scientific objective.

---

# 34. Hypothesis-Conditioned Grounding

Preferred predictions:

\[
P(patch_i, box|FS)
\]

\[
P(patch_i, box|FA)
\]

\[
P(token_j|TS)
\]

\[
P(token_j|TA)
\]

Grounding should therefore explain **which evidence supports which hypothesis**, not merely identify a generic suspicious region.

---

# 35. Attribution-Derived Authenticity

Let:

\[
p_{FS},p_{FA},p_{TS},p_{TA}
\]

be hypothesis probabilities.

Canonical authenticity aggregation:

\[
P(Fake)
=
1-
(1-p_{FS})
(1-p_{FA})
(1-p_{TS})
(1-p_{TA}).
\]

This is the canonical differentiable Noisy-OR decision.

Then:

\[
P(Real)=1-P(Fake).
\]

---

# 36. Training Objective

Initial total objective:

\[
\mathcal{L}
=
\lambda_{attr}\mathcal{L}_{attr}
+
\lambda_{bin}\mathcal{L}_{bin}
+
\lambda_{img}\mathcal{L}_{img}
+
\lambda_{txt}\mathcal{L}_{txt}
+
\lambda_{ev}\mathcal{L}_{ev}.
\]

All loss coefficients must live in YAML.

---

# 37. Attribution Loss

Manipulation attribution is multi-label.

\[
\mathcal{L}_{attr}
=
BCE(
[p_{FS},p_{FA},p_{TS},p_{TA}],
Y
).
\]

Class imbalance handling may include:

- positive-class weights;
- focal loss;
- asymmetric loss.

Any nonstandard loss that materially changes performance requires ablation.

---

# 38. Binary Loss

Binary ground truth supervises the derived Noisy-OR output:

\[
\mathcal{L}_{bin}
=
BCE(P(Fake),y).
\]

The binary label therefore reinforces consistency without introducing a direct binary shortcut.

---

# 39. Image Grounding Loss

Canonical implementation MUST ignore MTCNN correspondence.

A dense patch heatmap plus continuous normalized box regression is required.

The image loss combines Smooth-L1, generalized IoU, and dense patch overlap
supervision from `fake_image_box`.

The simplest scientifically valid implementation should be tested first.

---

# 40. Text Grounding Loss

Text grounding operates over tokenizer subwords aligned to word-level DGM⁴ supervision.

Possible loss:

- BCE;
- weighted BCE;
- focal loss.

The canonical evaluation is ultimately reconstructed at the required grounding level.

---

# 41. Evidence Sufficiency Constraint

An optional evidence consistency objective may enforce:

\[
P(h)\uparrow
\Rightarrow
Support(E_h)\uparrow.
\]

Its purpose is to discourage predictions such as:

```text
FA probability = 0.98
but
no meaningful local image evidence selected
```

The loss must remain interpretable and simple.

---

# 42. Available Hardware

Target server:

```text
GPU VRAM: 48 GB
Available disk: approximately 200 GB
CUDA environment: 13.0
```

The project intentionally excludes:

- MLLMs;
- LLM fine-tuning;
- RAG;
- external evidence retrieval;
- reinforcement learning.

Compute should instead support:

- stronger encoders;
- dense feature extraction;
- high-resolution input;
- graph reasoning;
- controlled ablations;
- multiple seeds.

---

# 43. Precision and Memory

Preferred precision:

```yaml
runtime:
  precision: bf16
```

Optional techniques:

- gradient checkpointing;
- gradient accumulation;
- efficient attention where supported;
- PyTorch SDPA;
- memory-efficient optimizer strategies.

These must not alter the scientific formulation.

---

# 44. Training Stages

## Stage A — Head / Graph Warm-Up

Initially freeze:

```text
SigLIP2
DINOv3
DeBERTa
```

Train:

- projection layers;
- evidence modules;
- router;
- graph;
- hypothesis nodes;
- grounding heads.

Canonical duration:

```text
3 epochs
```

## Stage B — Full Backbone Fine-Tuning

Starting at epoch 4, unfreeze every trainable parameter in:

```text
SigLIP2
DINOv3
DeBERTa
```

Canonical HERA does not stop at a last-block-only adaptation stage. At the
Stage A → Stage B boundary, rebuild the optimizer, preserve compatible Adam
state for the already-trained HERA modules, and fail fast unless each
backbone reports `trainable_parameters == total_parameters`.

Starting LR regions:

```text
backbone:
1e-6 → 5e-6

new modules:
~1e-4
```

Use BF16 autocast with FP32 parameter storage, gradient checkpointing, and
gradient accumulation to keep full fine-tuning memory-safe without silently
freezing backbone parameters. All exact values are YAML parameters.

---

# 45. Mandatory tqdm Rule

The `tqdm` package is mandatory.

Prefer:

```python
from tqdm.auto import tqdm
```

Every meaningful long-running process MUST expose visible progress.

This includes:

- annotation scanning;
- dataset preflight;
- hash checking where measurable;
- image validation;
- metadata preprocessing;
- cache generation;
- manifest generation;
- train batches;
- validation batches;
- test batches;
- threshold search;
- subgroup evaluation;
- cross-generator evaluation;
- intervention evaluation;
- result aggregation;
- long output-generation processes.

No long-running silent loop is permitted.

Progress bars should expose useful state where possible:

```text
epoch
batch
loss
learning rate
primary metric
```

Use `tqdm.write()` when emitting messages during progress execution.

---

# 46. Training Epoch Lifecycle

Every completed epoch MUST follow:

```text
TRAIN
  ↓
tqdm training progress
  ↓
VALIDATION
  ↓
tqdm validation progress
  ↓
threshold tuning on validation
  ↓
metric computation
  ↓
sklearn classification reports
  ↓
print thresholds
  ↓
checkpoint decision
  ↓
persist epoch metrics
```

Validation after every epoch is mandatory.

---

# 47. sklearn Classification Report Rule

Classification reports MUST come directly from:

```python
from sklearn.metrics import classification_report
```

Handcrafted substitutes are forbidden.

After every epoch, validation must print a binary report.

Canonical classes:

```text
real
fake
```

It must also print a multi-label attribution report for:

```text
face_swap
face_attribute
text_swap
text_attribute
```

Recommended:

```python
classification_report(
    ...,
    digits=4,
    zero_division=0,
)
```

Manual formatting may not replace the sklearn report.

---

# 48. Threshold Tuning

Threshold tuning is mandatory.

Thresholds are evaluation decision parameters, not trainable model parameters.

## Binary threshold

Find:

\[
t^{*}_{binary}
\]

from validation predictions.

Default optimization target SHOULD be Macro-F1 unless changed through YAML.

## Hypothesis thresholds

Allow independent thresholds:

\[
t_{FS},t_{FA},t_{TS},t_{TA}.
\]

Possible objectives:

- per-class F1;
- CF1-oriented joint criterion.

Exact strategy belongs in YAML.

---

# 49. Threshold Anti-Leakage

Thresholds:

**MAY** be tuned on validation.

Thresholds:

**MUST NOT** be tuned on test.

Workflow:

```text
train
  ↓
validation predictions
  ↓
threshold tuning
  ↓
freeze best-checkpoint thresholds
  ↓
test
```

For research transparency, experiments SHOULD record:

```text
metrics at threshold 0.5

and

metrics at tuned thresholds
```

---

# 50. Threshold Progress Visibility

If threshold optimization evaluates a search grid or iterative algorithm, its progress MUST also be exposed through tqdm.

No hidden threshold search loop.

---

# 51. Test Evaluation Contract

Final test evaluation must follow this exact high-level order:

```text
load best checkpoint
        ↓
load frozen validation thresholds
        ↓
run test inference with tqdm
        ↓
compute official DGM4 metrics
        ↓
print sklearn binary classification_report
        ↓
print sklearn multi-label classification_report
        ↓
aggregate all final metrics
        ↓
write summary.csv
```

Test labels may not influence threshold selection.

---

# 52. summary.csv Requirement

Every final evaluation run MUST produce:

```text
summary.csv
```

The file must be machine-readable by:

```python
pandas.read_csv(...)
```

It should consolidate all final scalar results and run metadata.

Required categories include:

### Run information

```text
run_id
model_name
seed
checkpoint
split
config_name
```

### Thresholds

```text
binary_threshold
fs_threshold
fa_threshold
ts_threshold
ta_threshold
```

### Binary metrics

```text
auc
acc
eer
```

### Attribution metrics

```text
map
cf1
of1
```

### Image grounding

```text
image_mean_iou
image_iou50
image_iou75
```

### Text grounding

```text
text_precision
text_recall
text_f1
```

### sklearn-derived class metrics

```text
real_precision
real_recall
real_f1

fake_precision
fake_recall
fake_f1

fs_precision
fs_recall
fs_f1

fa_precision
fa_recall
fa_f1

ts_precision
ts_recall
ts_f1

ta_precision
ta_recall
ta_f1
```

### Reproducibility information

```text
dataset_hashes
git_commit
precision
effective_batch_size
peak_vram
training_duration
```

The exact schema must be stable once Phase evaluation is frozen.

---

# 53. Complete requirements.txt Rule

Repository root MUST contain:

```text
requirements.txt
```

before real training begins.

The intended workflow is:

```bash
pip install -r requirements.txt
```

followed by project execution.

The project MUST NOT rely on incremental dependency discovery during training.

Forbidden workflow:

```text
train
→ missing package
→ pip install
→ train
→ missing another package
→ pip install again
```

Requirements must cover all runtime libraries.

Likely categories include:

```text
torch
transformers
tokenizers
safetensors

scikit-learn
numpy
scipy
pandas

PyYAML
tqdm
Pillow

graph library/dependencies
```

Development-only tools may additionally use:

```text
requirements-dev.txt
```

but this does not replace `requirements.txt`.

---

# 54. Repository Structure

The following folders are mandatory:

```text
config/
src/
scripts/
```

Canonical initial structure:

```text
HERA-DGM/
│
├── config/
│   ├── prepare_dgm4.yaml
│   ├── train_baseline.yaml
│   ├── train_hera.yaml
│   ├── evaluate_hera.yaml
│   ├── cross_generator.yaml
│   └── intervention.yaml
│
├── src/
│   ├── compat/
│   ├── data/
│   ├── models/
│   ├── evidence/
│   ├── graph/
│   ├── losses/
│   ├── metrics/
│   ├── thresholding/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── cross_generator.py
│   └── intervention.py
│
├── tests/
├── docs/
├── outputs/
├── logs/
├── requirements.txt
└── README.md
```

Additional folders may be introduced when justified.

---

# 55. Folder Responsibilities

## `config/`

Contains runtime YAML configuration only.

## `src/`

Contains actual implementation:

- datasets;
- model components;
- losses;
- metrics;
- graph mechanisms;
- trainers;
- threshold algorithms;
- evaluation logic.

## `scripts/`

Contains only executable entry points.

Example:

```text
scripts/train.py
```

should orchestrate:

```text
load YAML
→ validate config
→ initialize project components
→ execute trainer
```

It MUST NOT contain the complete model/training implementation.

---

# 56. Canonical Command Format

All directly runnable project commands MUST follow:

```bash
python <script.py> <config.yaml>
```

Examples:

```bash
python scripts/prepare_data.py config/prepare_dgm4.yaml
```

```bash
python scripts/train.py config/train_baseline.yaml
```

```bash
python scripts/train.py config/train_hera.yaml
```

```bash
python scripts/evaluate.py config/evaluate_hera.yaml
```

```bash
python scripts/cross_generator.py config/cross_generator.yaml
```

```bash
python scripts/intervention.py config/intervention.yaml
```

This command style is mandatory from preprocessing through final evaluation.

---

# 57. No Ad-Hoc CLI Hyperparameters

Forbidden canonical command:

```bash
python scripts/train.py \
    --lr 1e-4 \
    --epochs 20 \
    --batch-size 16 \
    --threshold 0.51
```

All such values MUST exist in YAML.

Each executable should normally receive only:

```text
config_path
```

as a positional runtime argument.

---

# 58. YAML Is the Runtime Source of Truth

All runtime information must come from YAML or be deterministically derived from it.

This includes:

- dataset root;
- annotation paths;
- image root;
- cache paths;
- output paths;
- pretrained model names;
- input resolution;
- evidence Top-K;
- graph type;
- graph layers;
- graph hidden dimension;
- graph heads;
- loss coefficients;
- epochs;
- batch size;
- effective batch size;
- optimizer;
- scheduler;
- learning rates;
- weight decay;
- precision;
- workers;
- random seed;
- thresholds;
- threshold search;
- checkpoint metric;
- early stopping;
- tqdm behavior;
- logging settings.

Hidden experimental hyperparameters inside source code are forbidden.

---

# 59. Config Snapshot

Every run must preserve the exact YAML used.

Example:

```text
outputs/
└── hera_seed42_YYYYMMDD_HHMMSS/
    ├── config.yaml
    ├── checkpoints/
    ├── metrics/
    ├── logs/
    └── summary.csv
```

The copied `config.yaml` must be the effective configuration used by the run.

---

# 60. Failure Visibility

Silent exception swallowing is forbidden.

Forbidden:

```python
try:
    ...
except:
    pass
```

Important failures must report:

- processing stage;
- exception type;
- file/sample where relevant;
- whether execution stops or skips.

Unexpected dataset or model inconsistency should fail fast by default.

---

# 61. Controlled Strong-Backbone Baselines

To distinguish architecture gain from backbone gain, HERA must be compared under controlled backbones.

## B0 — Strong Global Baseline

```text
SigLIP2
+
DINOv3
+
DeBERTa
        ↓
simple pooling / concatenation
        ↓
conventional prediction heads
```

## B1 — Cross-Attention Baseline

Same encoders.

```text
features
→ conventional cross-modal attention/fusion
→ multitask outputs
```

## B2 — GATv2 Baseline

Same encoders/evidence where practical.

```text
graph
→ GATv2
→ conventional multitask outputs
```

## B3 — HGT Without Attribution-First Constraint

Tests whether HGT alone accounts for gains.

## B4 — Full HERA-DGM

```text
heterogeneous evidence
→ reliability routing
→ HGT
→ FS / FA / TS / TA
→ hypothesis-conditioned grounding
→ Noisy-OR
→ authenticity
```

---

# 62. Required Ablations

At minimum:

```text
w/o SigLIP2 semantic evidence

w/o DINOv3 dense/forensic evidence

w/o DeBERTa contextual evidence

w/o evidence reliability routing

GATv2 instead of HGT

w/o hypothesis → evidence feedback

generic grounding instead of hypothesis-conditioned grounding

independent binary head instead of Noisy-OR

shared TS/TA evidence aggregation

hypothesis-specific TS/TA aggregation
```

Ablations should preserve comparable backbones and training settings.

---

# 63. Primary Evaluation Metrics

## Binary detection

Report:

- AUC;
- Accuracy;
- EER.

## Fine-Grained Manipulation Attribution

Report:

- mAP;
- CF1;
- OF1.

CF1 is one of the primary metrics for HERA-DGM.

## Image grounding

Report:

- mean IoU;
- IoU@0.5;
- IoU@0.75.

## Text grounding

Report:

- Precision;
- Recall;
- F1.

Overall claims must not rely only on binary Accuracy.

---

# 64. Required Subgroup Evaluation

Results should be analyzed by:

### Atomic manipulation

```text
FS
FA
TS
TA
```

### Manipulation modality

```text
image-only
text-only
mixed
```

### Composition

```text
single manipulation
mixed manipulation
```

### Text evidence

```text
TS
TA
```

### Visual evidence scale

At least:

```text
small manipulated face
medium manipulated face
large manipulated face
```

using a documented grouping protocol.

---

# 65. Cross-Generator Evaluation

This is a required major analysis.

## Face Swap

```text
learn/train using SimSwap evidence
→ evaluate InfoSwap
```

and reverse:

```text
InfoSwap
→ SimSwap
```

## Face Attribute

```text
HFGI
→ StyleCLIP
```

and:

```text
StyleCLIP
→ HFGI
```

The official IID train/val/test split remains the primary benchmark.

Cross-generator experiments are separate generalization experiments.

---

# 66. Evidence Intervention Experiment

For hypothesis \(h\):

\[
\Delta_h^{evidence}
=
P(h|X)
-
P(h|X\setminus E_h).
\]

Compare to:

\[
\Delta_h^{random}
=
P(h|X)
-
P(h|X\setminus E_{random}).
\]

Expected scientific behavior:

\[
\Delta_h^{evidence}
>
\Delta_h^{random}.
\]

A meaningful margin supports the claim that selected evidence is functionally relevant to prediction.

This experiment is a core validation of HERA-DGM, not optional visualization.

---

# 67. Sampling

The dataset MUST NOT be blindly deduplicated.

Different manipulation variants can originate from the same source ID.

Optional engineering strategy:

```text
ID-aware batch sampling
```

to reduce over-concentration of related variants within individual batches.

This is not a novelty claim.

---

# 68. Checkpoint Selection

Test data must never select checkpoints.

Primary candidate validation metric:

```text
CF1
```

Possible tie-breakers:

```text
mAP
AUC
text grounding F1
image mean IoU
```

Exact selection hierarchy must be defined in YAML.

---

# 69. Reproducibility

Final major experiments should use multiple random seeds if computationally feasible.

Preferred:

```text
3 seeds
```

Final important metrics should report:

\[
mean\pm std
\]

where appropriate.

Development experiments may use one seed.

---

# 70. Run Logging

Every canonical experiment must preserve:

```text
run ID
git commit
config snapshot
random seed
dataset hashes
model checkpoint identifiers
library versions
GPU information
CUDA information
precision
parameter count
trainable parameter count
batch size
effective batch size
learning rates
best validation metrics
thresholds
test metrics
training duration
peak VRAM
```

---

# 71. Transformer/Environment Metadata

At runtime, the system SHOULD print and record:

```text
Python version
PyTorch version
Transformers version
CUDA runtime information
GPU model
available VRAM
```

This is particularly important because the project explicitly supports Transformers compatibility across API generations.

---

# 72. Scientific Kill Criteria

These are internal go/no-go thresholds, not publication guarantees.

## A — Attribution

Full HERA should improve CF1 by approximately ≥1 absolute point over a controlled same-backbone conventional baseline.

## B — Mixed Manipulation

Full HERA should provide meaningful gains on mixed-manipulation samples.

## C — Generalization

Cross-generator degradation should be lower than conventional baselines.

## D — Evidence Faithfulness

Removing selected evidence should reduce hypothesis confidence substantially more than removing comparable random evidence.

## E — Grounding

Attribution improvements should not come at the cost of severe grounding degradation.

If HERA only improves binary Accuracy while attribution, grounding, and generalization remain unchanged, the scientific thesis is not sufficiently supported.

---

# 73. Publication Interpretation

A ≥1 CF1 gain is an internal continuation signal.

It is **not sufficient by itself** to claim top-tier publication readiness.

A strong paper should ideally establish some combination of:

- improved CF1;
- competitive or improved standard DGM⁴ metrics;
- stronger mixed-manipulation behavior;
- significantly improved cross-generator robustness;
- evidence-intervention faithfulness;
- strong controlled ablations.

---

# 74. Implementation Philosophy

## Principle 1 — Thesis Before Module

A module should exist because it implements or tests a scientific hypothesis.

Do not add modules solely to make the architecture appear novel.

## Principle 2 — Strong Backbone Is a Control Variable

Performance gains caused only by stronger pretrained encoders are not HERA novelty.

## Principle 3 — Avoid Unnecessary Complexity

Do not introduce:

- MLLMs;
- RAG;
- external retrieval;
- RL;
- unnecessary generative models;
- large external databases.

## Principle 4 — Evidence Must Matter

The system should ideally demonstrate:

```text
what evidence was found
        ↓
which manipulation it supports
        ↓
how that manipulation produces the authenticity verdict
```

---

# 75. Minimum Repository Layout Before Phase 1

At the beginning of Phase 1, repository structure should minimally become:

```text
HERA-DGM/
│
├── config/
│
├── src/
│   ├── compat/
│   ├── data/
│   └── utils/
│
├── scripts/
│
├── tests/
│
├── docs/
│
├── outputs/
│
├── requirements.txt
└── README.md
```

Later model-specific directories may be introduced according to the Constitution.

---

# 76. Frozen Development Order

Project development proceeds sequentially.

```text
PHASE 0
Scientific & Engineering Constitution
[FROZEN]

PHASE 1
Dataset Contract
Preflight
Path Resolution
Tokenizer Alignment
Dataset Loader
Data Tests

PHASE 2
Strong Controlled Baselines

PHASE 3
Evidence Extraction Layer

PHASE 4
Hypothesis–Evidence Graph

PHASE 5
Hypothesis-Conditioned Grounding
+
Attribution-Derived Noisy-OR

PHASE 6
Evidence Reliability / Uncertainty Routing

PHASE 7
Full HERA-DGM Training

PHASE 8
Controlled Ablations

PHASE 9
Cross-Generator Generalization

PHASE 10
Evidence Intervention
+
Subgroup Analysis

PHASE 11
Final Reproducibility
Tables
Figures
Paper Results
```

Later phases must not be implemented prematurely unless explicitly authorized.

---

# 77. Phase 1 Scope Boundary

Phase 1 is strictly limited to:

- repository initialization required for data infrastructure;
- dataset schema;
- path resolver;
- dataset hash verification;
- preflight validation;
- label mapping;
- image preprocessing contracts;
- bbox transformation utilities;
- `fake_text_pos` → tokenizer alignment;
- DGM4 Dataset/DataLoader;
- optional sampler foundations;
- data-related YAML configuration;
- tqdm progress visibility;
- unit/integration tests for the data pipeline;
- requirements installation support.

Phase 1 MUST NOT implement:

- full SigLIP2 model pipeline;
- DINOv3 reasoning branch;
- HERA graph;
- HGT;
- hypothesis reasoning;
- Noisy-OR training;
- full HERA loss;
- full training pipeline.

Those belong to later phases.

---

# 78. Phase 1 Entry Conditions

All conditions are satisfied:

```text
✓ Dataset root known

✓ Dataset split sizes frozen

✓ Dataset SHA-256 frozen

✓ Image mapping verified

✓ 206,362 images fully decoded

✓ fake_text_pos semantics verified

✓ fake_image_box coordinates verified

✓ MTCNN boxes audited as unused metadata

✓ fake bbox supervision verified independently of model proposals

✓ generator-class mapping verified

✓ text-only image consistency verified

✓ official splits preserved

✓ Transformers compatibility rule frozen

✓ tqdm visibility mandatory

✓ sklearn classification_report mandatory

✓ threshold tuning mandatory for later training

✓ summary.csv mandatory for final evaluation

✓ requirements.txt mandatory

✓ config/src/scripts repository structure frozen

✓ YAML-only runtime settings frozen

✓ canonical CLI format frozen
```

Phase 1 is therefore authorized.

---

# 79. Final Identity of HERA-DGM

HERA-DGM is **not**:

```text
HAMMER + HGT
```

It is not:

```text
stronger encoders + classifier
```

It is not:

```text
a graph placed after multimodal fusion
```

Its intended identity is:

> **An evidence-centric attribution-first framework in which localized semantic and forensic evidence is routed toward explicit manipulation hypotheses, and authenticity is derived from those evidence-supported hypotheses.**

The project succeeds scientifically only if experiments show that this formulation produces more faithful, fine-grained, or generalizable manipulation detection than appropriate controlled alternatives.

---

# 80. Constitution Precedence Rule

This Phase 0 Final document replaces all previous Phase 0 drafts.

From this point onward:

```text
THIS DOCUMENT
=
SINGLE SOURCE OF TRUTH
```

If an older Phase 0 document conflicts with this version, this Phase 0 Final version wins.

If a future Phase requires changing a frozen scientific contract, the change must be explicitly discussed and documented as a Constitution amendment before implementation.

Agents must never silently reinterpret or modify this Constitution.
