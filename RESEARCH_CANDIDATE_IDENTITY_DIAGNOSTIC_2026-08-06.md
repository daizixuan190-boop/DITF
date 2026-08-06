# Candidate-conditioned identity capacity diagnostic (2026-08-06)

## Why this experiment exists

Full-SPair evidence fixes the immediate problem more narrowly than “add more
features”:

- frozen DiTF baseline: about 67.56 point PCK;
- spectral attention + RoMa system: about 69.53 (+1.97);
- attention top-20 recall: about 92;
- simple RoMa confidence/cycle/Jacobian gates add only about 0.04;
- same-image deformation, native cycle, shared prototypes, and absolute token
  identity all fail to resolve cross-instance semantic part identity.

Therefore the next question is a capacity question: can a pair-conditioned
decoder learn the missing choice rule from the information already present in
the frozen FLUX/DiTF and cross-image attention candidate set?

## Locked experiment

Candidate pool:

1. frozen cross-image mutual-attention top-20;
2. frozen original DiTF top-1 as one explicit fallback candidate.

Decoder:

- reuses `CandidateIdentityVerifier`;
- sees candidate-conditioned FLUX attention, Q/K, value, token, channel,
  native-cosine, and normalized geometry evidence;
- uses cross-candidate and cross-query context;
- receives no category name or keypoint identity;
- uses no DINO and no RoMa;
- ground truth never enters candidate feature construction or inference.

Supervision is deliberately diagnostic, not label-free: only SPair `trn`
keypoint correspondences and boxes construct the listwise PCK-aligned target.
Every image appearing in the SPair test split is excluded from training.

## Required evidence saved

Evaluation saves, per point:

- baseline, attention, and decoder predictions;
- all 21 candidates, scores, and PCK hits;
- selected rank, candidate kind, and top-1/top-2 margin;
- attention top-20 and 21-candidate pool Oracle hits;
- an explicit `gt_used_for_inference=false` marker.

Summaries report:

- baseline/attention/decoder/attention-Oracle/pool-Oracle PCK;
- rescued, harmed, and net correct points versus baseline;
- baseline-correct retention;
- Oracle-gap recovery;
- baseline-fallback selection rate;
- per-category results.

## Decision rule

- If discovery20 and heldout20 reach at least 75 with high baseline retention,
  the candidate representation has sufficient information. The research task
  becomes replacing SPair labels with a general, label-free source of the
  learned cross-instance identity constraint.
- If training PCK rises but held-out pair20 does not, the decoder memorizes
  dataset/category regularities; inspect category failures and score margins
  before changing the architecture.
- The training JSON reports online pre-update predictions, loss, and gradient
  norms; it is an optimization-health trace, not a final training-set capacity
  score. Capacity decisions must use the held-out pair20 evaluation.
- If held-out evaluation cannot recover a material fraction of the pool Oracle
  gap despite healthy optimization, use the saved failure records to identify
  the missing factor (symmetry, occlusion, viewpoint, or part structure)
  rather than tuning gates.
- A result below 75 is still conclusive only when pool Oracle remains high and
  training optimization is healthy; otherwise first diagnose candidate recall
  or optimization from the recorded recoverability, loss, and gradient norms.
