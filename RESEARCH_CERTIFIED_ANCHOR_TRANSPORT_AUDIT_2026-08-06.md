# Certified-anchor local transport audit (2026-08-06)

## Evidence motivating this audit

The supervised discovery20 capacity diagnostic established:

- baseline point PCK: 68.76;
- frozen attention top-20 Oracle: 92.11;
- baseline plus attention pool Oracle: 93.39;
- 656 points are baseline errors with a correct attention candidate;
- the 64-pair decoder rescued 162 but harmed 237 baseline-correct points.

Score-margin thresholding cannot improve on the baseline, even when its
threshold is selected post-hoc.  Thus generic confidence gating is rejected.

A label-only leave-one-out audit of the saved candidate JSON then found that
local affine transport from other true correspondences identifies 57.28% of
routeable residual points.  With only oracle-known correct baseline anchors,
the rate is 45.83%; with all baseline anchors it falls to 30.74%.

The mechanism hypothesis is therefore narrow and falsifiable: the candidate
set contains enough pair-conditioned local spatial information, but a usable
method requires reliable anchors without target labels.

## Label-free protocol

An observed anchor requires both:

1. frozen DiTF forward/backward nearest-neighbour cycle closure, normalized by
   source feature-cell diagonal;
2. frozen DiTF baseline and frozen cross-attention top-1 spatial agreement,
   normalized by target feature-cell diagonal.

Certified anchor target locations fit a leave-one-out local affine map from a
query's nearest source-point neighbours.  The nearest attention top-20
candidate to the transport prediction is considered only when the transform
has three non-collinear anchors and cell-normalized support.  Otherwise the
method returns the original DiTF baseline exactly.

No category labels, target keypoints, RoMa, DINO, learned parameters, or
PCK-tuned inference threshold may enter this path.  GT is stored only to
measure anchor precision, coverage, rescues/harm, and an explicitly labelled
oracle-anchor ceiling.

## Decision rule

- Positive direction: observable anchors have materially higher precision than
  baseline, enough coverage for local support, and the baseline-preserving
  transport rule produces positive net PCK on discovery20 before heldout20.
- Negative direction: if observed anchors are too sparse/impure or transport
  cannot outperform baseline while the oracle-anchor ceiling is high, the
  unresolved problem is self-supervised anchor certification, not local
  geometry; do not tune transport thresholds as a final method.
- If both observable and oracle-anchor transport are weak, reject this spatial
  route and seek a different information source for candidate identity.
