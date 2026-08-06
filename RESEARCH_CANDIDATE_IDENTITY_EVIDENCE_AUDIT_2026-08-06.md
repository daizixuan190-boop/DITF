# Candidate Identity Evidence Audit (2026-08-06)

## Scope and protocol boundary

This is a read-only synthesis of artifacts already present in `D:\ditfjson`.
It introduces neither a matcher nor a selection threshold.  Ground truth is
used only to measure candidate rank and PCK, never to define an inference
rule.

There are two different evidence scopes and they must not be conflated:

| Scope | Artifact | What it can establish |
|---|---|---|
| Full SPair, 88,328 points / 12,234 pairs | `fjsar_frozen_flux_spectral_roma_identity_full_spair_seed2027_frozen_flux_spectral_roma_identity_audit.json` | Whether the existing spectral + RoMa representation produces a stable overall gain. |
| discovery20, 2,663 points / 360 pairs | attention top-20 candidate, RoMa, DINO, multi-layer, and anchor-transport audits | Which scores rank a PCK-valid candidate ahead of its top-20 distractors. |

No full-SPair top-20 candidate dump exists in the current directory.  The
full artifact contains point predictions and RoMa diagnostics but deliberately
records `attention_top1_pck_hit` and `attention_top20_pck_hit` as null.  It
must therefore not be cited as a full candidate-ranking experiment.

## Full-SPair result: the information-injection result is real but limited

The frozen global cosine matcher on all 88,328 annotated points reports:

| Representation | Point PCK |
|---|---:|
| Official DiTF baseline | 67.561 |
| Native + filtered attention spectral appearance | 68.406 |
| Spectral appearance + RoMa coordinate identity | 69.527 |

The final system makes 4,799 baseline-error rescues and 3,062 harms, for a
net +1,737 correct points (+1.967 PCK).  The incremental RoMa contribution
after spectral appearance is +1.121 PCK.  Direct warp is only 51.493 PCK and
identity-only is 39.585 PCK.  Thus RoMa is useful as complementary continuous
pair-conditioned evidence inside a representation; it is not a reliable
standalone candidate selector.

The logged full reliability fields do not provide a safe replacement: the
prior RoMa audit found the best rescue-vs-harm observable, composite
reliability, has only about 0.63 AUC across discovery and held-out splits.

## What separates a true proposal from its distractors?

All ranks below are measured only among the fixed FLUX attention top-20 pool.
`oracle_gap` means the native DiTF point is wrong, attention top-1 is wrong,
and at least one proposal is PCK-valid.  It is the high-value group for
recovering the attention oracle, not an inference-time label.

### Native FLUX candidate similarity is the baseline-preserving signal

On the saved 5-pair/category descriptor audit (714 points; 122 oracle-gap
points), native FLUX similarity ranks a PCK-valid attention proposal:

| Group | native top-1 | native top-3 | Median rank |
|---|---:|---:|---:|
| All proposal-covered points (665) | 72.18% | 88.27% | 1 |
| Oracle-gap points (122) | 14.75% | 50.82% | 3 |
| Attention-harms-native points (101) | 75.25% | 91.09% | 1 |

This explains both its value and its limitation.  It strongly preserves
baseline-like correct solutions, but it resolves only 18 of 122 high-value
oracle-gap cases.  A new branch must retain this evidence rather than replace
it.

### Existing FLUX-only local alternatives do not create identity

On the same oracle-gap group, local self-similarity reaches 10.66% top-1 and
attention-Jacobian evidence reaches 18.03% top-1.  On the larger
multilayer audit's 485 oracle-gap cases, ranking candidates with individual
official FLUX blocks gives only 48--70 top-1 hits (9.9--14.4%, depending on
block), despite 203--294 top-5 hits.  These are useful weak ordering signals,
not a safe identity decision.

This rules out the claim that the missing information is hidden in a different
single frozen FLUX layer or in simple local invariant pooling.  The
real-part competition audit independently agrees: in 63.13% of evaluated
feature failures, another annotated semantic part already outranks the GT
part under center similarity; enlarging local context to radius 4 improves
the mean margin by only 0.00194.

### Independent correspondence encoders do rank candidates, but cannot route safely

On the same 2,663-point discovery20 candidate pool:

| Candidate-only scorer | Top-1 | Top-3 | Top-5 | Median GT rank | Baseline rescues / harms |
|---|---:|---:|---:|---:|---:|
| RoMa bidirectional warp | 59.37 | 69.73 | 75.93 | 1 | 262 / 517 |
| DINOv2 token cosine | 56.59 | 75.33 | 82.20 | 1 | 193 / 522 |

For the harder `both_wrong_top20_hit` cohort (493 points), RoMa top-1 is
36.31% and DINO top-1 is 25.35%, against uniform candidate selection of
18.40%.  Therefore both contain genuine pair-conditioned candidate identity
evidence.  Their top-1 output is nevertheless below native DiTF because
neither score knows when the native prediction is already correct.

This is evidence for two separate requirements:

1. a candidate-conditioned relation score can add information beyond FLUX;
2. a final method needs a baseline-preserving uncertainty mechanism, not a
   hard replacement of DiTF by its candidate ranker.

### Geometry and anchors are not the missing discriminator

The certified-anchor audit is decisive here.  Observable cycle plus
attention-agreement anchors have 46.30% coverage and 85.16% post-hoc
precision, yet local affine transport produces 63.24 PCK against 68.76
baseline.  More importantly, supplying oracle-correct baseline anchors
(100% precision, 68.76% coverage) still yields only 65.30 PCK.  Thus neither
better anchor mining nor tuning local-affine support addresses the identity
selection error.

## Mechanism conclusion

The remaining high-value residual is not a lack of candidate retrieval and
not a lack of absolute or locally affine position.  It is a *relative
cross-instance part identity* problem:

```text
source local part role + source local appearance/context
                  compared with
target candidate local part role + target local appearance/context
                  under a shared pair-level transformation/visibility state.
```

Existing scores each cover only a projection of this relation:

- attention: semantic-region proposal coverage;
- native DiTF: global appearance similarity and baseline preservation;
- RoMa: learned pairwise geometric/appearance correspondence;
- DINO: discriminative local visual similarity;
- cycle/Jacobian/local affine: local geometric consistency.

None of the existing scalar scores determines identity reliably.  This is why
their direct rerankers damage the baseline and why a score-margin gate cannot
repair them.

### Hand-designed candidate relation fusion is also already negative

`fjsar_candidate_conditioned_verification_discovery20_seed2027_candidates.json`
tested an explicitly candidate-preserving, parameter-free rank vote over
attention posterior, native identity, candidate-centered local relation, and
weak anchor topology.  Its saved cohort contains 906 deliberately difficult
points (488 oracle-gap and 418 attention-harms-native).  It obtains 376
correct points, below the native baseline's 418 on exactly those same points.

This is a separate negative result from affine transport.  It rejects the
idea that a manually chosen combination of the currently logged local and
topological scalar features is the missing representation.  The artifact does
not preserve branch-level candidate values for a new retrospective separation
test, so no claim should be made about which individual hand-crafted term is
at fault.

## Research decision

Do not run full supervised candidate-decoder training, anchor transport, or
another hand-designed score fusion.  Their necessary evidence has been
falsified or is too weak.

Before implementing a new method, the next audit must measure a *joint,
candidate-conditioned relation representation* rather than another scalar:

1. form source-patch / target-candidate-patch relation features with a shared
   pair context, while retaining the candidate axis;
2. use labels only offline to test whether this joint relation separates the
   PCK-valid candidate from all 19 distractors on oracle-gap residuals;
3. separately test whether an observable uncertainty quantity can distinguish
   "native correct" from "candidate branch should override" without PCK-tuned
   gating;
4. proceed to a method only if this audit demonstrates a material, pair-held-
   out upper bound beyond native similarity and the geometry-only ceiling.

The source of such a relation is deliberately undecided by this audit.  The
evidence establishes the information contract, not a premature architecture.
