# DiTF Train-Free Attention Correspondence Workspace

This workspace is intentionally narrowed to the current SPair-71k research line:
train-free semantic correspondence with FLUX/DiTF features and pair-conditioned
cross-attention replay.

The active question is whether high attention top-K recall can be converted into
point-level identity without training, dataset-fitted parameters, selectors,
rerankers, gates, or fallback tricks.

## Active Entrypoints

- `eval_spair.py`: original DiTF SPair baseline evaluator.
- `eval_spair_matcher_ablation.py`: focused SPair evaluator for current attention-side modes.
- `spair_matchers.py`: cosine NN baseline plus FJSAR attention descriptors.
- `flux_joint_replay.py`: exact FLUX replay and cross-attention extraction.
- `analyze_fjsar_candidates.py`: lightweight attention proposal dump analysis.
- `tests/test_spair_matchers.py`: focused toy tests for replay and attention descriptors.

## Supported Matchers

`eval_spair_matcher_ablation.py --matcher` currently exposes only:

- `nn`: official cosine NN baseline.
- `fjsar_attn`: pure attention diagnostic, not intended as the final method.
- `fjsar_attention_signature`: attention distribution as identity signature.
- `fjsar_part_sharpen`: attention-region common component removal.
- `fjsar_orthogonal_context`: native identity with orthogonal attention context.
- `fjsar_dense_partial_graph_matching`: attention-only candidates with dense
  relation edges and exact sparse partial assignment.
- `fjsar_expert_preserving_attention_hypothesis_conditioned_replay`: exact
  attention candidates with expert-preserving QK/V identity routing.
- `fjsar_all`: shared-feature pair5 audit for all attention-side modes.

Deprecated selector/rerank/fusion/CSLS/local-graph/JGM and SPair ownership
method experiments have been removed from the active workspace. AP-10K, DAVIS,
and baseline reproduction utilities are preserved as reproduction code, not as
the current method-development surface.

## Minimal Verification

The local Python environment used during cleanup did not include `pytest`.
Focused tests can still be run with a simple import runner, or with `pytest`
after installing the project environment.

Compilation check:

```bash
python -m py_compile spair_matchers.py eval_spair_matcher_ablation.py flux_joint_replay.py flux_multiblock.py tests/test_spair_matchers.py
```

## Latent Expert Mechanism Audit

`--fjsar_latent_expert_audit` is diagnostic-only. It keeps the raw
`[ensemble, head, point, candidate]` exact bidirectional cross probabilities
(with bidirectional ranks retained as a control) and asks
whether a stable minority head/member can recover mutual-attention top-20
candidates that are lost by early ensemble/head averaging. Predictions are not
changed, candidates come only from mutual cross-attention, and GT is used only
for explicitly labelled upper bounds and post-hoc PCK statistics.

The audit requires every annotated point in each image pair, so it enforces
`--fjsar_dump_case_filter all` and `--fjsar_dump_max_records 0`. It writes a
detailed `*_latent_expert_audit.json` plus a compact
`*_latent_expert_summary.json`.

The remote workspace already has the canonical replay-state cache. Reuse it
strictly; do not repopulate an overlapping cache:

```bash
python eval_spair_matcher_ablation.py \
  --dataset_path /root/autodl-tmp/datasets/SPair-71k \
  --save_path Features/spair_flux_main_640_e8_cd \
  --output_json analysis/spair_matcher_ablation/fjsar_attention_latent_expert_audit_discovery20_seed2027.json \
  --matcher fjsar_attn \
  --img_size 640 640 \
  --t 260 \
  --k 28 \
  --ensemble_size 8 \
  --cd \
  --subset discovery \
  --pairs_per_cat 20 \
  --split_seed 2027 \
  --fjsar_candidate_topk 20 \
  --fjsar_oracle_audit \
  --fjsar_oracle_topk 1 3 5 10 20 \
  --fjsar_latent_expert_audit \
  --fjsar_latent_expert_topk 1 3 5 10 20 \
  --fjsar_memory_cache_gb 4 \
  --fjsar_disk_cache_path /root/autodl-tmp/cache/fjsar_spair_flux_640_e8_cd \
  --fjsar_require_disk_cache
```

## Dense Candidate-Edge Separability Audit

`--fjsar_dense_candidate_edge_audit` is a falsification audit, not a matcher.
It builds a partial-assignment data contract over every dense FLUX source token,
keeps only mutual cross-attention top-K target candidates, and sends one
max-product message over directed local source-grid edges. DiTF features are
used only to compare source/target edge self-similarity. No native candidate,
gate, fallback, or GT enters scoring, and the matcher prediction is unchanged.

The detailed JSON retains candidate-level scores and graph contracts. The
compact summary reports edge-only recovery, harm, net lift over attention top-1,
and lift over the uniform candidate expectation. A full partial graph solver
should be implemented only if this evidence separates candidates on oracle-gap
points.

```bash
python eval_spair_matcher_ablation.py \
  --dataset_path /root/autodl-tmp/datasets/SPair-71k \
  --save_path Features/spair_flux_main_640_e8_cd \
  --output_json analysis/spair_matcher_ablation/fjsar_dense_candidate_edge_discovery20_seed2027.json \
  --matcher fjsar_attn \
  --img_size 640 640 \
  --t 260 \
  --k 28 \
  --ensemble_size 8 \
  --cd \
  --subset discovery \
  --pairs_per_cat 20 \
  --split_seed 2027 \
  --fjsar_candidate_topk 20 \
  --fjsar_oracle_audit \
  --fjsar_oracle_topk 1 3 5 10 20 \
  --fjsar_dense_candidate_edge_audit \
  --fjsar_dense_candidate_edge_radius 1 \
  --fjsar_dump_case_filter all \
  --fjsar_dump_max_records 0 \
  --fjsar_memory_cache_gb 4 \
  --fjsar_disk_cache_path /root/autodl-tmp/cache/fjsar_spair_flux_640_e8_cd \
  --fjsar_require_disk_cache
```

The discovery20 seed-2027 audit met the minimum threshold for that solver, but
the evidence is weak enough that the resulting matcher remains a falsifiable
experiment rather than a validated improvement. On 494 attention-top1-error /
attention-top20-hit points, relation-edge top1 recovered 22.06% versus an
18.51% uniform-candidate expectation (+3.55 points). The point-weighted lift
has a pair-clustered standard error of 1.80 points; the pair-equal lift is
+5.67 +/- 2.26 points, and 12 of 18 categories have positive lift. Recovered
candidates have mean original attention rank 10.33. Spatial edges are below
uniform (-2.12 points) and joint edges are effectively uniform (+0.31 points),
so neither enters the matcher.

## Dense Attention Partial Graph Matching

`fjsar_dense_partial_graph_matching` turns the supported relation-edge signal
into a complete matching problem. Every dense source token is a graph node and
retains only its mutual cross-attention top-K targets. The unary is standardized
attention log probability; the only pairwise signal is the audited local DiTF
self-similarity relation message. An exact sparse maximum-weight bipartite
solver gives every real target unit capacity, while optional context nodes have
private dustbins. Annotated query source tokens must receive a real attention
candidate. DiTF descriptors never enter the unary, and there is no native
candidate injection, gate, fallback, or GT-dependent inference.

The run writes the requested result JSON and automatically adds detailed
`*_dense_partial_graph_audit.json` and compact
`*_dense_partial_graph_summary.json` files. Both report recovery/harm relative
to attention top1, collision reduction, dustbin use, category stability, and
native/GT contract violations.

```bash
python eval_spair_matcher_ablation.py \
  --dataset_path /root/autodl-tmp/datasets/SPair-71k \
  --save_path Features/spair_flux_main_640_e8_cd \
  --output_json analysis/spair_matcher_ablation/fjsar_dense_partial_graph_matching_discovery20_seed2027.json \
  --matcher fjsar_dense_partial_graph_matching \
  --img_size 640 640 \
  --t 260 \
  --k 28 \
  --ensemble_size 8 \
  --cd \
  --subset discovery \
  --pairs_per_cat 20 \
  --split_seed 2027 \
  --fjsar_candidate_topk 20 \
  --fjsar_oracle_audit \
  --fjsar_oracle_topk 1 3 5 10 20 \
  --fjsar_memory_cache_gb 4 \
  --fjsar_disk_cache_path /root/autodl-tmp/cache/fjsar_spair_flux_640_e8_cd \
  --fjsar_require_disk_cache
```

The discovery20 seed-2027 result falsified the graph-assignment mechanism as a
matching method: exact attention reached 59.56 point PCK, relation-only 43.41,
attention plus relation belief 55.99, and partial assignment 54.11. The solver
removed target collisions, but collision rate did not predict gain and the
final solver harmed more attention-correct points than it recovered. This route
is therefore closed; do not tune graph weights, radius, capacity, or dustbins.

## Expert-Preserving Attention Hypothesis-Conditioned Replay

`fjsar_expert_preserving_attention_hypothesis_conditioned_replay` tests the
remaining attention-side mechanism directly. Mutual exact cross-attention still
creates the only candidate pool. For every candidate it retains the full
`[ensemble, head, point, candidate]` tensor and computes two independent pieces
of evidence before expert averaging: exact bidirectional QK support and
symmetric candidate-specific V residual identity. The V residual subtracts the
attention basin common component in both directions; candidate values are never
averaged into a readout descriptor.

Within each head, QK and V signals are standardized only across that point's
candidates. A pair-level head is selected by QK/V candidate-ranking agreement,
then standardized aggregate attention, selected-head QK, and selected-head V
identity are added with fixed equal weight. There is no fitted parameter,
native candidate, native gate, native fallback, or GT-dependent inference.
Pair-expert and point-head variants are saved only as diagnostic controls; they
are not used for the reported method PCK.

The run automatically writes detailed `*_expert_hypothesis_audit.json` and
compact `*_expert_hypothesis_summary.json`. They report each control signal's
top-K, recovery and harm relative to exact attention top1, selected-head and
expert histograms, agreement margins, category groups, and all native/GT
contract violations.

```bash
python eval_spair_matcher_ablation.py \
  --dataset_path /root/autodl-tmp/datasets/SPair-71k \
  --save_path Features/spair_flux_main_640_e8_cd \
  --output_json analysis/spair_matcher_ablation/fjsar_expert_preserving_attention_hypothesis_conditioned_replay_discovery20_seed2027.json \
  --matcher fjsar_expert_preserving_attention_hypothesis_conditioned_replay \
  --img_size 640 640 \
  --t 260 \
  --k 28 \
  --ensemble_size 8 \
  --cd \
  --subset discovery \
  --pairs_per_cat 20 \
  --split_seed 2027 \
  --fjsar_candidate_topk 20 \
  --fjsar_oracle_audit \
  --fjsar_oracle_topk 1 3 5 10 20 \
  --fjsar_memory_cache_gb 4 \
  --fjsar_disk_cache_path /root/autodl-tmp/cache/fjsar_spair_flux_640_e8_cd \
  --fjsar_require_disk_cache
```

## Candidate Identity Decodability Audit

`--fjsar_identity_decodability_audit` is a supervised mechanism diagnostic,
not a matcher and not a reported unsupervised result. It freezes the established
mutual-attention top-20 pool and exports candidate-aligned state families from
the real cached FLUX block boundary: aggregate attention, every ensemble/head
QK signal, every ensemble/head value/readout signal, and symmetric relations
from block input, Q/K/V, MLP, native block output, and true cross-attention block
output. A fixed annotation-free CountSketch also exposes channel-wise products
and absolute differences without serializing every raw 3072/12288-dimensional
candidate pair. Native cosine and absolute geometry are isolated as controls
and never enter the main `all_internal` probe.

The evaluator writes binary per-pair shards, a manifest, and a compact summary.
Linear probes for every state family and one shallow nonlinear all-state probe
use deterministic category-held-out outer folds. Scaling and fitting see only
outer-training categories. GT creates candidate labels after the attention pool
is frozen; it never enters feature construction. A positive result proves that
candidate identity is decodable from the audited state families. A negative
result bounds these probes but is not described as information-theoretic proof
of absence.

The audit accepts exactly one replay feature source: a complete canonical cache,
or explicit fresh extraction via `--extract_native_in_memory`. Fresh extraction
uses the bounded in-run RAM cache and does not write persistent replay entries;
the required per-pair probe shards can be directed to `/tmp`:

```bash
python eval_spair_matcher_ablation.py \
  --dataset_path /root/autodl-tmp/datasets/SPair-71k \
  --save_path Features/spair_flux_main_640_e8_cd \
  --output_json analysis/spair_matcher_ablation/fjsar_identity_decodability_discovery20_seed2027.json \
  --matcher fjsar_attn \
  --img_size 640 640 \
  --t 260 \
  --k 28 \
  --ensemble_size 8 \
  --cd \
  --subset discovery \
  --pairs_per_cat 20 \
  --split_seed 2027 \
  --fjsar_candidate_topk 20 \
  --fjsar_oracle_audit \
  --fjsar_oracle_topk 1 3 5 10 20 \
  --fjsar_identity_decodability_audit \
  --fjsar_identity_decodability_folds 3 \
  --fjsar_dump_case_filter all \
  --fjsar_dump_max_records 0 \
  --fjsar_memory_cache_gb 4 \
  --extract_native_in_memory \
  --fjsar_identity_decodability_shard_path /tmp/fjsar_identity_decodability_shards
```

For a complete canonical cache, replace the final two flags with
`--fjsar_disk_cache_path PATH --fjsar_require_disk_cache`.

The primary decision fields are
`mechanism_decision.best_internal_category_heldout_top1`,
`supervised_probe_reaches_80`, and `supervised_probe_reaches_90` in the
generated `*_identity_decodability_summary.json`. Probe numbers must never be
reported as train-free PCK, and reaching 80 does not itself establish an
unsupervised 80-point method.

For the nonlinear capacity check after a linear audit, use only the stable
internal families and the explicit native-control comparison. This path is a
supervised diagnostic, keeps category-held-out folds, and may use CUDA for the
one-hidden-layer PyTorch probe:

```bash
python analyze_identity_decodability.py \
  --manifest analysis/spair_matcher_ablation/fjsar_identity_decodability_discovery20_seed2027_v5_identity_decodability_manifest.json \
  --output_json analysis/spair_matcher_ablation/fjsar_identity_decodability_discovery20_seed2027_v5_torch_mlp_summary.json \
  --fold_count 3 \
  --seed 2027 \
  --probe_names stable_internal native_plus_stable_internal \
  --mlp_only \
  --torch_mlp \
  --device cuda
```

`stable_internal` excludes the auxiliary CountSketch channel family. The
channel sketch is clipped before future float16 shard serialization; existing
shards with recorded non-finite values should not be used as the final
mechanism claim.

## Candidate-Clamped Causal Replay Audit

`--fjsar_candidate_clamped_causal_replay_audit` is a feature-side causal audit,
not a matcher. Exact mutual cross-attention supplies the only top-20 candidate
pool. For every source point and candidate, block 27 is replayed as an
independent bidirectional hypothesis: the exact local attention contribution
and each ensemble/head's original total cross mass are preserved, while only
the conditional cross value is replaced by the candidate token in both
directions. The intervention then passes through the real attention slice of
`linear2`, the original AdaLN gate, and the residual path; the parallel MLP
contribution remains exactly the normal block output.

The adjacent block 28 is fully unclamped. Its unmodified Q/K operator reads the
intervened source and target states, and the primary candidate score is the
mean bidirectional negative log rank after this release. The forced block-27
cosine is never used. Pre-intervention rank, causal rank improvement, mutual
top-1 vote, and mutual top-5 vote are retained as controls. The audit does not
change matcher predictions and contains no native candidate, native gate,
fallback, fitted selector, or GT-dependent score.

The run writes detailed
`*_candidate_clamped_causal_replay_audit.json` and compact
`*_candidate_clamped_causal_replay_summary.json`. The primary falsification set
is `both_wrong_top20_hit`: baseline wrong, attention top-1 wrong, but attention
top-20 contains a PCK-correct candidate. The summary reports top-1/3/5/10,
recovery/harm/net versus attention top-1, candidate-uniform expectation,
pair-equal lift and its standard error, category lift, cross mass, intervention
strength, and contract violations. Do not build a matcher unless post-release
top-1 exceeds 18.40% and its pair-equal lift is larger than 1.96 standard errors.
For the observed 493-point gap set, reaching roughly 75 point PCK would require
about 162 recoveries, or 32.9%.

This command strictly reuses the canonical replay-state cache and refuses a
missing or stale entry; it does not create another cache:

```bash
python eval_spair_matcher_ablation.py \
  --dataset_path /root/autodl-tmp/datasets/SPair-71k \
  --save_path Features/spair_flux_main_640_e8_cd \
  --output_json analysis/spair_matcher_ablation/fjsar_candidate_clamped_causal_replay_discovery20_seed2027.json \
  --matcher fjsar_attn \
  --img_size 640 640 \
  --t 260 \
  --k 28 \
  --ensemble_size 8 \
  --cd \
  --subset discovery \
  --pairs_per_cat 20 \
  --split_seed 2027 \
  --fjsar_candidate_topk 20 \
  --fjsar_oracle_audit \
  --fjsar_oracle_topk 1 3 5 10 20 \
  --fjsar_candidate_clamped_causal_replay_audit \
  --fjsar_candidate_clamped_causal_replay_topk 1 3 5 10 20 \
  --fjsar_dump_case_filter all \
  --fjsar_dump_max_records 0 \
  --fjsar_memory_cache_gb 4 \
  --fjsar_disk_cache_path /root/autodl-tmp/cache/fjsar_spair_flux_640_e8_cd \
  --fjsar_require_disk_cache
```

Pair5 attention-side smoke sweep should keep the official SPair geometry and
compare all attention-side modes on the same extracted pair features:

```bash
python eval_spair_matcher_ablation.py \
  --dataset_path /root/autodl-tmp/datasets/SPair-71k \
  --save_path Features/spair_flux_main_640_e8_cd \
  --output_json analysis/spair_matcher_ablation/fjsar_all_discovery_5pc_seed2027.json \
  --matcher fjsar_all \
  --img_size 640 640 \
  --t 260 \
  --k 28 \
  --ensemble_size 8 \
  --cd \
  --subset discovery \
  --pairs_per_cat 5 \
  --split_seed 2027 \
  --fjsar_oracle_audit \
  --fjsar_memory_cache_gb 4 \
  --fjsar_disk_cache_path /root/autodl-tmp/cache/fjsar_spair_flux_640_e8_cd \
  --fjsar_require_disk_cache
```
