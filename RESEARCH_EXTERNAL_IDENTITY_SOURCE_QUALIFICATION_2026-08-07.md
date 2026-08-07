# External Identity Source Qualification (2026-08-07)

## Decision boundary

The locked train-free system has a real full-SPair improvement from 67.561 to
69.527 point PCK, but its fixed attention proposal pool still leaves a
candidate-identity problem. This note does not propose a matcher. It records
whether the external sources reviewed on 2026-08-07 are suitable for a
strict, low-cost candidate-capacity experiment.

The current evidence boundary is the GT-only RoMa relation capacity result:
pair-heldout strict-current-residual top-1 = 35.31, top-3 = 61.48,
category-median = 34.85. It failed its pre-registered 45% / +10 point gate.
Therefore accessible RoMa local, multi-scale, and GP relation fields are
closed as an identity source. This does not imply that canonical 3D part
information has been tested.

## Protocol correction: Geometry Matters / 3D-SC

The two supplied links refer to the same work:
`Geometry Matters: 3D Foundation Priors for Learning Semantic Correspondence`
(arXiv:2605.30093, 3D-SC).

The frequently cited **76.3** in Table C2 is the mean of the 18
per-category, **per-keypoint** PCK@0.1 values. It is not the main-table
benchmark value and must not be directly compared with a system using a
different global point aggregation. The paper's main Table 1 reports
**73.0 per-image PCK@0.1** on SPair-71k for 3D-SC. The authors themselves
exclude Geo-SC from their main comparison because that method reports a
different per-keypoint aggregation. Any future comparison must report both
the identical SPair split/normalization/aggregation and the current DiTF
metric; it must not make a 69.53-vs-76.3 claim without that reconciliation.

## What 3D-SC actually uses

3D-SC is not a frozen PartField concatenation. Its described pipeline is:

1. SAM3 mask plus SAM3D single-image mesh and camera estimation;
2. per-instance render-and-compare scale/translation pose refinement;
3. OrientAnything V2-based yaw canonicalization;
4. PartField per-vertex descriptors rasterized into each image;
5. DINO + SD + PartField candidate generation and cross-mesh geodesic
   pseudo-label filtering;
6. a four-layer, 5M correspondence adapter trained for 200k iterations on
   the retained pseudo-labels.

The paper states that the SAM3 mask is obtained using the image together with
the dataset-provided category label. It therefore avoids manual match and
pose annotations, but it is not category-label-free at inference. A DiTF
extension claiming category-agnostic or label-free matching cannot silently
inherit this condition; it must either supply an equally general instance
mask route or report that extra test-time input explicitly.

Its useful information is materially different from the closed RoMa branch:
canonical, surface-relative part descriptors and a cross-instance mesh
correspondence. This directly targets repeated parts, left/right and
front/rear ambiguity. However, the paper reports that PartField is trained
with a part-level contrastive objective on rigid objects and has negative or
weak gains on several non-rigid categories. That limitation is aligned with
our current atlas, in which the residual also includes articulated animal
parts and high-viewpoint change.

At the time of this review the public `GenIntel/3D-SC` repository explicitly
says **"Code coming soon"**. Thus 3D-SC itself provides no reproducible
frozen inference interface yet. Its paper estimates 12.42 seconds per object
for canonical reconstruction, about 18 hours for full-SPair pseudo-label
generation, and about 4 GPU-hours of adapter training on one L40. That is a
research pipeline, not a budget-compatible first experiment and cannot be
copied as the proposed DiTF method.

## Describe Anything qualification

Describe Anything (DAM, ICCV 2025) accepts a point/box/scribble/mask and
returns detailed localized natural-language descriptions. Its public README
documents description generation and demos, not a frozen dense canonical
part descriptor, cross-image alignment representation, or a pretrained
candidate comparison metric. It uses a 3B model under an NVIDIA
non-commercial model license.

Therefore DAM is **not qualified** for the immediate capacity audit. A
source-point plus twenty target candidates would turn each correspondence
into expensive localized caption generation and text arbitration, recreating
a scalar/cross-modal routing problem without evidence that the output
encodes stable front/rear, left/right, or repeated-part identity. It should
not consume GPU time unless a documented, frozen per-region embedding and
label-free cross-instance comparison rule are first found.

## Qualification table

| Source | New identity information beyond FLUX/RoMa | Frozen candidate-axis interface now | Method overlap / cost | Decision |
|---|---|---|---|---|
| 3D-SC complete pipeline | Yes: canonical surface-relative PartField + mesh relation | No released 3D-SC interface; upstream components still require separate verification | Very high overlap with published reconstruction, pseudo-label and adapter pipeline; full preprocessing/training is costly | Keep only as a future source audit, do not copy or train |
| PartField alone rendered on reconstructed meshes | Plausibly yes, but unverified on our data/protocol | Not yet verified: needs official weights, mesh input contract, SAM3D and canonical-pose path | Could be a distinct frozen identity branch only if all pieces are available | Verify public interfaces before GPU work |
| Describe Anything | Region semantic text, not demonstrated canonical identity | No documented vector relation interface | 3B inference per candidate; non-commercial weights | Close for current capacity audit |
| RoMa internal relation fields | Already tested | Yes | GT-only capacity failed at 35.31% residual top-1 | Closed |

## Formal interface qualification protocol

No proposed identity source may enter an 18-pair smoke test until each item
below is answered from the primary paper and public code. Unknown is a fail,
not permission to guess an interface.

| Requirement | Pass condition | Disqualifying result |
|---|---|---|
| Deployment input | Inference uses only image plus an optional marked point or category-agnostic object mask; no SPair category, keypoint, pose or pairing label | Dataset category text, keypoint, pose, GT mask, or any test-time correspondence label is required |
| Candidate identity output | Exports a candidate-level vector, canonical coordinate, or part description for source and every target candidate | Only a mask, box, depth, flow, confidence, or scalar post-hoc score is exposed |
| Shared comparison space | The two instances produce directly comparable outputs in a documented shared space, with a defined source-to-candidate similarity or relation | Per-instance clusters, local scores, or a representation with no cross-instance comparison contract |
| Object context | The representation conditions the point on the full object / object structure rather than a local crop alone | Point appearance or local patch alone is the claimed identity signal |
| Viewpoint, visibility, symmetry | A documented canonicalization or 3D/object-centric mechanism resolves these factors before comparison | They are addressed only by an image-space cycle, threshold, reranking, or hand-built consistency score |
| Innovation boundary | The same representation-to-correspondence pipeline is not already the claimed method in semantic correspondence | A direct feature concatenation, pseudo-label training, or adapter reproduction of an existing correspondence paper |

The two mandatory conditions are **candidate identity output** and **shared
comparison space**. Failing either closes the source. Passing them alone is
not sufficient for a research route.

### PartField status against the protocol

The official PartField paper and `nv-tlabs/PartField` code establish the
following without interpretation:

| Requirement | Evidence | Status |
|---|---|---|
| Deployment input | The public inference program accepts `.obj`/`.glb` mesh files or `.ply` point/splat files. It has no image + marked-point/mask inference path. 3D-SC obtains that mesh through a separate SAM3D path and uses a dataset-provided category label for its SAM3 mask. | **Fail as a standalone deployable source** |
| Candidate identity output | `model_trainer_pvcnn_only_demo.py` exports `part_feat_*.npy`, a 448-dimensional feature per sampled face/point. | Pass |
| Shared comparison space | PartField reports that cross-shape consistency *emerges* and shows correspondence through Functional Maps, but provides neither a canonical coordinate nor a cross-instance identity guarantee. | Conditional, unproven |
| Object context | The encoder consumes the complete sampled 3D shape and predicts a triplane feature field. | Pass |
| Viewpoint / visibility / symmetry | PartField alone consumes a 3D shape but does not supply image-to-mesh alignment, visibility, or canonical yaw. 3D-SC adds pose refinement and OrientAnything yaw canonicalization externally. | Fail as a standalone source |
| Innovation boundary | PartField already presents correspondence as an application, and 3D-SC already renders its descriptors into images for semantic correspondence. | High overlap |

**PartField-alone decision: reject before smoke.** This is not a claim that
the features are useless. It is a precise statement that an image-to-mesh and
canonicalization composite has not yet passed the deployment and novelty
requirements, so a FLUX+PartField experiment would currently be an
uncontrolled reproduction rather than evidence for a new method.

### PartField as an offline-only teacher: conditional, not yet a route

Using PartField only to supervise an image-only student would be materially
different from running reconstruction and PartField on SPair test images.
It could be considered only after a teacher-side cross-shape capacity audit.
The audit itself must not be described as zero cost: PartField inference is a
GPU computation and requires an external 3D dataset plus the released
Objaverse-trained checkpoint. It has zero SPair-label cost, not zero compute
cost.

Before spending GPU time, a data/interface preflight must identify a public
3D benchmark with **cross-instance equivalent-part annotations**. Per-shape
part segmentation mIoU is insufficient: it can certify that each mesh is
partitioned, but cannot test whether a source front wheel and target front
wheel occupy a shared feature-space role rather than merely both being
"wheel". The frozen evaluation manifest must contain cross-shape queries for
front/rear, repeated parts, symmetric endpoints, head/tail and articulated
limbs where the selected dataset supports them.

The teacher capacity audit has two non-interchangeable conditions:

1. same-instance multi-view consistency is a rendering sanity check only;
2. cross-instance nearest-neighbour part-role accuracy is the actual gate.

It must additionally report an independently random-rotation control for
each target shape. A large drop when canonical CAD orientation is scrambled
means the descriptor relies on supplied shape alignment rather than encoding
the transferable identity needed by an image-only student. In that case a
canonicalization source would be required, returning the route to the
already-published 3D-SC family; do not distill it.

The first teacher run is capped at a small fixed manifest and one GPU-hour.
It advances only if the manifest, checkpoint, required 3D dependencies, and
evaluation labels have passed the data preflight. There is no large mesh
download, rendering corpus, student training, or SPair experiment before
this audit passes.

### Current teacher-audit readiness (2026-08-07)

The local DiTF workspace contains no PartField checkpoint, mesh/point-cloud
asset, PartNet/Objaverse/ShapeNet evaluation data, or cross-instance 3D part
annotations. The PartField README mentions PartObjaverse-Tiny for
per-category segmentation mIoU, but its public metric implementation and
annotation contract could not be retrieved in the current network
environment. Therefore it is **not verified** that PartObjaverse-Tiny
contains the cross-instance role labels needed here. It must not be used as
the teacher audit merely because it is mentioned by the PartField repository.

The audit is consequently `NOT READY`, not failed. To change this status,
the following must be documented before downloading/running anything:

1. exact public dataset release and license;
2. mesh paths plus part annotations for at least two instances per role;
3. annotation semantics that distinguish repeated roles when claimed
   (for example front versus rear rather than only wheel);
4. original coordinate-frame provenance sufficient to define the independent
   random-rotation control; and
5. the official PartField Objaverse checkpoint URL, checksum and compatible
   inference environment.

## Step 1.5 front-end audit correction: SAM 3D Objects

The official `facebookresearch/sam-3d-objects` code narrows the previously
uncertain image-to-surface interface, but does not yet make the SPair teacher
capacity experiment valid.

Verified from the public `notebook/inference.py` and
`inference_pipeline_pointmap.py`:

- the public call is `Inference(image, mask, seed=...)`, where the mask is
  placed in the input RGBA alpha channel; no category argument is required;
- the pipeline computes an input-aligned dense `pointmap` from its depth
  model, infers camera intrinsics when the depth model does not provide them,
  and returns the downsampled `pointmap` with the final output;
- the pipeline predicts `rotation`, `translation`, and `scale`, and can
  return a `glb` mesh or Gaussian output;
- pose/layout post-optimization is optional. The public `Inference.__call__`
  currently calls `run(..., with_layout_postprocess=False, ...)`.

This means the pixel-to-3D-point arrow is plausible through the returned
pointmap, but the pixel-to-**PartField surface** arrow is not yet proven. The
pointmap frame, reconstructed mesh/gaussian frame, pose inversion, visibility,
and nearest-surface error must be measured on a single image. The public
README's example uses `output["gs"]`, while the current pipeline code exposes
`glb`/`gaussian`; the probe must print actual keys and must not assume the
README alias.

The first permissible Step 1.5 operation is therefore a one-image, no-label
interface probe:

```text
category-free mask
 -> SAM 3D Objects
 -> actual output keys + pointmap + pose
 -> exact frame transform
 -> mesh/gaussian surface lookup
 -> silhouette and point-to-surface reprojection checks
```

It must not rank candidates, use SPair keypoint GT, run PartField over a
dataset, or train. It advances to the 18-panel/strict residual teacher
capacity only if the measured mapping is valid and the non-GT mask route is
documented. The SAM 3D Objects model is under the project's SAM License and
its official setup advertises a substantial GPU requirement; the one-image
probe is a cost gate, not evidence that a full test-set pass is affordable.

### TAP: verified carrier interface, not a verified identity source

The official TAP repository and `image_decoder.py` establish that its point,
box and sketch prompts pass through a full-image encoder and a two-way image
decoder. For every prompt it exports `sem_embeds`, a 1024-D vector for each
of its mask tokens. The official ViT-B/L/H weights are released under Apache
2.0. Thus TAP satisfies the *potential student carrier* interface:
`image + marked point -> region-conditioned vector`.

This does **not** establish a shared cross-instance part-role space,
canonical viewpoint handling, or front/rear and repeated-part identity.
Raw TAP remains disqualified as a frozen identity branch. It is relevant only
if a separate teacher passes the preceding capacity audit, and even then its
own synthetic transfer gate must be passed before any SPair panel is run.

### Fancy123: single-instance reconstruction, not a candidate-identity source

`Fancy123: One Image to High-Quality 3D Mesh Generation via Plug-and-Play
Deformation` (CVPR 2025) is now publicly implemented in
`YuQiao0303/Fancy123`. Its result is useful to record because it is an
open, training-free, single-image mesh generator, but that property is not
the identity interface required by this project.

The primary paper and the released code establish the following facts:

- `run_init.py` takes one RGB image (or a directory), optionally removes its
  background with `rembg`, generates six synthetic views through Zero123++ /
  InstantMesh, and writes an instance-local mesh. The released refinement
  program writes `final_mesh*.obj`.
- The refinement estimates an input camera independently for each image by a
  coarse-to-fine elevation search and a 100-iteration LPIPS optimization,
  then uses a per-mesh Jacobian deformation to make that mesh render like
  that same input image. It also projects that input image back onto the
  mesh. These operations create a usable *within-instance* rasterization
  path, but the public output contract is a mesh, not pixel-to-surface
  correspondences or a candidate vector.
- Each input is therefore reconstructed in its own generated-multiview frame.
  In particular, InstantMesh uses relative azimuth and Fancy123 assigns the
  input azimuth to zero while estimating elevation. This is input-view
  alignment, not a common object-front / left-right / front-rear canonical
  coordinate across two instances.
- The authors explicitly state a relevant limitation: the RGBA-only 3D
  deformation can create artifacts when different semantic parts have similar
  colors. Repeated wheels, legs, symmetric endpoints, and similarly coloured
  parts are exactly the residual cases in which a new identity source is
  required.
- The official setup is a separate CUDA 11.8 / PyTorch 2.1 environment with
  Kaolin, xFormers, TorchSparse/Cluster/Scatter, TensorRT optional, InstantMesh
  and Unique3D dependencies. It is not a lightweight plug-in for the current
  DiTF environment. The paper reports about 62 seconds per object on an A100
  (6 s multiview diffusion, 4 s LRM, 27 s geometry refinement, 10 s 2D
  deformation, 15 s 3D deformation), before local setup and I/O overhead.

| Qualification requirement | Fancy123 evidence | Status for the SPair identity audit |
|---|---|---|
| Deployment input | RGB input; its released preprocessing can use category-free `rembg`, but it has no documented object-mask argument and its natural-scene object-isolation behavior is unverified | Conditional only |
| Candidate identity output | Final textured `.obj`; no released per-pixel surface id, canonical coordinate, per-surface part descriptor, or source-to-candidate similarity | **Fail** |
| Shared cross-instance comparison space | Two independently generated meshes; input-view pose fitting only, no cross-instance surface map or object-yaw canonicalization | **Fail** |
| Object context | Full single-object reconstruction uses the whole image/object | Pass |
| Viewpoint, visibility, symmetric ambiguity | Per-image camera fit and smooth deformation improve image fidelity, but do not identify which repeated part is which across instances | **Fail** |
| Innovation boundary | Fancy123 alone is a reconstruction-quality pipeline. Adding an external canonical part field and correspondence adapter would re-enter the already-published 3D-SC family | No independent correspondence mechanism |

**Decision: close Fancy123 as a standalone frozen identity teacher before any
18-pair smoke.** This is a strict interface failure, not a negative claim
about mesh quality. A source pixel and a target candidate can each be
ray-cast to their own Fancy123 mesh only after adding unexported pose/raster
logic. Even if that engineering were done, comparing the two local XYZ,
normals, curvature, or depth values would be another unary/local geometric
score. It would not manufacture cross-instance part identity, and would
repeat the already closed geometry-scalar route under a new name.

The apparent compute cost does not justify bypassing the failure: an 18-pair
panel contains 36 images, so the paper's ideal 62-second figure is already
about 37 minutes of A100 inference plus a separate dependency setup; a
discovery/heldout pair20 pass would be 720 image reconstructions, about 12.4
ideal GPU-hours. Neither run answers the required comparison question without
a separate shared representation.

Fancy123 may be reconsidered only as an interchangeable front end *after* a
separately qualified canonical identity descriptor supplies all of the
following: an image-pixel-to-surface mapping, a documented shared
cross-instance part representation, a category-free object-mask route, and
an independently validated canonicalization. At that point Fancy123 must be
compared against the lower-cost available reconstruction front end on the
same frozen teacher audit; it is not entitled to a SPair experiment merely
because it emits a mesh.

### Pixal3D: conditional front-end qualification for a PartField-only audit

`Pixal3D: Pixel-Aligned 3D Generation from Images` (SIGGRAPH 2026,
arXiv:2605.10922) is materially different from Fancy123 for the narrow
purpose of an image-to-surface *front end*. It must not be called an identity
source: the possible cross-instance identity information would still come
only from frozen PartField features. The question here is narrower: can an
SPair pixel be mapped, without inventing a camera or nearest-face rule, to
the visible face of the exact generated mesh that is given to PartField?

The primary paper and the current `TencentARC/Pixal3D` `master` code establish
the following static facts:

- The method represents objects in the input-camera coordinate frame and
  conditions its 3D volume by explicitly back-projecting image features along
  input pixel rays. This is an image-to-surface interface claim, not a
  cross-instance semantic-correspondence claim.
- The released `inference.py` preprocesses the input, uses MoGe-2 to infer
  the preprocessed image FOV by default, computes `distance`, and passes
  `camera_angle_x`, `distance`, and `mesh_scale` to the generation pipeline.
  The paper's earlier fixed-FOV inference description is therefore not the
  current released inference contract; the current code is authoritative for
  an experiment.
- `proj_camera_to_render_params(camera_angle_x, distance)` returns the
  renderer's OpenCV extrinsics and normalized intrinsics. The companion
  `render_proj_aligned_video` says its first frame matches the projected
  input image, but its docstring attributes this to empirical testing. That
  statement remains a runtime hypothesis until an overlay regression passes.
- `MeshRenderer.render(..., return_types=[...])` supports `mask`, `depth`,
  `coord`, `normal`, and `attr`. `coord` is raster interpolation of the
  original mesh vertices, and the nvdiffrast coverage channel is
  `rast[..., -1]`. In the released PBR path, `mesh.material_ids` is indexed
  with `(tri_id - 1)`, establishing a concrete one-based foreground triangle
  convention in that code path.
- `run_inference` holds the raw `mesh = mesh_list[0]` before calling
  `o_voxel.postprocess.to_glb(..., remesh=True)`. The final GLB is therefore
  not safe for face-indexed features. A probe must export the pre-remesh
  internal triangle mesh for PartField and render that same topology.
- PartField exposes both face sampling and vertex sampling. Its released
  `correspondence_demo.yaml` selects `vertex_feature: True`, producing one
  448-D row per mesh vertex. This is the qualified low-cost path here: retain
  the raw Pixal3D topology with `preprocess_mesh=False`, rasterize the raw
  triangle id and barycentric weights, and interpolate those vertex rows at
  each visible pixel. The face-sampling demo defaults to 1000 random samples
  per face and is not appropriate for a high-face-count interface probe.
  PartField explicitly includes a demo on Trellis-generated meshes, which is
  evidence that generated meshes are within its intended input domain. It is
  not evidence that its features transfer across SPair viewpoints.

| Qualification requirement | Pixal3D + PartField status | Decision |
|---|---|---|
| Category-free image input | Pixal3D accepts an image and uses its own foreground removal when alpha is absent; neither its public image call nor PartField requires SPair category/keypoint labels | Conditional pass; inspect failures on real SPair objects |
| Original SPair pixel to model pixel | `preprocess_image` resizes, obtains/uses alpha, crops a 1.1x square foreground box, and composites it. It returns only the image, not scale/bbox metadata | Conditional pass; instrument existing preprocessing metadata and test round trip |
| Projection camera | Explicit current-code contract through MoGe FOV and `proj_camera_to_render_params` | Static pass; visual overlay still required |
| Pixel to front-most surface | Official nvdiffrast rasterization supplies mask/depth/coordinate and an internal triangle index | Static pass; expose triangle id without antialiasing and verify it |
| Stable surface to PartField row | Possible only with the internal pre-remesh mesh, `preprocess_mesh=False`, `num_features == num_vertices`, and the renderer's triangle id plus verified barycentric weights | Conditional pass; strict row-count and interpolation assertions are mandatory |
| Cross-instance part identity | Pixal3D supplies none. PartField reports emergent cross-shape consistency, but it independently centers/scales each mesh and supplies no yaw canonicalization | **Unknown: the only question for the later frozen capacity audit** |
| Symmetry / view change | No canonical yaw or left/right correspondence is supplied by the combined interface | Unresolved; do not make a gain claim |

**Static decision: conditional PASS only for a one-to-three-image interface
regression. Do not yet run an 18-pair PCK panel.** The accepted chain is

```text
SPair pixel -> recorded Pixal3D preprocessing transform -> projected renderer
-> visible raw-mesh triangle id + barycentrics -> same-topology interpolated
PartField vertex representation (448-D)
```

No cross-image Pixal XYZ, depth, normal, pose score, FLUX score, RoMa score,
router, training, or learned fusion is permitted in the first capacity test.
Those additions would hide whether raw PartField provides new identity
information and would repeat closed scalar/geometry routes.

The interface regression has four required invariants on one to three images:

1. original-to-preprocessed-to-original point round-trip from the exact resize
   and crop metadata;
2. projected first-view mask overlay against the exact preprocessed image,
   checked for mirroring, rotation and translation errors;
3. foreground pixels map to `mask=1`, finite depth/coordinates and a valid
   discrete triangle, while background pixels map to no surface; and
4. pre-remesh `num_vertices == partfield_features.shape[0]`, with rasterized
   coordinates proving the recovered barycentric weights interpolate the
   three vertices of `faces[tri_id - 1]` in the original face order.

Failure of any invariant closes this specific front end before PCK. Passing
all four permits only the pre-registered `raw PartField cosine over the
existing attention top-20` smoke: no representation composition. The later
strict-residual gate remains top-1 >= 45%, with explicit viewpoint-2 and
articulated strata. A future student/distillation decision additionally
requires a current-plus-teacher oracle union at least about 77%; a score below
that has insufficient headroom for a credible path from 69.527 to 75.

### Pixal3D cost and storage gate

The public `TencentARC/Pixal3D` model repository is **24 GB**. Its released
inference function instantiates all four 1.3B flow stages even in
`--low_vram` mode; low-VRAM mode lowers peak VRAM (the README reports about
10--12 GB versus about 18 GB) but does not reduce checkpoint storage.
The released PartField Objaverse checkpoint is 1.24 GB. MoGe-2, DINOv3/NAF
dependencies, the two environments, Hugging Face/Xet cache overhead, probe
outputs and the raw meshes require additional storage.

The previously reported 11 GB free on `/root/autodl-tmp` is therefore a hard
no-go. Allocate at least **45 GB free on the partition used by model caches**
before downloading. This is a storage requirement, not a request to run a
large experiment: the first permitted GPU work remains the one-to-three-image
regression. Do not use the final remeshed GLB or attempt a partial Pixal3D
checkpoint download; the released inference path requires the complete
pipeline.

### Existing dense functional correspondence model: unqualified pending code

`Weakly-Supervised Learning of Dense Functional Correspondences` is relevant
literature, but no public pretrained, image-only point embedding interface
has been verified in this review. Its official paper page could not be read
from the available network path and a single repository search was rate
limited. This is an availability status, not negative evidence about the
paper. It must remain outside every experiment until an official repository
or release documents weights, inference input, dense output tensor and a
cross-instance comparison rule.

## Smoke protocol after a source passes interface qualification

The smoke set is adversarially selected from the current residual atlas, not
randomly sampled. It contains exactly 18 pairs/panels: 3 rigid front/rear
endpoints, 3 rigid repeated-wheel/leg-like parts, 3 articulated legs/hooves,
3 head/tail ambiguities, 3 viewpoint-level-2 cases, and 3 current-correct
controls. The selection manifest must freeze pair ids, keypoint indices and
failure-mode labels before running the new source.

For each panel, the only question is whether the frozen representation ranks
at least one PCK-valid member of the existing FLUX attention top-20 above the
visible wrong candidates because of a source-to-candidate part-role identity
signal. GT is evaluation-only and merely identifies PCK-valid candidates.

Prohibited at this stage: FLUX/RoMa/DINO scores, any router, training,
prompt variants, score fusion, threshold selection, or representation
composition. Absence of obvious qualitative separation closes the source;
it does not justify more smoke panels or parameter searches.

## Pre-registered first test if, and only if, a usable PartField interface exists

The first experiment must be an **offline capacity audit**, not a matcher,
adapter, pseudo-label generator, score router, or full-SPair run.

Fixed inputs:

- existing FLUX attention top-20 coordinates;
- the existing discovery20 / heldout20 strict current-error, top-20-hit
  joins;
- the same PCK-valid-candidate definition used in the residual atlas;
- no DINO/SD/FLUX/RoMa score fusion during the source-capacity measurement.

Required outputs per candidate:

- a frozen source-to-target canonical part similarity or descriptor distance;
- validity/visibility for the projected surface point;
- a source-specific and target-specific canonical coordinate or equivalent
  object-relative descriptor; and
- a documented label-free uncertainty signal.

Advance gate:

1. heldout strict residual top-1 at least 45%, substantially above RoMa's
   34.29% and the GT-only RoMa relation result of 35.31%;
2. positive results on articulated and viewpoint-2 strata, not only rigid
   categories; and
3. a current-correct control demonstrating an observable intervention signal
   better than direct replacement's 73--75% retention.

If this capacity test fails, close the particular canonical source without
training or representation-composition design. If it passes, the next design
question is narrowly scoped: how to compose frozen FLUX appearance with the
canonical identity descriptor while retaining the candidate axis and leaving
native successes unchanged. That is distinct from reproducing 3D-SC's
pseudo-label-plus-adapter pipeline.

## Sources checked

- https://arxiv.org/html/2605.30093
- https://github.com/GenIntel/3D-SC
- https://github.com/NVlabs/describe-anything
- https://arxiv.org/html/2411.16185v2
- https://github.com/YuQiao0303/Fancy123
- https://arxiv.org/html/2605.10922
- https://github.com/TencentARC/Pixal3D
- https://huggingface.co/TencentARC/Pixal3D
- https://github.com/nv-tlabs/PartField
- https://huggingface.co/mikaelaangel/partfield-ckpt
- `RESEARCH_CURRENT_PLUS2_RESIDUAL_ATLAS_2026-08-06.md`
