# `grounding_dino` — GD proposal source (Stage2 §3.6)

GD as a **second, independent proposer**. The mechanism is self-contained here and
**not wired into the policy yet**; nothing in this package imports the policy, and
nothing writes memory or picks actions.

## Why this shape (measured)

| Fact | Evidence |
| :--- | :--- |
| "GD box -> TSP3D ROI re-query" is empty | C4: 35 episodes bit-identical to the anchor; B2 (`gdRcSrc`): **0** `[GD] ROI re-query` hits in 63 episodes, the per-episode budget was never even consumed |
| Unthresholded GD-only admission floods memory | `gdRcSrc` 3979 admissions / 63 episodes, SR `30/61 -> 15/61` (monotonically worse) |
| TSP3D cannot take SAM's slot | TSP3D input is `point_clouds + text` only; SAM's slot is 2D box -> 2D mask, and VLVM has no SAM in the loop at all |

So: GD proposes, geometry lands (2D box -> depth -> cluster), TSP3D only votes.

## Frame alignment

* GD reads the frame the caller just fed to the fusion chain (`rgb`, `depth_m`,
  `CameraView`); `every_n` guards the rate.
* The pre-TSP3D fusion chain is untouched, and the GD point set is **never**
  injected into it or sent to TSP3D as a re-query (`region.py` only builds a
  candidate).
* Consequence: no extra TSP3D forward; `_n_infer` on the TSP3D server is unchanged.

## Vote model (`provenance.py`)

```
frame candidates = TSP3D boxes (as usual)  ∪  GD proposals (this package)

GD proposal co-locates with an ADMITTED TSP3D candidate  -> BOTH   (strong)
otherwise                                                -> GD     (single vote)
TSP3D alone                                              -> TSP3D  (unchanged)
```

`same_object` = XY centroid distance <= `merge_dist` **or** AABB `IaU >= merge_iau`
**or** containment >= 0.8 — evaluated against **admitted** TSP3D candidates only, so
a box rejected by the admission gates cannot make a GD proposal look strong.

Accuracy-first defaults: `admit_as_suspicious(BOTH)=False`, and `is_lockable(...)`
refuses to lock a `GD` entry seen only once (`weak_guard=True`) — single-vote
entries live as pending/soft and are reaped by the existing suspicious near-field
hysteresis / soft-only free-space erasure.

## Module map

| File | Role |
| :--- | :--- |
| `config.py` | `GdProposalConfig` — all knobs (rate, threshold, chain, merge, weak guard) |
| `types.py` | `CameraView`, `GdProposal`, `Tsp3dCandidate`, `ProposalStats`, `MatchResult` |
| `vocab.py` | class-level text alignment: normalize / aliases / strict single-class match / captions |
| `client.py` | `GdDetector` — guarded server access (failure = "not checked", never negative) |
| `region.py` | 2D box -> world cluster: the VLFM `_extract_object_cloud` chain |
| `boxes.py` | AABB corners / centre / IaU / containment / nearest point |
| `proposer.py` | `GdProposer` — frame -> `List[GdProposal]` + counters |
| `provenance.py` | vote merge + admission / lock rules (pure functions) |

## VLFM mapping (照搬 / 适配 / 新增)

| VLFM | Here | Class |
| :--- | :--- | :--- |
| `GroundingDINOClient.predict(rgb, caption)` + `filter_by_class` | `client.GdDetector` + `vocab.match_phrase` | 照搬 + 适配（严格单类） |
| `coco_threshold 0.8 / non_coco_threshold 0.4` | `box_thr = 0.4` (single GD detector -> the non-COCO value; 0.8 belongs to the YOLO COCO detector) | 适配 |
| `MobileSAM.segment_bbox` -> mask | in-box rectangle (`region.rect_mask`) | 适配（角色替代） |
| `_extract_object_cloud`: erode -> holes to far -> backproject -> random 5000 -> DBSCAN largest cluster | `region.region_points_world` | **照搬** |
| `too_offset` -> random `range_id` | `region.too_offset_rect` -> `GdProposal.suspect` (caller writes it as suspicious) | 适配 |
| `update_explored` (near re-look deletes flagged ids) | existing suspicious near-field hysteresis / soft-only free-space erasure | 适配 |
| `dist < 1.0 m` reject | `min_dist` gate on the closest cluster point to the camera | **照搬** |
| no cross-frame dedup (point concat + closest point) | `provenance.attach_votes` (frame) + existing EMA memory merge (cross-frame) | **新增** |
| value map untouched by GD boxes; no visualisation / VQA / dual-detector retry | same (value map is frozen by the project rules) | 一致 |

## Usage (policy wiring comes later)

```python
from vlfm.tsp3d_models.grounding_dino import GdProposalConfig, GdProposer, CameraView
from vlfm.tsp3d_models.grounding_dino import Tsp3dCandidate, attach_votes, admit_as_suspicious

proposer = GdProposer(GdProposalConfig(box_thr=0.4, every_n=1))
if should_run(step, proposer.cfg.every_n):
    cam = CameraView(fx, fy, H, W, tf_camera_to_episodic, min_depth, max_depth)
    rng = np.random.RandomState((det_seed * 100003 + step) & 0x7FFFFFFF)
    proposals = proposer.propose(rgb, depth_m, cam, target_classes, robot_xy, rng)
    for prop, vote, matched_idx in attach_votes(proposals, tsp3d_candidates, proposer.cfg):
        ...  # vote == BOTH -> corroborate the matched entry; GD -> suspicious pending write
```

Diagnostics: `proposer.summary()` gives `{frames, boxes_above_thr, proposals, rejects}`
(rejection keys: `below_thr` / `other_class` / `tiny_box` / `eroded_empty` /
`no_cluster` / `too_close` / `no_gd`).
