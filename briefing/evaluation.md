# Evaluation

## Data

Real multi-person scenes from the CMU Panoptic studio, observed by the five
calibrated 1920x1080 HD cameras of the standard multi-view protocol (nodes 3,
6, 12, 13, 23).  3D answers are the released Panoptic body annotations
(15 body joints: neck, nose, mid-hip, and left/right shoulder, elbow, wrist,
hip, knee, ankle; metres, Panoptic world frame, y pointing down).

Each view supplies 2D pose detections from a stochastic detector model applied
to the real annotations and cameras (the benchmark does not ship videos or an
image-based detector).  Detections have 15 keypoints `[u, v, confidence]` and a
detection `score`, and reproduce the usual failure modes: size-dependent and
heavy-tailed joint noise, inter-person occlusion (noisier joints, joints
attached to the occluding person, people dropped when mostly hidden),
truncation at image borders, whole-body left/right flips, duplicate partial
detections and person-shaped false positives.  Each camera has its own error
profile.  Detections carry no identity information.  People in each frame are
rotated about the vertical axis by a random angle.

* **Train split (public):** `data/downloads/panoptic_train.json`, 300 frames /
  1022 people from five sequences of the standard Panoptic training split,
  built by `scripts/download.sh`.  It includes the 3D answers.
* **Test split (hidden):** 400 frames / 1367 people from the four sequences of
  the standard Panoptic test split, built inside the hidden evaluator at test
  time with its own detector-noise and rotation draws.  Test inputs and answers
  are never on disk outside the evaluator.

## Submitting a method

A method is a directory `checkpoints/<name>/` with an entry point `predict.sh`:

    bash checkpoints/<name>/predict.sh INPUTS.json OUTPUT.json

It is run from the workspace root, without network, with a 30-minute limit per
split (1 CPU in the standard job).  It must work on any frames it is given and
must not read anything but `INPUTS.json`, its own code/configuration and public
data.  Any tuning or training happens beforehand, on the train split.
`INPUTS.json` (see `eval/harness.py`):

    {"joint_names": [...15], "units": "meters",
     "frames": [{"frame_id": "...",
                 "cameras": [{"camera_id", "image_size", "K", "R", "t"}],
                 "views": [{"camera_id": "...",
                            "detections": [{"keypoints": [[u, v, conf] x 15], "score": s}]}]}]}

with `x_cam = R @ X + t` and `[u, v, 1] ~ K @ x_cam`.  `OUTPUT.json`:

    {"frames": {frame_id: {"predicted_poses": [{"joints_3d": [[x, y, z] x 15], "score": s}]}}}

## Validity

A run is **failed** — and receives the worst value of every metric (AP 0,
recall 0, MPJPE 500 mm) — when the predictor crashes, times out, writes no or
unparsable output, omits an input frame, emits more poses in a frame than that
frame has 2D detections, or emits a pose without exactly 15 joints of 3 finite
coordinates (magnitude at most 10^6 m) or without a finite score.  A joint that
is finite but far from the person is not invalid — it is simply a large error.
Nothing else the predictor writes is read.

## Metrics (paper protocol)

Every predicted pose is compared with each annotated person of its frame by
MPJPE over that person's annotated joints and assigned to the closest one;
predictions are ranked by `score` over the whole split.

- `ap_25mm`, `ap_50mm`, `ap_100mm`, `ap_150mm` (higher is better): average
  precision where a prediction is a true positive if its MPJPE to its closest
  person is below the threshold and that person was not already claimed by a
  higher-scored prediction.  Duplicates and spurious poses are false positives.
- `recall_500mm` (higher is better): share of people that are the closest
  person of some prediction within 500 mm.
- `mpjpe_mm` (lower is better): mean MPJPE of the first prediction claiming
  each person within 500 mm (500 when nothing matches).

## Running

    bash scripts/evaluate_train.sh   # every checkpoints/*/predict.sh on train -> scoring/scores_train.json
    bash scripts/evaluate.sh NAME    # one method on train -> scoring/scores.json
    bash scripts/evaluate_test.sh    # hidden test (evaluation job only) -> scoring/scores_test.json

All scored numbers are recomputed by `eval/` from the raw predicted poses.

## Baselines

- `pairwise_baseline`: pairwise two-view reprojection matching with union-find
  grouping.
- `mvpose_geometric`: the paper's optimisation-based competitor family —
  pairwise epipolar affinities made cycle-consistent by convex multi-way
  matching (geometric cues only).
