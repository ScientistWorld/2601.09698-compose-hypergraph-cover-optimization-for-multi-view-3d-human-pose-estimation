# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

COMPOSE formulates multi-view 2D pose correspondence as a hypergraph
partitioning problem.  Nodes are the 2D pose detections of all views; a
hyperedge is a set of detections from distinct views hypothesised to be one
person, so consistency is judged across all its views at once instead of by
reconciling pairwise matches.

## Approach

1. 2D poses are detected independently in each view.
2. Each candidate hyperedge is triangulated with closed-form DLT; its cost is
   the reprojection error summed over the hyperedge's views and averaged over
   the J joints, C(e) = 1/J * sum_j sum_{v in e} ||pi_v(y_j) - u_{v,j}||^2.
3. Geometric pruning: hyperedges with C(e) > tau are discarded, and hyperedges
   are only grown from surviving ones instead of instantiating every
   combination (the full set grows as prod_v (1 + n_v) - 1).
4. Score s(e) = exp(-lambda * C(e)); singleton hyperedges with a fixed prior
   score cover detections seen in one view only.
5. A weighted exact cover, max sum_e (s(e) - gamma) x_e subject to every
   detection being covered exactly once, is solved as an ILP (PuLP / CBC
   branch and bound); gamma penalises the number of selected hyperedges and
   so favours large hyperedges.
6. Selected multi-view hyperedges are triangulated into the final 3D poses.

Hyperparameters (tau, lambda, gamma, s_single) are tuned on non-test frames.
In this gym: `method/compose.py` (pipeline), `method/tune_compose.py`
(selection on the public train split), `method/compose_config.json` (selected
values), `scripts/method.sh` (registers `checkpoints/compose/predict.sh`).

## Main Claims

The primary claim is that higher-order multi-view association improves 3D pose
accuracy over pairwise optimization-based matching (MvPose), especially in
strict-threshold AP and MPJPE, without 3D supervision.  Secondary claims are
that geometric pruning makes the ILP tractable (about 23 ms per frame on
Panoptic) and that the method generalises to Shelf and Campus.
