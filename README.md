# COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening).

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2601.09698"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2601.09698-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2601.09698](https://arxiv.org/abs/2601.09698) |
|:---|:---|
| **Domain** | Computer Vision / Multi-view 3D Human Pose Estimation |
| **TL;DR** | COMPOSE replaces pairwise multi-view correspondence with a geometrically pruned hypergraph exact-cover formulation for robust 3D multi-person pose recovery. |

## Summary

The paper addresses 3D multi-person pose estimation from sparse calibrated
camera views. It treats candidate cross-view associations as hyperedges,
scores them by triangulation reprojection error, prunes implausible candidates,
and solves a weighted exact-cover ILP before triangulating the final 3D poses.
On CMU Panoptic, COMPOSE improves strict AP and MPJPE over the optimization
baseline MvPose while remaining training-free.

## Problem

How can a system recover accurate 3D poses for multiple people from a small set
of synchronized calibrated RGB cameras when each view supplies noisy,
occluded, and ambiguous 2D human pose detections?

## The paper's method

COMPOSE formulates multi-view 2D pose correspondence as a hypergraph
partitioning problem. Candidate hyperedges represent sets of detections across
multiple camera views that could belong to one 3D person.

## Key results of the paper

- CMU Panoptic: AP@25 improves from 37.63 for MvPose to 54.66 for COMPOSE.
- CMU Panoptic: MPJPE improves from 26.46 mm for MvPose to 23.62 mm.
- Campus: COMPOSE reports 97.3% average PCP, above the reproduced MvPose result of 95.8%.

## Run it

```bash
pip install sciworld
sw research 2601.09698 --mode innovate --agent claude-code
sw evaluate 2601.09698
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

