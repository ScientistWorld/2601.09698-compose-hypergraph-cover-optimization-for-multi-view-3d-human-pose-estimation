# Overview

- **Paper ID:** 2601.09698
- **Title:** COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation
- **Domain:** Computer Vision / Multi-view 3D Human Pose Estimation
- **TL;DR:** COMPOSE replaces pairwise multi-view correspondence with a geometrically pruned hypergraph exact-cover formulation for robust 3D multi-person pose recovery.

## Short Summary

The paper addresses 3D multi-person pose estimation from sparse calibrated
camera views. It treats candidate cross-view associations as hyperedges,
scores them by triangulation reprojection error, prunes implausible candidates,
and solves a weighted exact-cover ILP before triangulating the final 3D poses.
On CMU Panoptic, COMPOSE improves strict AP and MPJPE over the optimization
baseline MvPose while remaining training-free.

## Key Results

- CMU Panoptic: AP@25 improves from 37.63 for MvPose to 54.66 for COMPOSE.
- CMU Panoptic: MPJPE improves from 26.46 mm for MvPose to 23.62 mm.
- Campus: COMPOSE reports 97.3% average PCP, above the reproduced MvPose result of 95.8%.
