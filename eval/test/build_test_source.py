#!/usr/bin/env python3
"""Build ``eval/test/test_source.json`` -- the hidden test split's 3D answers.

Provenance (public CMU Panoptic release, standard multi-view pose *test*
sequences, standard five HD cameras 3, 6, 12, 13, 23):

    for seq in 160906_pizza1 160422_haggling1 160906_ian5 160906_band4; do
      mkdir -p eval/test/downloads/$seq
      curl -L --fail -o eval/test/downloads/$seq/calibration_$seq.json \
        http://domedb.perception.cs.cmu.edu/webdata/dataset/$seq/calibration_$seq.json
      curl -L --fail -o eval/test/downloads/$seq/hdPose3d_stage1_coco19.tar \
        http://domedb.perception.cs.cmu.edu/webdata/dataset/$seq/hdPose3d_stage1_coco19.tar
    done
    python eval/test/build_test_source.py

The committed output keeps, per sequence, the five cameras and 100 evenly
spaced frames with at least two annotated people (15 body joints, metres, and
the per-joint annotation-validity mask).  Poses are stored as base64 float32 so
the file is compact.  The raw downloads are not needed at test time.
"""

from __future__ import annotations

import argparse
import base64
import json
from array import array
from pathlib import Path
import tarfile
from typing import Any

TEST_SEQUENCES = ("160906_pizza1", "160422_haggling1", "160906_ian5", "160906_band4")
CAMERA_NODES = (3, 6, 12, 13, 23)
NUM_JOINTS = 15
ROOT = 2
FRAMES_PER_SEQUENCE = 100


def load_cameras(path: Path) -> list[dict[str, Any]]:
    calib = json.loads(path.read_text(encoding="utf-8"))
    by_node = {int(cam["node"]): cam for cam in calib["cameras"] if cam.get("type") == "hd"}
    out = []
    for node in CAMERA_NODES:
        cam = by_node[node]
        out.append(
            {
                "camera_id": f"hd_00_{node:02d}",
                "image_size": [int(cam["resolution"][0]), int(cam["resolution"][1])],
                "K": [[float(v) for v in row] for row in cam["K"]],
                "R": [[float(v) for v in row] for row in cam["R"]],
                "t": [float(row[0]) / 100.0 for row in cam["t"]],
            }
        )
    return out


def parse_body(body: dict[str, Any]) -> tuple[list[float], int] | None:
    flat = [float(v) for v in body["joints19"]]
    coords: list[float] = []
    mask = 0
    count = 0
    for j in range(NUM_JOINTS):
        x, y, z, c = flat[4 * j : 4 * j + 4]
        coords.extend([x / 100.0, y / 100.0, z / 100.0])
        if c > 0.1:
            mask |= 1 << j
            count += 1
    if not mask & (1 << ROOT) or count < 8:
        return None
    return coords, mask


def load_frames(tar_path: Path) -> list[list[tuple[list[float], int]]]:
    frames = []
    with tarfile.open(tar_path) as archive:
        members = sorted(
            (m for m in archive.getmembers() if m.isfile() and m.name.endswith(".json")), key=lambda m: m.name
        )
        for member in members:
            try:
                payload = json.load(archive.extractfile(member))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # a few released frame files are empty
            bodies = [b for b in (parse_body(body) for body in payload.get("bodies", [])) if b is not None]
            if len(bodies) >= 2:
                frames.append(bodies)
    return frames


def evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(step * k + step / 2)] for k in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="eval/test/downloads")
    parser.add_argument("--output", default="eval/test/test_source.json")
    args = parser.parse_args()
    sequences = []
    for seq in TEST_SEQUENCES:
        seq_dir = Path(args.root) / seq
        cameras = load_cameras(seq_dir / f"calibration_{seq}.json")
        frames = evenly_spaced(load_frames(seq_dir / "hdPose3d_stage1_coco19.tar"), FRAMES_PER_SEQUENCE)
        encoded = []
        for bodies in frames:
            values = array("f", [v for coords, _ in bodies for v in coords])
            encoded.append(
                {"n": len(bodies), "m": [mask for _, mask in bodies], "p": base64.b64encode(values.tobytes()).decode()}
            )
        sequences.append({"cameras": cameras, "frames": encoded})
    payload = {"format": "compose_gym_hidden_source_v1", "num_joints": NUM_JOINTS, "sequences": sequences}
    Path(args.output).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {args.output}: {sum(len(s['frames']) for s in sequences)} frames")


if __name__ == "__main__":
    main()
