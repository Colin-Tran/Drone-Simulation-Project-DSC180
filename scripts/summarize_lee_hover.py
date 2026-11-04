"""Summarize pre-action samples from test_lee_hover.py; no simulator required."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.logs:
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        metadata = entries[0]["metadata"]
        rows = entries[1:]
        target = metadata["target"][:3]
        errors = [sum((a - b) ** 2 for a, b in zip(r["position_w"], target, strict=True)) for r in rows]
        tilts = [
            math.degrees(
                math.acos(max(-1, min(1, 1 - 2 * (r["quaternion_wxyz"][1] ** 2 + r["quaternion_wxyz"][2] ** 2))))
            )
            for r in rows
        ]
        summary = {
            "file": str(path),
            "samples": len(rows),
            "control_hz": 1 / (metadata["physics_dt"] * metadata["decimation"]),
            "position_rmse_m": math.sqrt(sum(errors) / len(rows)),
            "altitude_range_m": [min(r["position_w"][2] for r in rows), max(r["position_w"][2] for r in rows)],
            "max_xy_error_m": max(math.dist(r["position_w"][:2], target[:2]) for r in rows),
            "max_tilt_deg": max(tilts),
            "saturated_action_fraction": sum(any(r["saturated"]) for r in rows) / len(rows),
            "last_pre_action_position": rows[-1]["position_w"],
            "initial_thrust_total_N": sum(metadata["initial_thrust"]),
            "terminated": rows[-1]["terminated"],
            "truncated": rows[-1]["truncated"],
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
