"""Freeze normal-drive roles before fitting or evaluating anomaly scores.

Requires completed recordings without infractions except minimum-speed warnings.
Repeated recordings of one route stay together. These are in-domain reruns,
not evidence of unseen-town or unseen-road generalization.
"""

import argparse
import datetime
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ("validation", "calibration", "drive_reference", "test")


def partition(groups, seed):
    """Balance route groups across roles within scenarios, then globally."""
    rng = random.Random(seed)
    scenario_groups = defaultdict(list)
    for group, rows in sorted(groups.items()):
        scenarios = {row["scenario"] for row in rows}
        if len(scenarios) != 1:
            raise ValueError("One route group has multiple scenario labels")
        scenario_groups[rows[0]["scenario"]].append(group)
    counts = Counter()
    assignments = {}
    for _scenario, group_ids in sorted(scenario_groups.items()):
        rng.shuffle(group_ids)
        local = Counter()
        for group in group_ids:
            tie_order = list(ROLES)
            rng.shuffle(tie_order)
            role = min(tie_order, key=lambda key: (local[key], counts[key]))
            assignments[group] = role
            local[role] += 1
            counts[role] += len(groups[group])
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.data_root.resolve()
    training = args.training_data.resolve()
    groups, excluded = defaultdict(list), []
    for drive in sorted((root / "data").glob("*/*")):
        if not drive.is_dir():
            continue
        drive = drive.resolve()
        if drive.is_relative_to(training):
            raise ValueError(f"Recording overlaps training data: {drive}")
        result_path = drive / "results.json"
        reasons = []
        if not result_path.exists() or not (drive / "metas/0000.pkl").exists():
            reasons.append("Missing results or initial metadata")
        else:
            result = json.loads(result_path.read_text())
            if result["status"] not in ("Completed", "Perfect"):
                reasons.append(result["status"])
            reasons.extend(
                f"{key}: {len(value)}"
                for key, value in result["infractions"].items()
                if key != "min_speed_infractions" and value
            )
        if reasons:
            excluded.append({"drive": str(drive), "reasons": reasons})
            continue
        group = (
            result["town_name"]
            + ":"
            + re.sub(r"_rep\d+$", "", result["route_id"], flags=re.IGNORECASE)
        )
        groups[group].append(
            {
                "drive": str(drive),
                "route_group": group,
                "scenario": drive.parent.name,
                "town": result["town_name"],
                "recorded_frames": len(list((drive / "metas").glob("*.pkl"))),
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            },
        )
    assignments = partition(groups, args.seed)
    roles = {role: [] for role in ROLES}
    inventory = {}
    for group, rows in groups.items():
        for row in rows:
            roles[assignments[group]].append(row["drive"])
            inventory[row["drive"]] = row
    if any(not rows for rows in roles.values()):
        raise ValueError("Not enough clean route groups for four nonempty roles")
    output = {
        "schema": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": args.seed,
        "data_root": str(root),
        "training_data": str(training),
        "roles": {role: sorted(rows) for role, rows in roles.items()},
        "inventory": inventory,
        "excluded": excluded,
        "normal_criteria": "Completed/Perfect, initial metadata present, no recorded infractions except minimum speed warnings.",
        "limitations": [
            "Recording paths are disjoint from training; physical-road overlap is not excluded.",
            "These expert drives were previously used in exploratory Stage-1 analyses; they are not a pristine benchmark.",
            "No recorded infraction does not establish that every action is normative.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "drives": len(inventory),
                "route_groups": len(groups),
                "excluded": len(excluded),
                "roles": {key: len(rows) for key, rows in roles.items()},
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
