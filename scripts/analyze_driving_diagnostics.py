"""Summarize a SensorAgent JSONL trace without importing CARLA or the policy.

Example: python -m scripts.analyze_driving_diagnostics --trace TRACE.jsonl
    --output analysis/turn --plot --step 420

Flags identify moments to inspect. Sparse navigation targets are not a desired
trajectory, and neither their bearing nor a predicted endpoint proves a wrong turn.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

FLAG_DESCRIPTIONS = {
    "sparse_target_bearing_gap": "Route lookahead and sparse target bearings differ by at least 60 degrees; their horizons and purpose differ.",
    "plan_heading_change": "Estimated navigation-frame bearing of the route lookahead changed at least 35 degrees within one second.",
    "localization_sources_apart": "Filtered and noisy observed XY estimates differ by at least one meter.",
    "controller_modified": "Executed steer/throttle/brake differs from the recorded controller output.",
    "lateral_controllers_apart": "Route and waypoint steering outputs differ by at least 0.3; the configured controller selection determines execution.",
    "navigation_command_transition": "Current or next navigation command changed from the preceding record.",
}
LIMITATIONS = [
    "Candidate flags and ranking locate evidence for review; they do not establish a wrong branch or a causal module.",
    "Sparse target-point bearings are navigation cues, not an exact desired trajectory; route and target horizons differ.",
    "Observed paths and projected predictions use GPS-derived filtered navigation XY plus the observed compass. This frame need not share the simulator world origin; these are estimates, not simulator ground truth.",
    "Model target inputs are projected at the filtered observed pose for comparison with the predicted route; their original position source is recorded separately in navigation.target_position_source.",
    "Observed distance is the sum between consecutive valid filtered XY records, not leaderboard route completion.",
    "Trace termination does not establish route success, failure, or evaluator completion; consult the evaluator result and run status separately.",
    "Records retain file order and original timestamps; missing records and a truncated final line are reported rather than interpolated.",
]


def clean_json(value, counts):
    if isinstance(value, float) and not math.isfinite(value):
        counts["nonfinite_values"] += 1
        return None
    if isinstance(value, list):
        return [clean_json(item, counts) for item in value]
    if isinstance(value, dict):
        return {key: clean_json(item, counts) for key, item in value.items()}
    return value


def read_trace(path: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Retain complete records around malformed lines, with their source locations."""
    metadata, steps, issues = [], [], []
    counts = Counter()
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            counts["lines"] += 1
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                kind = "invalid_json" if line.endswith("\n") else "truncated_tail"
                issues.append({"line": line_number, "kind": kind, "detail": str(exc)})
                continue
            record = clean_json(record, counts)
            if not isinstance(record, dict):
                issues.append({"line": line_number, "kind": "non_object_record"})
                continue
            record["source_line"] = line_number
            if record.get("record_type") == "metadata":
                metadata.append(record)
                if record.get("schema_version") != 1:
                    issues.append(
                        {
                            "line": line_number,
                            "kind": "unsupported_schema",
                            "version": record.get("schema_version"),
                        },
                    )
            elif record.get("record_type") == "step":
                steps.append(record)
            else:
                issues.append(
                    {
                        "line": line_number,
                        "kind": "unknown_record_type",
                        "record_type": record.get("record_type"),
                    },
                )
    if not metadata:
        issues.append({"kind": "missing_metadata"})
    if len(metadata) > 1:
        issues.append({"kind": "multiple_metadata_records", "count": len(metadata)})
    if not steps:
        issues.append({"kind": "no_step_records"})
    return metadata, steps, issues, dict(counts)


def section(record, name):
    value = record.get(name)
    return value if isinstance(value, dict) else {}


def scalar(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


def vector(value):
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, list) and not any(
        isinstance(item, list | dict) for item in value
    ):
        return [scalar(item) for item in value]
    return []


def xy(value):
    values = vector(value)
    if len(values) >= 2 and all(item is not None for item in values[:2]):
        return values[:2]
    return None


def points(value):
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if xy(value) is not None:
        return [xy(value)]
    if not isinstance(value, list):
        return []
    return [point for item in value if (point := xy(item)) is not None]


def angle_difference(first, second):
    if first is None or second is None:
        return None
    return (first - second + 180.0) % 360.0 - 180.0


def bearing(point):
    if point is None or math.hypot(*point) < 1e-6:
        return None
    return math.degrees(math.atan2(point[1], point[0]))


def route_lookahead(route, distance=10.0):
    """First predicted route point at least distance meters from ego, else last."""
    return next(
        (point for point in route if math.hypot(*point) >= distance),
        route[-1] if route else None,
    )


def project_world(local_points, position, compass):
    """CARLA ego XY (forward/right) to the GPS-derived navigation frame."""
    if position is None or compass is None:
        return []
    cosine, sine = math.cos(compass), math.sin(compass)
    return [
        [position[0] + cosine * x - sine * y, position[1] + sine * x + cosine * y]
        for x, y in local_points
    ]


def make_timeline(steps, issues):
    rows, previous, previous_position = [], None, None
    distance, seen_steps = 0.0, set()
    for record in steps:
        observed = section(record, "observed_state")
        navigation = section(record, "navigation")
        inputs = section(record, "model_inputs")
        prediction = section(record, "predictions")
        history = section(record, "history")
        actual_ages = [
            age
            for age in vector(history.get("actual_tick_ages"))
            if age is not None and age >= 0
        ]
        requested_ages = [
            age for age in vector(history.get("requested_tick_ages")) if age is not None
        ]
        padding = history.get("padding_mask")
        position, noisy = (
            xy(observed.get("filtered_state")),
            xy(observed.get("noisy_state")),
        )
        compass = scalar(observed.get("compass_radians"))
        target = xy(inputs.get("target_point"))
        next_target = xy(inputs.get("target_point_next"))
        route = points(prediction.get("pred_route"))
        lookahead = route_lookahead(route)
        step, timestamp = (
            scalar(record.get("step")),
            scalar(record.get("timestamp_seconds")),
        )
        if step is not None and step in seen_steps:
            issues.append(
                {"line": record["source_line"], "kind": "duplicate_step", "step": step},
            )
        if step is not None:
            seen_steps.add(step)
        if timestamp is None or step is None:
            issues.append(
                {"line": record["source_line"], "kind": "missing_step_or_timestamp"},
            )
        if position is not None and previous_position is not None:
            distance += math.dist(position, previous_position)
        previous_position = position
        route_bearing, target_bearing = bearing(lookahead), bearing(target)
        gap = angle_difference(route_bearing, target_bearing)
        localization_gap = (
            math.dist(position, noisy)
            if position is not None and noisy is not None
            else None
        )
        row = {
            "source_line": record["source_line"],
            "step": step,
            "timestamp_seconds": timestamp,
            "filtered_x_m": position[0] if position else None,
            "filtered_y_m": position[1] if position else None,
            "compass_degrees": math.degrees(compass) if compass is not None else None,
            "speed_mps": scalar(observed.get("speed_mps")),
            "observed_distance_m": distance,
            "command_id": scalar(navigation.get("command_id")),
            "next_command_id": scalar(navigation.get("next_command_id")),
            "target_position_source": navigation.get("target_position_source"),
            "planner_position_source": navigation.get("planner_position_source"),
            "pop_distance_m": scalar(navigation.get("pop_distance_m")),
            "target_x_m": target[0] if target else None,
            "target_y_m": target[1] if target else None,
            "target_distance_m": math.hypot(*target) if target else None,
            "target_bearing_degrees": target_bearing,
            "next_target_bearing_degrees": bearing(next_target),
            "route_lookahead_x_m": lookahead[0] if lookahead else None,
            "route_lookahead_y_m": lookahead[1] if lookahead else None,
            "route_lookahead_distance_m": math.hypot(*lookahead) if lookahead else None,
            "route_lookahead_bearing_degrees": route_bearing,
            "route_target_bearing_gap_degrees": abs(gap) if gap is not None else None,
            "route_lookahead_world_bearing_degrees": angle_difference(
                math.degrees(compass) + route_bearing, 0.0,
            )
            if compass is not None and route_bearing is not None
            else None,
            "localization_gap_m": localization_gap,
            "predicted_speed_mps": scalar(prediction.get("pred_target_speed_scalar")),
            "route_steer": scalar(prediction.get("route_steer")),
            "waypoints_steer": scalar(prediction.get("waypoints_steer")),
            "history_oldest_tick_age": max(actual_ages) if actual_ages else None,
            "history_newest_tick_age": min(actual_ages) if actual_ages else None,
            "history_requested_oldest_tick_age": max(requested_ages)
            if requested_ages
            else None,
            "history_requested_newest_tick_age": min(requested_ages)
            if requested_ages
            else None,
            "history_padding_count": sum(value is True for value in padding)
            if isinstance(padding, list)
            else None,
        }
        for name in ("initial_braking", "stuck_detector", "force_move"):
            row[name] = section(record, "interventions").get(name)
        for source in ("controller_control", "executed_control"):
            for control in ("steer", "throttle", "brake"):
                row[f"{source}_{control}"] = scalar(
                    section(record, source).get(control),
                )
        modifications = [
            abs(
                row[f"controller_control_{control}"]
                - row[f"executed_control_{control}"],
            )
            for control in ("steer", "throttle", "brake")
            if row[f"controller_control_{control}"] is not None
            and row[f"executed_control_{control}"] is not None
        ]
        row["max_control_modification"] = max(modifications) if modifications else None
        row["lateral_controller_gap"] = (
            abs(row["route_steer"] - row["waypoints_steer"])
            if row["route_steer"] is not None and row["waypoints_steer"] is not None
            else None
        )
        row["delta_timestamp_seconds"] = (
            timestamp - previous["timestamp_seconds"]
            if previous is not None
            and timestamp is not None
            and previous["timestamp_seconds"] is not None
            else None
        )
        row["route_world_bearing_change_degrees"] = (
            angle_difference(
                row["route_lookahead_world_bearing_degrees"],
                previous["route_lookahead_world_bearing_degrees"],
            )
            if previous is not None
            else None
        )
        if (
            row["delta_timestamp_seconds"] is not None
            and row["delta_timestamp_seconds"] <= 0
        ):
            issues.append(
                {
                    "line": record["source_line"],
                    "kind": "nonincreasing_timestamp",
                    "delta_seconds": row["delta_timestamp_seconds"],
                },
            )
        flags, score = [], 0.0
        if (
            gap is not None
            and abs(gap) >= 60
            and row["target_distance_m"] >= 5
            and row["route_lookahead_distance_m"] >= 5
        ):
            flags.append("sparse_target_bearing_gap")
            score += abs(gap) / 60
        if (
            row["delta_timestamp_seconds"] is not None
            and 0 < row["delta_timestamp_seconds"] <= 1
            and row["route_world_bearing_change_degrees"] is not None
            and abs(row["route_world_bearing_change_degrees"]) >= 35
        ):
            flags.append("plan_heading_change")
            score += abs(row["route_world_bearing_change_degrees"]) / 35
        if localization_gap is not None and localization_gap >= 1:
            flags.append("localization_sources_apart")
            score += min(localization_gap, 3)
        if (
            row["max_control_modification"] is not None
            and row["max_control_modification"] > 1e-5
        ):
            flags.append("controller_modified")
            score += row["max_control_modification"]
        if (
            row["lateral_controller_gap"] is not None
            and row["lateral_controller_gap"] >= 0.3
        ):
            flags.append("lateral_controllers_apart")
            score += row["lateral_controller_gap"]
        if previous is not None and any(
            row[name] is not None
            and previous[name] is not None
            and row[name] != previous[name]
            for name in ("command_id", "next_command_id")
        ):
            flags.append("navigation_command_transition")
            score += 0.5
        row["flags"], row["review_priority"] = "|".join(flags), score
        rows.append(row)
        previous = row
    return rows


def selected_geometry(record):
    observed, inputs, prediction = (
        section(record, name)
        for name in ("observed_state", "model_inputs", "predictions")
    )
    position, compass = (
        xy(observed.get("filtered_state")),
        scalar(observed.get("compass_radians")),
    )
    return {
        "coordinate_frame": "gps_derived_navigation",
        "projection": "Estimated GPS-derived navigation-frame XY using filtered observed position and observed compass; not ground truth. Simulator-world coordinates require origin calibration before comparison.",
        "filtered_position_world_m": position,
        "compass_radians": compass,
        "predicted_route_estimated_world_m": project_world(
            points(prediction.get("pred_route")), position, compass,
        ),
        "predicted_waypoints_estimated_world_m": project_world(
            points(prediction.get("pred_future_waypoints")), position, compass,
        ),
        "model_targets_estimated_world_m": {
            name: project_world(points(inputs.get(name)), position, compass)
            for name in ("target_point_previous", "target_point", "target_point_next")
        },
        "sparse_planner_targets_world_m": points(
            section(record, "navigation").get("remaining_target_points_world"),
        ),
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def analyze(trace: Path, output: Path, selected_step=None, plot=False):
    metadata, steps, issues, counts = read_trace(trace)
    rows = make_timeline(steps, issues)
    candidates = []
    # Keep up to 30 separated inspection moments while preserving every flag in CSV.
    for row in sorted(
        (row for row in rows if row["flags"]),
        key=lambda row: row["review_priority"],
        reverse=True,
    ):
        timestamp = row["timestamp_seconds"]
        if any(
            timestamp is not None
            and candidate["timestamp_seconds"] is not None
            and abs(timestamp - candidate["timestamp_seconds"]) < 1
            for candidate in candidates
        ):
            continue
        candidates.append(row)
        if len(candidates) == 30:
            break
    if selected_step is not None:
        matches = [
            record for record in steps if scalar(record.get("step")) == selected_step
        ]
        if len(matches) != 1:
            raise ValueError(
                f"--step {selected_step} must identify exactly one recorded step; found {len(matches)}",
            )
        selected = matches[0]
    elif candidates:
        selected = next(
            record
            for record in steps
            if record["source_line"] == candidates[0]["source_line"]
        )
    else:
        selected = steps[-1] if steps else None
    timestamps = [
        row["timestamp_seconds"] for row in rows if row["timestamp_seconds"] is not None
    ]
    speeds = [row["speed_mps"] for row in rows if row["speed_mps"] is not None]
    summary = {
        "trace": str(trace.resolve()),
        "route_id": metadata[0].get("route_id") if metadata else None,
        "metadata": metadata,
        "step_records": len(steps),
        "read_counts": counts,
        "first_timestamp_seconds": timestamps[0] if timestamps else None,
        "last_timestamp_seconds": timestamps[-1] if timestamps else None,
        "elapsed_timestamp_seconds": timestamps[-1] - timestamps[0]
        if timestamps
        else None,
        "observed_path_length_m": rows[-1]["observed_distance_m"] if rows else 0.0,
        "mean_speed_mps": sum(speeds) / len(speeds) if speeds else None,
        "max_speed_mps": max(speeds) if speeds else None,
        "missing_filtered_positions": sum(row["filtered_x_m"] is None for row in rows),
        "flagged_steps": sum(bool(row["flags"]) for row in rows),
        "flag_counts": dict(
            Counter(flag for row in rows for flag in row["flags"].split("|") if flag),
        ),
        "candidate_moments": len(candidates),
        "candidate_selection": "Up to 30 moments ranked by review_priority, separated by at least one second when timestamps exist; all flags remain in timeline.csv.",
        "selected_step": selected.get("step") if selected else None,
        "flag_descriptions": FLAG_DESCRIPTIONS,
        "issues": issues,
        "limitations": LIMITATIONS,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "timeline.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0])
            if rows
            else ["source_line", "step", "timestamp_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "candidate_moments.json",
        {
            "purpose": "Inspection candidates, not causal or wrong-turn verdicts.",
            "moments": candidates,
        },
    )
    write_json(
        output / "selected_step.json",
        {
            "record": selected,
            "geometry": selected_geometry(selected) if selected else None,
        },
    )
    if plot and selected is not None:
        write_plots(output, rows, selected)
    write_json(output / "summary.json", summary)
    return summary


def write_plots(output, rows, selected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = selected_geometry(selected)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    path = [
        [row["filtered_x_m"], row["filtered_y_m"]]
        if row["filtered_x_m"] is not None
        else [math.nan, math.nan]
        for row in rows
    ]
    for axis in axes:
        if path:
            axis.plot(
                *zip(*path, strict=True),
                color="0.6",
                label="Observed filtered path (navigation frame)",
            )
        for name, label, style in (
            (
                "predicted_route_estimated_world_m",
                "Predicted route (estimated navigation frame)",
                "b.-",
            ),
            (
                "predicted_waypoints_estimated_world_m",
                "Predicted waypoints (estimated navigation frame)",
                "g.-",
            ),
            (
                "sparse_planner_targets_world_m",
                "Sparse planner targets (not desired path)",
                "kx",
            ),
        ):
            if geometry[name]:
                axis.plot(*zip(*geometry[name], strict=True), style, label=label)
        for name, targets in geometry["model_targets_estimated_world_m"].items():
            if targets:
                axis.scatter(
                    *zip(*targets, strict=True),
                    label=f"Model {name} (projected at filtered pose)",
                    s=55,
                )
        position = geometry["filtered_position_world_m"]
        if position is not None:
            axis.scatter(*position, color="red", s=65, label="Selected observed pose")
            if geometry["compass_radians"] is not None:
                axis.arrow(
                    *position,
                    3 * math.cos(geometry["compass_radians"]),
                    3 * math.sin(geometry["compass_radians"]),
                    width=0.15,
                    color="red",
                )
        axis.set(
            xlabel="Navigation-frame x (m; +x right)",
            ylabel="Navigation-frame y (m; +y downward)",
            aspect="equal",
        )
        axis.grid(alpha=0.25)
    axes[0].set_title("Whole observed rollout")
    selected_timestamp = scalar(selected.get("timestamp_seconds"))
    time_label = (
        f"{selected_timestamp:.2f} s"
        if selected_timestamp is not None
        else "timestamp unavailable"
    )
    axes[1].set_title(f"Selected step {selected.get('step')} at {time_label}")
    position = geometry["filtered_position_world_m"]
    if position is not None:
        axes[1].set_xlim(position[0] - 35, position[0] + 35)
        axes[1].set_ylim(position[1] - 35, position[1] + 35)
    for axis in axes:
        axis.invert_yaxis()
    axes[0].legend(fontsize=7)
    fig.suptitle("Observed motion and predicted paths — GPS-derived navigation frame")
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(13, 15), sharex=True)
    times = [
        row["timestamp_seconds"] if row["timestamp_seconds"] is not None else math.nan
        for row in rows
    ]
    panels = (
        (("observed_distance_m", "Observed travel distance (not route completion)"),),
        (
            ("speed_mps", "Observed speed"),
            ("predicted_speed_mps", "Predicted target speed"),
        ),
        (
            ("compass_degrees", "Observed compass (wrapped)"),
            (
                "route_lookahead_world_bearing_degrees",
                "Route lookahead navigation bearing",
            ),
        ),
        (
            ("route_lookahead_bearing_degrees", "Route lookahead bearing"),
            ("target_bearing_degrees", "Sparse target bearing"),
        ),
        (
            ("executed_control_steer", "Executed steer"),
            ("route_steer", "Route controller"),
            ("waypoints_steer", "Waypoint controller"),
        ),
        (
            ("executed_control_throttle", "Executed throttle"),
            ("executed_control_brake", "Executed brake"),
        ),
    )
    for axis, fields in zip(axes, panels, strict=True):
        for name, label in fields:
            axis.plot(
                times,
                [row[name] if row[name] is not None else math.nan for row in rows],
                label=label,
            )
        selected_timestamp = scalar(selected.get("timestamp_seconds"))
        if selected_timestamp is not None:
            axis.axvline(selected_timestamp, color="red", alpha=0.6)
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(alpha=0.25)
    for axis, label in zip(
        axes,
        ("Meters", "m/s", "Degrees", "Ego degrees", "Steering", "Control"),
        strict=True,
    ):
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Original simulator timestamp (seconds)")
    fig.suptitle("Control and geometry timeline; red line marks the selected step")
    fig.tight_layout()
    fig.savefig(output / "timeline.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--step", type=int, help="Exact recorded step for the geometry snapshot",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write trajectory.png and timeline.png (requires matplotlib)",
    )
    args = parser.parse_args()
    try:
        summary = analyze(args.trace, args.output, args.step, args.plot)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "route_id": summary["route_id"],
                "step_records": summary["step_records"],
                "flagged_steps": summary["flagged_steps"],
                "issues": len(summary["issues"]),
                "selected_step": summary["selected_step"],
            },
        ),
    )


if __name__ == "__main__":
    main()
