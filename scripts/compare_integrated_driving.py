"""Audit and compare official route results from two controlled driving bundles.

The fixed planned-route denominator includes missing attempts as zero. A score
gain is reported only when every primary attempt and comparison invariant passes.
Frozen-model ablations are descriptive, not matched-training attribution.
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

from scripts.after_integrated_training import live_process
from scripts.diagnose_turns import file_hash, inspect_attempt_outputs


def flags(value):
    return dict(item.split("=", 1) for item in value.split())


def audit_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    route_ids = [row["id"] for row in manifest["routes"]]
    if not route_ids or len(route_ids) != len(set(route_ids)):
        raise ValueError("Route plan is empty or duplicated")
    issues = []
    if manifest["attempts_per_route_variant"] != 1 or manifest["repetitions"] != 1:
        issues.append("Expected one attempt and repetition per planned route")
    for row in manifest["routes"]:
        route_path = directory / "routes" / f"{row['id']}.xml"
        if not route_path.exists() or file_hash(route_path) != row["sha256"]:
            issues.append(f"Staged route changed or missing: {row['id']}")
    weights = list((directory / "checkpoint").glob("model*.pth"))
    if len(weights) != 1 or file_hash(weights[0]) != manifest["checkpoint"]["sha256"]:
        issues.append("Selected checkpoint changed or there is not exactly one model")
    config_path = directory / "checkpoint/config.json"
    if file_hash(config_path) != manifest["effective_training_config"]["sha256"]:
        issues.append("Effective inference config changed")
    config = json.loads(config_path.read_text())
    for key, expected in (
        ("adapt_history_free", True),
        ("adapt_waypoints_from_tokens", True),
        ("adapt_decoding", "beam"),
        ("adapt_beam_width", 8),
    ):
        if config.get(key) != expected:
            issues.append(f"Unexpected token-control setting: {key}")
    cases, summaries = {}, {}
    for variant, configuration in manifest["variants"].items():
        selected = flags(configuration)
        if any(
            selected.get(name) != "waypoints"
            for name in ("steer_modality", "throttle_modality", "brake_modality")
        ):
            issues.append(f"Unexpected controller in {variant}")
        cases[variant] = {}
        for route_id in route_ids:
            folder = directory / variant / route_id
            inspected = inspect_attempt_outputs(folder)
            errors = list(inspected["output_errors"])
            try:
                attempt = json.loads((folder / "attempt.json").read_text())
                if (
                    attempt.get("status") != "evaluator_finished"
                    or attempt.get("returncode") != 0
                ):
                    errors.append("Attempt did not finish successfully")
                if attempt.get("closed_loop_config") != configuration:
                    errors.append(
                        "Attempt configuration differs from its frozen variant",
                    )
                if attempt.get("attempt") != 1:
                    errors.append(
                        "Attempt number differs from the one-attempt protocol",
                    )
            except (OSError, ValueError) as error:
                errors.append(str(error))
            record = (
                inspected.get("records", [{}])[0]
                if inspected["has_route_result"]
                else {}
            )
            if record and not re.fullmatch(
                rf"RouteScenario_{re.escape(route_id)}_rep0",
                str(record.get("route_id", "")),
            ):
                errors.append("Official record belongs to another route or repetition")
            score = record.get("scores", {}).get("score_composed", 0.0)
            if (
                not isinstance(score, int | float)
                or not math.isfinite(score)
                or not 0 <= score <= 100
            ):
                errors.append("Official score is not finite and within [0,100]")
                score = 0.0
            cases[variant][route_id] = {
                "valid": not errors,
                "errors": errors,
                "official_status": record.get("status"),
                "reported_score": float(score),
                "planned_score": float(score) if not errors else 0.0,
                "infractions": {
                    key: len(value) if isinstance(value, list) else value
                    for key, value in record.get("infractions", {}).items()
                },
            }
        rows = list(cases[variant].values())
        summaries[variant] = {
            "planned_routes": len(rows),
            "valid_routes": sum(row["valid"] for row in rows),
            "mean_with_fixed_denominator": sum(row["planned_score"] for row in rows)
            / len(rows),
            "invalid_route_ids": [
                key for key, row in cases[variant].items() if not row["valid"]
            ],
        }
    return manifest, config, issues, cases, summaries


def compare(baseline, combined):
    bm, bc, bi, br, bs = audit_run(baseline)
    cm, cc, ci, cr, cs = audit_run(combined)
    issues = [f"e26: {issue}" for issue in bi] + [f"combined: {issue}" for issue in ci]
    expected_routes = {row["id"]: row["sha256"] for row in bm["routes"]}
    if expected_routes != {row["id"]: row["sha256"] for row in cm["routes"]}:
        issues.append("Route IDs/XML/weather bytes differ")
    if bm["seed"] != cm["seed"]:
        issues.append("Evaluator seeds differ")
    for key in ("cuda_device", "graphics_adapter", "cpu_threads"):
        if bm["execution"].get(key) != cm["execution"].get(key):
            issues.append(f"Execution protocol differs: {key}")
    for key in ("kinematic_vocab_size", "kdisks_heading_weight"):
        if bc.get(key) != cc.get(key):
            issues.append(f"Motion vocabulary configuration differs: {key}")
    if {row["config_key"]: row["sha256"] for row in bm["codebooks"]} != {
        row["config_key"]: row["sha256"] for row in cm["codebooks"]
    }:
        issues.append("Codebook files differ")
    prefixes = (
        "lead/adapt/",
        "lead/inference/",
        "lead/data_loader/",
        "lead/common/",
        "lead/training/config_training.py",
    )
    runtime_sources = [
        {
            key: value
            for key, value in manifest["source"]["files"].items()
            if key.startswith(prefixes)
        }
        for manifest in (bm, cm)
    ]
    if runtime_sources[0] != runtime_sources[1]:
        issues.append("Driving runtime source versions differ")
    # Branch-disabling switches are only permitted in the explicit ablations.
    if flags(bm["variants"]["baseline"]) != flags(cm["variants"]["baseline"]):
        issues.append("Primary closed-loop controller/input settings differ")
    primary_valid = (
        not issues
        and all(row["valid"] for row in br["baseline"].values())
        and all(row["valid"] for row in cr["baseline"].values())
    )
    routes = []
    for route_id in expected_routes:
        before = br["baseline"][route_id]
        after = cr["baseline"].get(route_id, {"valid": False, "planned_score": 0.0})
        routes.append(
            {
                "route_id": route_id,
                "e26_score": before["planned_score"],
                "combined_score": after["planned_score"],
                "both_valid": before["valid"] and after["valid"],
                "delta": after["planned_score"] - before["planned_score"],
            },
        )
    delta = (
        cs["baseline"]["mean_with_fixed_denominator"]
        - bs["baseline"]["mean_with_fixed_denominator"]
    )
    return {
        "baseline_bundle": str(baseline.resolve()),
        "combined_bundle": str(combined.resolve()),
        "primary_comparison_valid": primary_valid,
        "protocol_issues": issues,
        "full_220_route_comparison": primary_valid
        and len(expected_routes) == 220
        and bm.get("evaluation_scope", {}).get("full_benchmark", False)
        and cm.get("evaluation_scope", {}).get("full_benchmark", False),
        "e26": bs,
        "combined": cs,
        "routes": routes,
        "validated_mean_delta": delta if primary_valid else None,
        "interpretation": "Fixed planned-route denominators include execution/missing failures as zero for accounting. A validated policy comparison requires every primary attempt and protocol check to pass. A subset cannot establish the full benchmark score. Frozen ablations are diagnostic only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--after-process",
        type=Path,
        help="Wait for the exact pid/process_start_ticks in this launch JSON before reading bundles",
    )
    args = parser.parse_args()
    if args.output.suffix != ".json":
        raise ValueError("--output must name a JSON file")
    if args.output.exists() or args.output.with_suffix(".md").exists():
        raise FileExistsError(args.output)
    if args.after_process:
        process = json.loads(args.after_process.read_text())
        while live_process(process["pid"], process["process_start_ticks"]):
            time.sleep(30)
    result = compare(args.baseline, args.combined)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "Paired token-driving comparison",
        "",
        f"Primary comparison valid: **{result['primary_comparison_valid']}**. Full 220-route comparison: **{result['full_220_route_comparison']}**.",
        "",
        "| Route | e26 | Combined | Delta | Both valid |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in result["routes"]:
        lines.append(
            f"| {row['route_id']} | {row['e26_score']:.2f} | {row['combined_score']:.2f} | {row['delta']:+.2f} | {row['both_valid']} |",
        )
    lines += ["", result["interpretation"], ""]
    if result["protocol_issues"]:
        lines += ["Protocol issues:", ""] + [
            f"- {value}" for value in result["protocol_issues"]
        ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(
        f"Comparison valid: {result['primary_comparison_valid']}; validated mean delta: {result['validated_mean_delta']}",
    )


if __name__ == "__main__":
    main()
