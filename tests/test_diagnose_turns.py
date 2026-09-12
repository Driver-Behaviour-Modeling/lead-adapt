"""Isolation and experiment validity checks, without importing CARLA or Torch."""

import argparse
import json
import os
import signal
import subprocess
import sys

import pytest

from scripts import diagnose_turns as runner


@pytest.fixture
def inputs(tmp_path):
    project = tmp_path / "project"
    checkpoint_dir = project / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model_0039.pth"
    checkpoint.write_bytes(b"selected checkpoint")
    (checkpoint_dir / "model_0038.pth").write_bytes(b"must not ensemble this")
    (project / "codebook.pkl").write_bytes(b"codebook")
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"kdisks_vocab_path": "codebook.pkl"}),
    )
    route_dir = project / "routes"
    route_dir.mkdir()
    # Deliberately include a comment, whitespace, and attributes: byte copying
    # must preserve all of them, not just reconstruct approximate waypoints.
    (route_dir / "14842.xml").write_bytes(
        b'<?xml version="1.0"?><routes><!-- original -->\n'
        b'<route id="14842" road_id="282" town="Town12">\n'
        b'<waypoints><position x="2.6" y="7" z="0" /></waypoints>'
        b'<scenarios><scenario type="GreenTurn" name="original" /></scenarios>'
        b'<weathers><weather route_percentage="0" wetness="100" /></weathers>'
        b"</route></routes>\n",
    )
    return project, checkpoint, route_dir


def prepare(inputs, output, variants=None, snapshot_steps=None):
    project, checkpoint, route_dir = inputs
    return runner.prepare_experiment(
        output,
        checkpoint,
        route_dir,
        ["14842"],
        variants or ["baseline"],
        project=project,
        snapshot_steps=snapshot_steps,
    )


def test_bundle_preserves_original_inputs_and_selects_exactly_one_model(
    inputs,
    tmp_path,
):
    output = tmp_path / "new experiment"
    manifest = prepare(inputs, output)
    _, checkpoint, route_dir = inputs
    assert (output / "routes/14842.xml").read_bytes() == (
        route_dir / "14842.xml"
    ).read_bytes()
    assert (output / "checkpoint/config.json").read_bytes() == (
        checkpoint.parent / "config.json"
    ).read_bytes()
    models = list((output / "checkpoint").glob("model*.pth"))
    assert len(models) == 1
    assert models[0].is_symlink() and models[0].resolve() == checkpoint
    assert models[0].read_bytes() == b"selected checkpoint"
    assert manifest["codebooks"][0]["sha256"] == runner.file_hash(
        inputs[0] / "codebook.pkl",
    )
    assert manifest["checkpoint"]["sha256"] == runner.file_hash(checkpoint)
    assert manifest["routes"][0]["reference_outcome"] == "failed_turn"
    assert manifest["attempts_per_route_variant"] == manifest["repetitions"] == 1


def test_existing_output_is_never_reused_or_overwritten(inputs, tmp_path):
    output = tmp_path / "existing evaluation"
    output.mkdir()
    sentinel = output / "result.json"
    sentinel.write_text("old result")
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        prepare(inputs, output)
    assert sentinel.read_text() == "old result"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize("ids", [["14842", "14842"], ["../14842"], ["3904"]])
def test_routes_are_unambiguous_and_cannot_escape_source_directory(inputs, ids):
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.checked_routes(inputs[2], ids)


def test_missing_weather_is_rejected_before_creating_output(inputs, tmp_path):
    route = inputs[2] / "14842.xml"
    route.write_text(
        '<routes><route id="14842"><waypoints/><scenarios/></route></routes>',
    )
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="weathers"):
        prepare(inputs, output)
    assert not output.exists()


def test_variants_change_exactly_one_behavior_flag():
    def parse(name):
        return dict(part.split("=", 1) for part in runner.variant_config(name).split())

    baseline = parse("baseline")
    for name, changed in runner.VARIANTS.items():
        values = parse(name)
        differences = {
            key: value for key, value in values.items() if baseline[key] != value
        }
        assert differences == changed
        assert values["record_driving_diagnostics"] == "true"
        assert values["debug_mode"] == "false"


def test_controlled_environment_drops_ambient_config_and_pythonpath(
    inputs,
    tmp_path,
    monkeypatch,
):
    manifest = prepare(inputs, tmp_path / "new")
    for key in runner.CONFIG_ENVIRONMENTS:
        monkeypatch.setenv(key, "dangerous_override=true")
    monkeypatch.setenv("PYTHONPATH", "/other/checkouts")
    env = runner.evaluation_environment(manifest, "GPU-example", inputs[0] / "carla")
    assert all(key not in env for key in runner.CONFIG_ENVIRONMENTS)
    assert "/other/checkouts" not in env["PYTHONPATH"]
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-example"
    assert env["PYTHONHASHSEED"] == "0"


def test_evaluator_uses_one_attempt_without_resume_or_accidental_recorder(tmp_path):
    command = runner.evaluator_command(
        "python",
        tmp_path,
        tmp_path / "run",
        tmp_path / "output",
        "14842",
        23000,
        23003,
        0,
    )
    assert "--repetitions=1" in command
    assert "--resume=False" in command
    assert not any(argument.startswith("--record") for argument in command)
    assert f"--checkpoint={tmp_path / 'run/result.json'}" in command
    assert "random.seed(0)" in command[2]


def test_infrastructure_failure_is_recorded_once_and_does_not_retry(
    inputs,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "new"
    manifest = prepare(inputs, output)
    starts = []

    def fail_start(*args, **kwargs):
        starts.append((args, kwargs))
        raise OSError("simulator cannot start")

    monkeypatch.setattr(runner.subprocess, "Popen", fail_start)
    monkeypatch.setattr(runner, "available_ports", lambda: (23000, 23003))
    args = argparse.Namespace(
        carla_root=tmp_path / "carla",
        graphics_adapter=0,
        python="python",
        boot_timeout=1,
        route_timeout=1,
        cpu_threads=4,
    )
    result = runner.run_one(manifest, "baseline", "14842", args, {})
    assert len(starts) == 1
    assert result["status"] == "infrastructure_error"
    assert result["has_route_result"] is False
    assert result["diagnostics_bytes"] == 0
    assert result["attempt"] == 1
    assert (
        json.loads((output / "baseline/14842/attempt.json").read_text())["error"]
        == "simulator cannot start"
    )


def test_cli_prepares_by_default():
    args = runner.parse_args([])
    assert args.run is False
    assert args.routes == list(runner.DEFAULT_ROUTES)
    assert args.variants == ["baseline"]
    assert args.cpu_threads == 4
    assert args.snapshot_steps is None


def test_interpreter_bin_is_available_without_conda_activation(inputs, tmp_path):
    manifest = prepare(inputs, tmp_path / "new")
    python = tmp_path / "conda/envs/lead/bin/python"
    env = runner.evaluation_environment(manifest, "0", inputs[0] / "carla", str(python))
    assert env["PATH"].split(os.pathsep)[0] == str(python.parent)


def create_attempt_output(
    tmp_path,
    status="Failed - Agent deviated from the route",
    steps=1,
):
    runner.write_json(
        tmp_path / "result.json",
        {"_checkpoint": {"records": [{"status": status}]}},
    )
    (tmp_path / "agent").mkdir()
    records = [{"record_type": "metadata"}]
    records.extend({"record_type": "step", "step": index} for index in range(steps))
    (tmp_path / "agent/driving_diagnostics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
    )


@pytest.mark.parametrize(
    "status",
    ["Perfect", "Failed - Agent deviated from the route", "Failed - Agent got blocked"],
)
def test_driving_failures_with_real_steps_are_valid_diagnostic_attempts(
    tmp_path,
    status,
):
    create_attempt_output(tmp_path, status=status)
    result = runner.inspect_attempt_outputs(tmp_path)
    assert result["has_route_result"] is True
    assert result["diagnostics_step_records"] == 1
    assert result["output_errors"] == []


@pytest.mark.parametrize(
    "status",
    [
        "Failed - Agent couldn't be set up",
        "Failed - Agent crashed",
        "Failed - Simulation crashed",
    ],
)
def test_evaluator_record_does_not_hide_agent_or_simulator_errors(tmp_path, status):
    create_attempt_output(tmp_path, status=status)
    result = runner.inspect_attempt_outputs(tmp_path)
    assert result["has_route_result"] is True
    assert any("execution failure" in error for error in result["output_errors"])


def test_metadata_alone_does_not_count_as_successful_instrumentation(tmp_path):
    create_attempt_output(tmp_path, steps=0)
    result = runner.inspect_attempt_outputs(tmp_path)
    assert result["diagnostics_bytes"] > 0
    assert result["diagnostics_step_records"] == 0
    assert any("metadata alone" in error for error in result["output_errors"])


def test_partial_last_diagnostic_line_is_reported(tmp_path):
    create_attempt_output(tmp_path)
    with (tmp_path / "agent/driving_diagnostics.jsonl").open("a") as stream:
        stream.write('{"record_type":')
    result = runner.inspect_attempt_outputs(tmp_path)
    assert result["diagnostics_step_records"] == 1
    assert any("Cannot parse" in error for error in result["output_errors"])


def test_termination_during_cleanup_stops_both_children_and_finalizes_attempt(
    inputs,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "new"
    manifest = prepare(inputs, output)

    class Process:
        returncode = 0

        def wait(self, timeout):
            return 0

    started = []
    stopped = []

    def start(*args, **kwargs):
        process = Process()
        started.append(process)
        return process

    def stop(process):
        if not stopped:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        stopped.append(process)

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    monkeypatch.setattr(runner, "stop_process_group", stop)
    monkeypatch.setattr(runner, "wait_for_server", lambda *args: None)
    monkeypatch.setattr(runner, "available_ports", lambda: (23000, 23003))
    args = argparse.Namespace(
        carla_root=tmp_path / "carla",
        graphics_adapter=0,
        python="python",
        boot_timeout=1,
        route_timeout=1,
        cpu_threads=4,
    )
    original_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        runner.run_one(manifest, "baseline", "14842", args, {})
    assert stopped == list(reversed(started))
    assert signal.getsignal(signal.SIGTERM) == original_handler
    assert (
        json.loads((output / "baseline/14842/attempt.json").read_text())["status"]
        == "interrupted"
    )


def test_cpu_thread_limits_override_ambient_pools_or_can_be_disabled(
    inputs,
    tmp_path,
    monkeypatch,
):
    manifest = prepare(inputs, tmp_path / "new")
    for name in runner.THREAD_ENVIRONMENTS:
        monkeypatch.setenv(name, "17")
    limited = runner.evaluation_environment(
        manifest,
        "0",
        inputs[0] / "carla",
        cpu_threads=4,
    )
    inherited = runner.evaluation_environment(
        manifest,
        "0",
        inputs[0] / "carla",
        cpu_threads=0,
    )
    assert all(limited[name] == "4" for name in runner.THREAD_ENVIRONMENTS)
    assert all(inherited[name] == "17" for name in runner.THREAD_ENVIRONMENTS)
    assert runner.torch_thread_setup(0) == ""


def test_evaluator_bootstrap_applies_real_torch_and_numba_thread_limits(
    inputs,
    tmp_path,
):
    manifest = prepare(inputs, tmp_path / "new")
    project = inputs[0]
    fake_evaluator = (
        project
        / "3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py"
    )
    fake_evaluator.parent.mkdir(parents=True)
    fake_evaluator.write_text(
        "import json, os, numba, torch\n"
        "print(json.dumps({'torch':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),"
        "'numba':numba.get_num_threads(),'blas':os.environ['OPENBLAS_NUM_THREADS']}))\n",
    )
    env = runner.evaluation_environment(
        manifest,
        "0",
        project / "carla",
        sys.executable,
        cpu_threads=4,
    )
    command = runner.evaluator_command(
        sys.executable,
        project,
        tmp_path,
        tmp_path,
        "14842",
        23000,
        23003,
        0,
        cpu_threads=4,
    )
    completed = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "torch": 4,
        "interop": 4,
        "numba": 4,
        "blas": "4",
    }


def test_snapshot_selection_is_recorded_and_applied_to_every_variant(inputs, tmp_path):
    steps = [240, 241, 305]
    manifest = prepare(inputs, tmp_path / "new", ["baseline", "history_aligned"], steps)
    assert manifest["snapshot_steps"] == steps
    for variant in ("baseline", "history_aligned"):
        assert (
            manifest["variants"][variant]
            == runner.variant_config(variant)
            + " diagnostic_snapshot_steps=[240,241,305]"
        )
        assert "diagnostic_snapshot_steps" not in runner.variant_config(variant)
    assert (
        runner.parse_args(["--snapshot-steps", "240", "241", "305"]).snapshot_steps
        == steps
    )


@pytest.mark.parametrize("values", [["1", "1"], ["-1"], ["1.5"]])
def test_cli_rejects_duplicate_negative_or_noninteger_snapshot_steps(values):
    with pytest.raises(SystemExit) as error:
        runner.parse_args(["--snapshot-steps", *values])
    assert error.value.code == 2


@pytest.mark.parametrize("steps", [[1, 1], [-1], [True], [1.5]])
def test_invalid_snapshot_selection_fails_before_output_creation(
    inputs,
    tmp_path,
    steps,
):
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        prepare(inputs, output, snapshot_steps=steps)
    assert not output.exists()


def publish_snapshot(run_dir, step, *, digest=None, filename=None):
    directory = run_dir / "agent/model_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{step:05d}.pth"
    path.write_bytes(f"snapshot payload for step {step}".encode())
    row = {
        "schema_version": 1,
        "step": step,
        "timestamp_seconds": step / 20,
        "file": filename or path.name,
        "sha256": digest or runner.file_hash(path),
        "size_bytes": path.stat().st_size,
    }
    with (directory / "index.jsonl").open("a") as index:
        index.write(json.dumps(row) + "\n")
    return path


def test_reached_snapshot_selection_is_verified_by_index_size_and_hash(tmp_path):
    create_attempt_output(tmp_path, steps=5)
    publish_snapshot(tmp_path, 1)
    publish_snapshot(tmp_path, 3)
    result = runner.inspect_attempt_outputs(tmp_path, [1, 3])
    assert result["snapshot_capture_complete"] is True
    assert result["snapshot_saved_steps"] == [1, 3]
    assert result["snapshot_missing_steps"] == []
    assert result["snapshot_missing_reached_steps"] == []
    assert result["snapshot_unobserved_steps"] == []
    assert len(result["snapshot_index_sha256"]) == 64
    assert result["output_errors"] == []


def test_early_route_end_reports_partial_capture_without_execution_error(tmp_path):
    create_attempt_output(tmp_path, steps=3)
    publish_snapshot(tmp_path, 1)
    result = runner.inspect_attempt_outputs(tmp_path, [1, 20])
    assert result["snapshot_capture_complete"] is False
    assert result["snapshot_missing_steps"] == [20]
    assert result["snapshot_unobserved_steps"] == [20]
    assert result["snapshot_missing_reached_steps"] == []
    assert result["output_errors"] == []


def test_missing_snapshot_at_an_observed_step_is_an_output_error(tmp_path):
    create_attempt_output(tmp_path, steps=5)
    publish_snapshot(tmp_path, 1)
    result = runner.inspect_attempt_outputs(tmp_path, [1, 3])
    assert result["snapshot_capture_complete"] is False
    assert result["snapshot_missing_reached_steps"] == [3]
    assert result["snapshot_unobserved_steps"] == []
    assert any("no verified snapshot" in error for error in result["output_errors"])


def test_no_snapshot_index_is_allowed_only_when_no_requested_steps_were_observed(
    tmp_path,
):
    create_attempt_output(tmp_path, steps=3)
    unobserved = runner.inspect_attempt_outputs(tmp_path, [20])
    assert unobserved["snapshot_capture_complete"] is False
    assert unobserved["snapshot_index_present"] is False
    assert unobserved["output_errors"] == []
    reached = runner.inspect_attempt_outputs(tmp_path, [1])
    assert reached["snapshot_missing_reached_steps"] == [1]
    assert reached["output_errors"]


@pytest.mark.parametrize(
    "corruption",
    ["hash", "size", "duplicate", "filename", "partial_index"],
)
def test_invalid_snapshot_publication_is_reported(tmp_path, corruption):
    create_attempt_output(tmp_path, steps=3)
    path = publish_snapshot(
        tmp_path,
        1,
        digest="0" * 64 if corruption == "hash" else None,
        filename="../../outside.pth" if corruption == "filename" else None,
    )
    if corruption == "size":
        path.write_bytes(b"truncated")
    elif corruption == "duplicate":
        publish_snapshot(tmp_path, 1)
    elif corruption == "partial_index":
        with (path.parent / "index.jsonl").open("a") as index:
            index.write('{"step":')
    result = runner.inspect_attempt_outputs(tmp_path, [1])
    assert result["snapshot_capture_complete"] is False
    assert result["snapshot_errors"]
    assert result["output_errors"]
