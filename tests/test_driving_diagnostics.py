import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lead.inference.driving_diagnostics import (
    DrivingDiagnostics,
    control_values,
    json_value,
    prediction_values,
)


def test_trace_flushes_before_close_and_never_overwrites(tmp_path):
    recorder = DrivingDiagnostics(tmp_path, {"route_id": "3905"})
    recorder.write(
        record_type="step",
        step=1,
        values=torch.tensor([1.0, float("nan"), float("inf")], requires_grad=True),
    )
    rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert rows[0]["schema_version"] == 1
    assert rows[1]["values"] == [1.0, None, None]
    with pytest.raises(FileExistsError):
        DrivingDiagnostics(tmp_path, {"route_id": "3905"})
    assert len(recorder.path.read_text().splitlines()) == 2
    recorder.close()
    recorder.close()


def test_small_predictions_are_detached_without_images_or_mutation():
    tensor = torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16)
    prediction = SimpleNamespace(
        pred_route=tensor,
        pred_semantic=torch.ones(1, 8, 10, 10),
        steer=0.2,
        throttle=0.5,
        brake=0.0,
    )
    values = prediction_values(prediction)
    assert "pred_semantic" not in values
    assert json_value(values)["pred_route"] == [[[1.0, 2.0]]]
    assert control_values(prediction) == {"steer": 0.2, "throttle": 0.5, "brake": 0.0}
    assert tensor.dtype == torch.bfloat16
    assert json_value(np.array([np.nan, 2.0])) == [None, 2.0]


def test_unsupported_values_do_not_write_partial_records(tmp_path):
    recorder = DrivingDiagnostics(tmp_path, {})
    with pytest.raises(TypeError):
        recorder.write(record_type="step", value=object())
    recorder.close()
    assert len(recorder.path.read_text().splitlines()) == 1
