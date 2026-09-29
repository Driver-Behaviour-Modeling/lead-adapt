"""ADAPT history sampling in the fixed current-ego coordinate frame.

Expert metadata reverses BaseAgent's tick-by-tick queues when saving them; it
does not downsample their contents. Thus metadata index 5 is five simulation
ticks old, even though metadata files themselves are saved every five ticks.
Training and the legacy runtime window both end at the current pose, so every
history delta and the first future delta span one waypoint interval. The
training_aligned runtime window ends one interval in the past; it reproduces the
history of checkpoints trained before the current pose was included.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class HistoryFeatures:
    """Oldest-to-newest poses and the timing of each returned sample.

    ``tick_ages`` describes the actual source samples. An empty source queue has
    no observed age, represented by -1 and a NaN timestamp. ``padding_mask``
    distinguishes repeated startup poses from observations at the requested age.
    Coordinates and yaws are passed through without another rotation.
    """

    positions: npt.NDArray[np.float32]
    yaws: npt.NDArray[np.float32]
    requested_tick_ages: npt.NDArray[np.int64]
    tick_ages: npt.NDArray[np.int64]
    padding_mask: npt.NDArray[np.bool_]
    tick_hz: float

    @property
    def time_offsets_seconds(self) -> npt.NDArray[np.float64]:
        """Actual sample timestamps relative to the current tick (seconds)."""
        return np.where(self.tick_ages >= 0, -self.tick_ages / self.tick_hz, np.nan)

    def timestamps(self, current_timestamp: float) -> npt.NDArray[np.float64]:
        """Actual sample timestamps on the caller's simulation clock."""
        return current_timestamp + self.time_offsets_seconds


def _sample_history(
    positions: npt.ArrayLike,
    yaws: npt.ArrayLike,
    *,
    num_history_poses: int,
    waypoints_spacing: int,
    newest_tick_age: int,
    newest_first: bool,
    pad_to_length: bool,
    tick_hz: float,
) -> HistoryFeatures:
    if num_history_poses < 1 or waypoints_spacing < 1:
        raise ValueError("History count and waypoint spacing must be positive.")
    if not np.isfinite(tick_hz) or tick_hz <= 0:
        raise ValueError("History tick frequency must be finite and positive.")
    positions_array = np.asarray(positions, dtype=np.float32)
    if positions_array.size == 0:
        positions_array = positions_array.reshape(0, 2)
    if positions_array.ndim != 2 or positions_array.shape[1] < 2:
        raise ValueError("History positions must have shape [N, 2] or [N, 3].")
    yaws_array = np.asarray(yaws, dtype=np.float32)
    if yaws_array.ndim != 1 or len(yaws_array) != len(positions_array):
        raise ValueError("History positions and yaws must describe the same ticks.")

    requested_ages = (
        newest_tick_age
        + np.arange(num_history_poses, dtype=np.int64)[::-1] * waypoints_spacing
    )
    available = requested_ages < len(positions_array)
    selected_ages = requested_ages[available]
    padding_mask = np.zeros(len(selected_ages), dtype=np.bool_)

    if pad_to_length:
        padding_mask = ~available
        if len(selected_ages):
            # Match legacy front padding: repeat the oldest sampled pose, which
            # need not be the oldest entry in the full tick queue.
            padding_age = selected_ages[0]
        else:
            # Before the aligned window has any samples, use the oldest available
            # observation. With an empty queue, -1 records that zeros are synthetic.
            padding_age = len(positions_array) - 1
        selected_ages = np.where(available, requested_ages, padding_age)
    else:
        # Dataset behavior intentionally remains variable-length at route startup.
        requested_ages = selected_ages.copy()

    if len(positions_array):
        indices = (
            selected_ages if newest_first else len(positions_array) - 1 - selected_ages
        )
        sampled_positions = positions_array[indices, :2].copy()
        sampled_yaws = yaws_array[indices].copy()
    else:
        sampled_positions = np.zeros((len(selected_ages), 2), dtype=np.float32)
        sampled_yaws = np.zeros(len(selected_ages), dtype=np.float32)
    return HistoryFeatures(
        positions=sampled_positions,
        yaws=sampled_yaws,
        requested_tick_ages=requested_ages,
        tick_ages=selected_ages,
        padding_mask=padding_mask,
        tick_hz=float(tick_hz),
    )


def sample_training_history(
    past_positions: npt.ArrayLike,
    past_yaws: npt.ArrayLike,
    *,
    num_history_poses: int,
    waypoints_spacing: int,
    tick_hz: float = 20.0,
) -> HistoryFeatures:
    """Select saved newest-first metadata with the training convention.

    For five poses and spacing five this returns ages [20, 15, 10, 5, 0]: the
    history ends at the current pose, which is also where the future starts.
    Unavailable startup samples are omitted, matching the existing dataloader.
    Sensor perturbation remains the dataloader's responsibility after sampling.
    """
    return _sample_history(
        past_positions,
        past_yaws,
        num_history_poses=num_history_poses,
        waypoints_spacing=waypoints_spacing,
        newest_tick_age=0,
        newest_first=True,
        pad_to_length=False,
        tick_hz=tick_hz,
    )


def sample_runtime_history(
    past_positions: npt.ArrayLike,
    past_yaws: npt.ArrayLike,
    *,
    num_history_poses: int,
    waypoints_spacing: int,
    mode: Literal["legacy", "training_aligned"] = "legacy",
    tick_hz: float = 20.0,
) -> HistoryFeatures:
    """Sample oldest-first BaseAgent queues, padding unavailable startup history.

    ``legacy`` exactly retains the existing runtime values: ages [20, 15, 10, 5,
    0] for five poses at spacing five, matching the training convention.
    ``training_aligned`` uses [25, 20, 15, 10, 5], the training inputs of
    checkpoints trained before the current pose was included.
    Changing the sampling mode does not change the current-ego coordinate frame.
    """
    if mode not in ("legacy", "training_aligned"):
        raise ValueError(f"Unknown history sampling mode: {mode!r}")
    return _sample_history(
        past_positions,
        past_yaws,
        num_history_poses=num_history_poses,
        waypoints_spacing=waypoints_spacing,
        newest_tick_age=waypoints_spacing if mode == "training_aligned" else 0,
        newest_first=False,
        pad_to_length=True,
        tick_hz=tick_hz,
    )
