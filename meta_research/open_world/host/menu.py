"""Public experiment menu. No hidden mismatch tags."""

COVER_PLAN = [
    {
        "initial_state": {"x": 0.4, "v": 0.0},
        "input_signal": {"kind": "sine", "amp": 0.6, "freq": 0.3, "phase": 0.0},
        "duration": 3.5,
        "sampling": {"dt": 0.05},
    },
    {
        "initial_state": {"x": -0.3, "v": 0.4},
        "input_signal": {"kind": "step", "amp": 0.5, "freq": 0.0, "phase": 0.0},
        "duration": 3.0,
        "sampling": {"dt": 0.05},
    },
    {
        "initial_state": {"x": 0.0, "v": 0.0},
        "input_signal": {"kind": "sine", "amp": 1.1, "freq": 0.22, "phase": 0.0},
        "duration": 3.5,
        "sampling": {"dt": 0.05},
    },
    {
        "initial_state": {"x": 0.7, "v": -0.2},
        "input_signal": {"kind": "pulse", "amp": 1.2, "freq": 0.8, "phase": 0.0},
        "duration": 3.0,
        "sampling": {"dt": 0.05},
    },
    {
        "initial_state": {"x": 0.2, "v": 0.6},
        "input_signal": {"kind": "chirp", "amp": 0.7, "freq": 0.18, "phase": 0.0},
        "duration": 3.5,
        "sampling": {"dt": 0.05},
    },
    {
        "initial_state": {"x": 0.5, "v": 0.0},
        "input_signal": {"kind": "sine", "amp": 0.5, "freq": 0.35, "phase": 0.0},
        "duration": 3.0,
        "sampling": {"dt": 0.05},
    },
    {
        "initial_state": {"x": -0.5, "v": 0.0},
        "input_signal": {"kind": "step", "amp": -0.6, "freq": 0.0, "phase": 0.0},
        "duration": 3.0,
        "sampling": {"dt": 0.05},
    },
    {
        "initial_state": {"x": 0.0, "v": 0.5},
        "input_signal": {"kind": "sine", "amp": 0.9, "freq": 0.45, "phase": 0.7},
        "duration": 3.0,
        "sampling": {"dt": 0.05},
    },
]

PROBE_SPECS = COVER_PLAN + [
    {
        "initial_state": {"x": 0.5, "v": 0.0},
        "input_signal": {"kind": "zero", "amp": 0.0, "freq": 0.0, "phase": 0.0},
        "duration": 2.5,
        "sampling": {"dt": 0.05},
    },
    {
        "initial_state": {"x": 0.0, "v": 0.0},
        "input_signal": {"kind": "step", "amp": 1.0, "freq": 0.0, "phase": 0.0},
        "duration": 2.5,
        "sampling": {"dt": 0.05},
    },
]
