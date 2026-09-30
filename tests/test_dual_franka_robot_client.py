"""No cameras or robots: verify reused control and native GR00T boundary."""

import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

# Upstream tests/examples is a regular package with the same name as the runtime namespace.
import examples
import numpy as np
import pytest


examples.__path__.append(str(Path(__file__).parents[1] / "examples"))

# isort: split

from examples.dual_franka import inference_dual_ee14 as client  # noqa: E402
from examples.dual_franka.robot_control import (  # noqa: E402
    execute_chunk,
    execution_commands,
    gripper_width,
    robot_session,
    rpc_state,
)


def test_metric_state_and_independent_grippers():
    obs = {
        "left_arm": {"gripper": {"position": 255, "open_position": 0, "closed_position": 255}},
        "right_arm": {"gripper": {"open_fraction": 0.5}},
    }
    state = rpc_state(obs, lambda _, side: np.arange(6) + (0 if side == "left_arm" else 6))
    np.testing.assert_array_equal(state[:12], np.arange(12))
    np.testing.assert_allclose(state[12:], [0, 0.0425])
    assert gripper_width({"gripper": {"width": 0.015}}) == 0.015
    with pytest.raises(ValueError):
        gripper_width({"gripper": {"width": float("nan")}})


def test_actions_and_rpc_rejection():
    actions = np.ones((40, 14), np.float32)
    actions[:, 12:] = [0.015, 0.065]
    rpc = Mock()
    commands = execute_chunk(rpc, actions, {"execute_horizon": 1}, sleep=lambda _: None)
    np.testing.assert_allclose(
        commands[0], [0.3] * 3 + [0.12] * 3 + [0.3] * 3 + [0.12] * 3 + [0, 0.065]
    )
    assert rpc.dual_robot_move_to_ee_pose.call_args.kwargs["delta"] is True
    assert rpc.left_gripper_goto.call_args.args == (0.0,)
    np.testing.assert_array_equal(actions[:, 12], np.full(40, 0.015, np.float32))
    rpc.reset_mock()
    rpc.dual_robot_move_to_ee_pose.return_value = {"ok": False}
    with pytest.raises(RuntimeError):
        execute_chunk(rpc, actions, {"execute_horizon": 2}, sleep=lambda _: None)
    rpc.left_gripper_goto.assert_not_called()
    actions[0, 0] = np.nan
    with pytest.raises(ValueError):
        execution_commands(actions)


@pytest.mark.parametrize("error", [None, KeyboardInterrupt, TimeoutError])
def test_reset_before_after(error):
    rpc = Mock()
    if error is None:
        with robot_session(rpc, True, 5):
            pass
    else:
        with pytest.raises(error), robot_session(rpc, True, 5):
            raise error()
    assert rpc.go_home.call_count == 2
    assert rpc.open_gripper.call_count == 4
    rpc.go_home.assert_called_with("both", duration_sec=5)


def test_failed_initial_reset_not_retried_and_replay_does_not_reset():
    rpc = Mock()
    rpc.go_home.return_value = False
    with pytest.raises(RuntimeError), robot_session(rpc, True, 5):
        pytest.fail("must abort before inference")
    assert rpc.go_home.call_count == 1
    rpc.reset_mock()
    with robot_session(rpc, False, 5):
        pass
    rpc.go_home.assert_not_called()


def test_raw_observation_and_action_contract():
    images = {key: np.zeros((240, 424, 3), np.uint8) for key in client.CAMERAS}
    obs = client.observation(np.arange(14), images, "task")
    assert obs["video"]["head"].shape == (1, 1, 240, 424, 3)
    np.testing.assert_array_equal(obs["state"]["right_ee"][0, 0], np.arange(6, 12))
    action = {k: np.zeros((1, 40, e - s)) for k, (s, e) in zip(client.KEYS, client.SLICES)}
    assert client.validate_action(action).shape == (40, 14)
    action["left_ee"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        client.validate_action(action)


def test_replay_never_imports_hardware(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture.npz"
    np.savez(
        fixture,
        state=np.zeros(14),
        text="task",
        **{k: np.zeros((240, 424, 3), np.uint8) for k in client.CAMERAS},
    )
    imports = builtins.__import__

    def guard(name, *args, **kwargs):
        assert "pyrealsense" not in name and "zerorpc" not in name and "robotiq_rpc" not in name
        return imports(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    policy = SimpleNamespace(infer=lambda *a, **kw: (np.zeros((40, 14)), 0.01), close=lambda: None)
    monkeypatch.setattr(client, "Client", lambda _: policy)
    client.run(
        {
            "fixture": str(fixture),
            "server": {},
            "runtime": {
                "mode": "replay",
                "execute": False,
                "iterations": 1,
                "output": str(tmp_path / "output"),
            },
        }
    )
    assert '"executed": false' in (tmp_path / "output/steps.jsonl").read_text()


def test_timeout_socket_replacement_closes_previous():
    from gr00t.policy.server_client import PolicyClient

    policy = PolicyClient.__new__(PolicyClient)
    policy._closed = True
    policy.socket = Mock()
    previous = policy.socket
    policy.context = Mock()
    policy.timeout_ms = 5
    policy.host = "127.0.0.1"
    policy.port = 5555
    policy._init_socket()
    previous.close.assert_called_once_with(linger=0)


def test_rpc_rotation_is_left_multiplication():
    from examples.dual_franka.dual_franka_robotiq_rpc_client import _pose_from_delta
    from scipy.spatial.transform import Rotation

    current = [1, 2, 3, 0.2, -0.1, 0.3]
    delta = [0.01, -0.02, 0.03, -0.1, 0.15, 0.04]
    result = _pose_from_delta(current, delta)
    expected = Rotation.from_rotvec(delta[3:]) * Rotation.from_rotvec(current[3:])
    np.testing.assert_allclose(result[:3], np.array(current[:3]) + delta[:3])
    np.testing.assert_allclose(
        Rotation.from_rotvec(result[3:]).as_matrix(), expected.as_matrix(), atol=1e-12
    )
