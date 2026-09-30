"""Control helpers copied from the existing StarVLA/OpenPI dual-Franka client."""

from contextlib import contextmanager
import logging
import time

import numpy as np


def check_state(state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Expected finite physical EE14 state")
    return state


def gripper_width(side):
    """Match recording/OpenPI: clamp calibrated openness, then convert to metres."""
    grip = side.get("gripper") or side.get("sensors", {}).get("robotiq")
    if not isinstance(grip, dict):
        raise ValueError("Missing gripper feedback")
    if grip.get("open_fraction") is not None:
        fraction = float(grip["open_fraction"])
    elif grip.get("position") is not None:
        opened = float(grip.get("open_position", 0))
        closed = float(grip.get("closed_position", 1))
        pos = float(grip["position"])
        if not np.isfinite([opened, closed, pos]).all() or abs(closed - opened) <= 1e-9:
            raise ValueError(
                f"Invalid gripper calibration: position={pos}, open={opened}, closed={closed}"
            )
        fraction = 1 - (pos - opened) / (closed - opened)
    elif grip.get("width") is not None:
        width = float(grip["width"])
        if not np.isfinite(width) or not 0 <= width <= 0.085001:
            raise ValueError(
                f"Invalid metric gripper feedback: width={width} m; expected [0, 0.085]"
            )
        return float(np.clip(width, 0, 0.085))
    else:
        raise ValueError("Missing calibrated gripper aperture")
    if not np.isfinite(fraction):
        raise ValueError(f"Non-finite gripper open fraction: {fraction}")
    return float(np.clip(fraction, 0, 1) * 0.085)


def rpc_state(full_obs, pose_from_side):
    return check_state(
        list(pose_from_side(full_obs, "left_arm"))
        + list(pose_from_side(full_obs, "right_arm"))
        + [gripper_width(full_obs[s]) for s in ["left_arm", "right_arm"]]
    )


def execution_commands(
    actions, execute_horizon=10, close_threshold_m=0.03, max_pos_delta=0.3, max_rot_delta=0.12
):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
        raise ValueError("Invalid action chunk")
    if not 0 < execute_horizon <= len(actions):
        raise ValueError("Invalid execution horizon")
    commands = actions[:execute_horizon].copy()
    for o in [0, 6]:
        commands[:, o : o + 3] = np.clip(commands[:, o : o + 3], -max_pos_delta, max_pos_delta)
        commands[:, o + 3 : o + 6] = np.clip(
            commands[:, o + 3 : o + 6], -max_rot_delta, max_rot_delta
        )
    widths = commands[:, 12:]
    commands[:, 12:] = np.where(widths < close_threshold_m, 0, np.clip(widths, 0, 0.085))
    return commands


def execute_chunk(rpc, actions, config, sleep=time.sleep):
    commands = execution_commands(
        actions,
        config.get("execute_horizon", 10),
        config.get("close_threshold_m", 0.03),
        config.get("max_pos_delta", 0.3),
        config.get("max_rot_delta", 0.12),
    )
    for c in commands:
        begin = time.perf_counter()
        checked_rpc(
            rpc.dual_robot_move_to_ee_pose,
            left_delta=c[:6],
            right_delta=c[6:12],
            delta=True,
            wait=False,
        )
        checked_rpc(
            rpc.left_gripper_goto,
            float(c[12]),
            speed=config.get("gripper_speed", 0.3),
            force=config.get("gripper_force", 10),
            blocking=False,
        )
        checked_rpc(
            rpc.right_gripper_goto,
            float(c[13]),
            speed=config.get("gripper_speed", 0.3),
            force=config.get("gripper_force", 10),
            blocking=False,
        )
        sleep(max(0, 1 / config.get("action_fps", 30) - (time.perf_counter() - begin)))
    return commands


def reset_robot(rpc, duration_sec):
    """Use the same home-then-open sequence as the OpenPI client."""
    print(f"[RESET] Both arms going home ({duration_sec:g}s), then opening grippers...", flush=True)
    for name, args, kwargs in [
        ("go_home", ("both",), {"duration_sec": duration_sec}),
        ("open_gripper", ("left_arm",), {}),
        ("open_gripper", ("right_arm",), {}),
    ]:
        result = getattr(rpc, name)(*args, **kwargs)
        if result is False or (isinstance(result, dict) and result.get("ok") is False):
            raise RuntimeError(f"Reset {name} failed: {result}")
    print("[RESET] Complete.", flush=True)


@contextmanager
def robot_session(rpc, execute, duration_sec):
    if not execute:
        yield
        return
    # Failed initial reset aborts inference without automatically retrying motion.
    reset_robot(rpc, duration_sec)
    try:
        yield
    except BaseException:
        try:
            reset_robot(rpc, duration_sec)
        except BaseException:
            logging.exception("Exit reset failed/interrupted; preserving original inference error")
        raise
    else:
        reset_robot(rpc, duration_sec)


def checked_rpc(method, *args, **kwargs):
    result = method(*args, **kwargs)
    if result is False or (isinstance(result, dict) and result.get("ok") is False):
        raise RuntimeError("Robot rejected command")
    return result
