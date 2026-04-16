#!/usr/bin/env python3
"""
Franka rehab ROM program with side-aware labeling, lower-home mapping,
and session logging.

Key features
- Choose patient profile and load settings once at the beginning of the session.
- Choose patient side for each exercise block.
- Choose exercise / ROM / cycles for each exercise block.
- Lower home pose changes by side and motion type.
- Upper home pose stays the same for both sides.
- Logs one summary row per cycle plus a downsampled time-series CSV.
"""

import argparse
import csv
import math
import os
import re
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from pylibfranka import CartesianPose, CartesianVelocities, ControllerMode, RealtimeConfig, Robot

# ---------------------------
# Rehab geometry
# ---------------------------
LOWER_HOME_EXT_LEFT_INT_RIGHT = {
    "x_rot_deg": -110.0,
    "y_rot_deg": 0.0,
    "z_rot_deg": 0.0,
    "x_mm": 150.0,
    "y_mm": 580.0,
    "z_mm": 280.0,
}

LOWER_HOME_EXT_RIGHT_INT_LEFT = {
    "x_rot_deg": -110.0,
    "y_rot_deg": 0.0,
    "z_rot_deg": 90.0,
    "x_mm": 150.0,
    "y_mm": 580.0,
    "z_mm": 280.0,
}

UPPER_HOME = {
    "x_rot_deg": 180.0,
    "y_rot_deg": 0.0,
    "z_rot_deg": 90.0,
    "x_mm": 0.0,
    "y_mm": 600.0,
    "z_mm": 485.0,
}

ROM_LEVELS_DEG = {
    "External Rotation Lower": [25.0, 40.0, 55.0, 70.0, 85.0],
    "External Rotation Upper": [30.0, 45.0, 60.0, 75.0, 90.0],
    "Internal Rotation Lower": [20.0, 30.0, 40.0, 50.0, 60.0],
    "Internal Rotation Upper": [15.0, 25.0, 35.0, 40.0, 45.0],
}

TIME_SERIES_LOG_INTERVAL_S = 0.05


@dataclass
class MotionSpec:
    name: str
    axis: str
    sign: float
    levels_deg: List[float]
    position: str  # "upper" or "lower"
    motion_family: str  # "external" or "internal"


@dataclass
class SoftSafetyLimits:
    max_force_n: float
    max_torque_nm: float
    max_joint_torque_nm: float
    consecutive_samples: int


@dataclass
class RehabProfile:
    label: str
    outbound_time_scale: float
    return_time_scale: float
    soft_return_time_scale: float
    force_scale: float
    torque_scale: float
    joint_torque_scale: float


@dataclass
class LoadSettings:
    label: str
    use_payload_compensation: bool
    payload_mass_kg: float
    payload_com_m: Tuple[float, float, float]
    payload_inertia: Tuple[float, ...]
    profile: RehabProfile
    baseline_force_n: float = 0.0
    baseline_torque_nm: float = 0.0
    baseline_joint_torque_nm: float = 0.0


@dataclass
class SessionInfo:
    session_id: str
    patient_id: str
    patient_side: str
    patient_profile_label: str
    load_label: str
    session_dir: str
    summary_csv_path: str
    timeseries_csv_path: str


@dataclass
class CycleResult:
    completed_normally: bool
    soft_stop_reason: str
    peak_force_n: float
    peak_torque_nm: float
    peak_joint_torque_nm: float
    samples_logged: int


MOTIONS = [
    MotionSpec(
        name="External Rotation Lower",
        axis="y",
        sign=+1.0,
        levels_deg=ROM_LEVELS_DEG["External Rotation Lower"],
        position="lower",
        motion_family="external",
    ),
    MotionSpec(
        name="External Rotation Upper",
        axis="x",
        sign=+1.0,
        levels_deg=ROM_LEVELS_DEG["External Rotation Upper"],
        position="upper",
        motion_family="external",
    ),
    MotionSpec(
        name="Internal Rotation Lower",
        axis="y",
        sign=-1.0,
        levels_deg=ROM_LEVELS_DEG["Internal Rotation Lower"],
        position="lower",
        motion_family="internal",
    ),
    MotionSpec(
        name="Internal Rotation Upper",
        axis="x",
        sign=-1.0,
        levels_deg=ROM_LEVELS_DEG["Internal Rotation Upper"],
        position="upper",
        motion_family="internal",
    ),
]

# ---------------------------
# Base robot parameters
# ---------------------------
DEFAULT_CARTESIAN_IMPEDANCE = [2200.0, 2200.0, 2600.0, 160.0, 160.0, 160.0]
DEFAULT_JOINT_IMPEDANCE = [2200.0, 2200.0, 2200.0, 1800.0, 1800.0, 1400.0, 1400.0]
LOWER_TORQUE_THRESHOLDS = [35.0, 35.0, 30.0, 30.0, 25.0, 20.0, 18.0]
UPPER_TORQUE_THRESHOLDS = [35.0, 35.0, 30.0, 30.0, 25.0, 20.0, 18.0]
LOWER_FORCE_THRESHOLDS = [35.0, 35.0, 35.0, 30.0, 30.0, 30.0]
UPPER_FORCE_THRESHOLDS = [35.0, 35.0, 35.0, 30.0, 30.0, 30.0]
ZERO_INERTIA = (0.0,) * 9

ADJUSTMENT_JOINT_IMPEDANCE = [250.0, 250.0, 250.0, 180.0, 180.0, 120.0, 120.0]
ADJUSTMENT_CARTESIAN_IMPEDANCE = [150.0, 150.0, 150.0, 20.0, 20.0, 20.0]
ADJUSTMENT_LOWER_TORQUE_THRESHOLDS = [60.0, 60.0, 50.0, 50.0, 40.0, 35.0, 30.0]
ADJUSTMENT_UPPER_TORQUE_THRESHOLDS = [60.0, 60.0, 50.0, 50.0, 40.0, 35.0, 30.0]
ADJUSTMENT_LOWER_FORCE_THRESHOLDS = [45.0, 45.0, 45.0, 35.0, 35.0, 35.0]
ADJUSTMENT_UPPER_FORCE_THRESHOLDS = [45.0, 45.0, 45.0, 35.0, 35.0, 35.0]

CALIBRATION_PROFILE_LIGHT = RehabProfile(
    label="Auto-calibrated load: lighter supported load",
    outbound_time_scale=1.00,
    return_time_scale=1.00,
    soft_return_time_scale=1.00,
    force_scale=1.00,
    torque_scale=1.00,
    joint_torque_scale=1.00,
)

CALIBRATION_PROFILE_MEDIUM = RehabProfile(
    label="Auto-calibrated load: medium supported load",
    outbound_time_scale=1.10,
    return_time_scale=1.05,
    soft_return_time_scale=1.10,
    force_scale=1.10,
    torque_scale=1.10,
    joint_torque_scale=1.10,
)

CALIBRATION_PROFILE_HEAVY = RehabProfile(
    label="Auto-calibrated load: heavier supported load",
    outbound_time_scale=1.20,
    return_time_scale=1.10,
    soft_return_time_scale=1.20,
    force_scale=1.20,
    torque_scale=1.20,
    joint_torque_scale=1.20,
)



# ---------------------------
# Math helpers
# ---------------------------
def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rot_x(rad: float) -> np.ndarray:
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(rad: float) -> np.ndarray:
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(rad: float) -> np.ndarray:
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_xyz_deg_to_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    return rot_z(deg2rad(z_deg)) @ rot_y(deg2rad(y_deg)) @ rot_x(deg2rad(x_deg))


def matrix_to_pose_16(R: np.ndarray, t_m: np.ndarray) -> List[float]:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_m
    return T.T.reshape(-1).tolist()


def build_home_pose(home: Dict[str, float]) -> np.ndarray:
    R = euler_xyz_deg_to_matrix(home["x_rot_deg"], home["y_rot_deg"], home["z_rot_deg"])
    t = np.array([home["x_mm"] / 1000.0, home["y_mm"] / 1000.0, home["z_mm"] / 1000.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def minimum_jerk_scalar(s: float) -> float:
    s = max(0.0, min(1.0, s))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def minimum_jerk_derivative(s: float) -> float:
    s = max(0.0, min(1.0, s))
    return 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4


def interpolate_pose(T_start: np.ndarray, T_goal: np.ndarray, s: float) -> np.ndarray:
    alpha = minimum_jerk_scalar(s)
    p = (1.0 - alpha) * T_start[:3, 3] + alpha * T_goal[:3, 3]
    R_blend = (1.0 - alpha) * T_start[:3, :3] + alpha * T_goal[:3, :3]
    U, _, Vt = np.linalg.svd(R_blend)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def pose_from_state(robot_state) -> np.ndarray:
    return np.array(robot_state.O_T_EE, dtype=float).reshape((4, 4)).T


def extract_interaction_metrics(robot_state) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    wrench = np.array(robot_state.O_F_ext_hat_K, dtype=float)
    joint_torque = np.array(robot_state.tau_ext_hat_filtered, dtype=float)
    force_mag = float(np.linalg.norm(wrench[:3]))
    torque_mag = float(np.linalg.norm(wrench[3:]))
    joint_torque_mag = float(np.max(np.abs(joint_torque)))
    return force_mag, torque_mag, joint_torque_mag, wrench, joint_torque


# ---------------------------
# Side / labeling helpers
# ---------------------------
def format_motion_label(motion: MotionSpec, patient_side: str) -> str:
    return f"{motion.name} - {patient_side}"


def get_home_dict_for_motion(motion: MotionSpec, patient_side: str) -> Dict[str, float]:
    if motion.position == "upper":
        return UPPER_HOME

    side = patient_side.lower()
    if motion.motion_family == "external":
        if side == "left":
            return LOWER_HOME_EXT_LEFT_INT_RIGHT
        return LOWER_HOME_EXT_RIGHT_INT_LEFT

    if side == "right":
        return LOWER_HOME_EXT_LEFT_INT_RIGHT
    return LOWER_HOME_EXT_RIGHT_INT_LEFT


def get_signed_angle_deg(motion: MotionSpec, patient_side: str, rom_deg: float) -> float:
    side = patient_side.lower()

    if motion.name == "External Rotation Lower":
        return -rom_deg if side == "left" else +rom_deg

    if motion.name == "Internal Rotation Lower":
        return +rom_deg if side == "left" else -rom_deg

    if motion.name == "External Rotation Upper":
        return +rom_deg if side == "left" else -rom_deg

    if motion.name == "Internal Rotation Upper":
        return -rom_deg if side == "left" else +rom_deg

    return motion.sign * rom_deg


# ---------------------------
# Safety
# ---------------------------
class SafetyMonitor:
    def __init__(
        self,
        limits: SoftSafetyLimits,
        baseline_force_n: float = 0.0,
        baseline_torque_nm: float = 0.0,
        baseline_joint_torque_nm: float = 0.0,
        grace_time_s: float = 0.20,
        clear_ratio: float = 0.85,
    ):
        self.limits = limits
        self.baseline_force_n = max(0.0, baseline_force_n)
        self.baseline_torque_nm = max(0.0, baseline_torque_nm)
        self.baseline_joint_torque_nm = max(0.0, baseline_joint_torque_nm)
        self.grace_time_s = max(0.0, grace_time_s)
        self.clear_ratio = min(max(clear_ratio, 0.0), 1.0)
        self.counter = 0

    def reset(self) -> None:
        self.counter = 0

    def check(self, robot_state, elapsed_s: float = 0.0) -> Tuple[bool, str]:
        force_mag, torque_mag, joint_torque_mag, _, _ = extract_interaction_metrics(robot_state)

        if elapsed_s < self.grace_time_s:
            self.counter = 0
            return False, ""

        force_delta = max(0.0, force_mag - self.baseline_force_n)
        torque_delta = max(0.0, torque_mag - self.baseline_torque_nm)
        joint_torque_delta = max(0.0, joint_torque_mag - self.baseline_joint_torque_nm)

        exceeded = []
        if force_delta >= self.limits.max_force_n:
            exceeded.append(f"d|F|={force_delta:.2f} N (raw {force_mag:.2f})")
        if torque_delta >= self.limits.max_torque_nm:
            exceeded.append(f"d|Tau_wrench|={torque_delta:.2f} Nm (raw {torque_mag:.2f})")
        if joint_torque_delta >= self.limits.max_joint_torque_nm:
            exceeded.append(f"dmax|Tau_joint|={joint_torque_delta:.2f} Nm (raw {joint_torque_mag:.2f})")

        if exceeded:
            self.counter += 1
        else:
            clear_force = self.limits.max_force_n * self.clear_ratio
            clear_torque = self.limits.max_torque_nm * self.clear_ratio
            clear_joint = self.limits.max_joint_torque_nm * self.clear_ratio
            if (
                force_delta < clear_force
                and torque_delta < clear_torque
                and joint_torque_delta < clear_joint
            ):
                self.counter = 0

        if self.counter >= self.limits.consecutive_samples:
            return True, "SOFT SAFETY EVENT: " + ", ".join(exceeded)
        return False, ""


# ---------------------------
# Robot helpers
# ---------------------------
def set_default_behaviour(robot: Robot) -> None:
    robot.set_collision_behavior(
        LOWER_TORQUE_THRESHOLDS,
        UPPER_TORQUE_THRESHOLDS,
        LOWER_FORCE_THRESHOLDS,
        UPPER_FORCE_THRESHOLDS,
    )
    robot.set_joint_impedance(DEFAULT_JOINT_IMPEDANCE)
    robot.set_cartesian_impedance(DEFAULT_CARTESIAN_IMPEDANCE)


def set_adjustment_behaviour(robot: Robot) -> None:
    robot.set_collision_behavior(
        ADJUSTMENT_LOWER_TORQUE_THRESHOLDS,
        ADJUSTMENT_UPPER_TORQUE_THRESHOLDS,
        ADJUSTMENT_LOWER_FORCE_THRESHOLDS,
        ADJUSTMENT_UPPER_FORCE_THRESHOLDS,
    )
    robot.set_joint_impedance(ADJUSTMENT_JOINT_IMPEDANCE)
    robot.set_cartesian_impedance(ADJUSTMENT_CARTESIAN_IMPEDANCE)


def wait_for_mechanical_settle(robot: Robot, settle_time_s: float = 0.75, poll_period_s: float = 0.02) -> None:
    """Wait until the robot/bracing setup has had a short chance to settle."""
    start = time.time()
    while time.time() - start < settle_time_s:
        robot.read_once()
        time.sleep(poll_period_s)


def run_guided_adjustment_mode(robot: Robot, prompt: str) -> None:
    print("\n" + "-" * 72)
    print("GUIDED ADJUSTMENT MODE")
    print("The robot is now in a lighter supported mode so you can position the brace.")
    print("Move slowly and keep a hand on the brace at all times.")
    print("Press ENTER when positioning is complete.")
    print("-" * 72)

    # Important: do not start an external Cartesian motion generator here.
    # We only soften the robot's internal hold behavior, then let the operator
    # position the brace and confirm when ready.
    set_adjustment_behaviour(robot)
    input(prompt)
    wait_for_mechanical_settle(robot, settle_time_s=0.75, poll_period_s=0.02)


def estimate_supported_load_while_holding(
    robot: Robot, sample_count: int = 120, sample_period_s: float = 0.02
) -> Tuple[float, float, float, float]:
    # Read-only measurement step: do not start any external Cartesian control.
    set_adjustment_behaviour(robot)
    wait_for_mechanical_settle(robot, settle_time_s=0.75, poll_period_s=0.02)

    force_vectors = []
    torque_vectors = []
    joint_torque_max_values = []

    for _ in range(sample_count):
        robot_state = robot.read_once()
        wrench = np.array(robot_state.O_F_ext_hat_K, dtype=float)
        joint_torque = np.array(robot_state.tau_ext_hat_filtered, dtype=float)
        force_vectors.append(wrench[:3])
        torque_vectors.append(wrench[3:])
        joint_torque_max_values.append(float(np.max(np.abs(joint_torque))))
        time.sleep(sample_period_s)

    mean_force_vector = np.mean(np.array(force_vectors), axis=0)
    mean_torque_vector = np.mean(np.array(torque_vectors), axis=0)
    avg_force_n = float(np.linalg.norm(mean_force_vector))
    avg_torque_nm = float(np.linalg.norm(mean_torque_vector))
    avg_joint_torque_nm = float(np.mean(joint_torque_max_values))
    effective_mass_kg = avg_force_n / 9.81
    return effective_mass_kg, avg_force_n, avg_torque_nm, avg_joint_torque_nm


def move_cartesian_pose(control, target_T: np.ndarray, move_time: float, hold_time: float = 0.25) -> None:
    robot_state, _ = control.readOnce()
    current_T = pose_from_state(robot_state)
    elapsed = 0.0
    while True:
        robot_state, period = control.readOnce()
        elapsed += period.to_sec()
        s = min(elapsed / move_time, 1.0)
        T_cmd = interpolate_pose(current_T, target_T, s)
        cmd = CartesianPose(matrix_to_pose_16(T_cmd[:3, :3], T_cmd[:3, 3]))
        if elapsed >= move_time + hold_time:
            cmd.motion_finished = True
            control.writeOnce(cmd)
            break
        control.writeOnce(cmd)


def move_to_home(robot: Robot, home_T: np.ndarray, move_time: float, dwell_time: float) -> bool:
    control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
    try:
        move_cartesian_pose(control, home_T, move_time=move_time, hold_time=dwell_time)
        return True
    except Exception as exc:
        print(f"Home move failed: {exc}")
        return False


def local_axis_in_base(R: np.ndarray, axis: str) -> np.ndarray:
    if axis == "x":
        v = R[:, 0]
    elif axis == "y":
        v = R[:, 1]
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    n = np.linalg.norm(v)
    return np.array([1.0, 0.0, 0.0]) if n < 1e-12 else v / n


def ramp_down_velocity(
    control,
    axis: str,
    omega_prev: float,
    log_callback: Optional[Callable] = None,
    base_context: Optional[Dict[str, object]] = None,
    duration_s: float = 0.8,
) -> None:
    elapsed = 0.0
    last_log_time = -1e9
    while elapsed < duration_s:
        robot_state, period = control.readOnce()
        dt = period.to_sec()
        elapsed += dt
        s = min(elapsed / duration_s, 1.0)
        fade = 1.0 - minimum_jerk_scalar(s)
        R = pose_from_state(robot_state)[:3, :3]
        axis_base = local_axis_in_base(R, axis)
        omega_vec = axis_base * (omega_prev * fade)
        cmd = CartesianVelocities([0.0, 0.0, 0.0, omega_vec[0], omega_vec[1], omega_vec[2]])
        control.writeOnce(cmd)
        if log_callback is not None and elapsed - last_log_time >= TIME_SERIES_LOG_INTERVAL_S:
            force_mag, torque_mag, joint_torque_mag, wrench, joint_torque = extract_interaction_metrics(robot_state)
            T = pose_from_state(robot_state)
            row = {
                **(base_context or {}),
                "phase": "soft_stop_ramp_down",
                "elapsed_s": round(float(elapsed), 4),
                "commanded_omega_rad_s": round(float(np.linalg.norm(omega_vec)), 6),
                "ee_x_m": round(float(T[0, 3]), 6),
                "ee_y_m": round(float(T[1, 3]), 6),
                "ee_z_m": round(float(T[2, 3]), 6),
                "force_mag_n": round(force_mag, 6),
                "torque_mag_nm": round(torque_mag, 6),
                "joint_torque_mag_nm": round(joint_torque_mag, 6),
                "wrench_fx": round(float(wrench[0]), 6),
                "wrench_fy": round(float(wrench[1]), 6),
                "wrench_fz": round(float(wrench[2]), 6),
                "wrench_tx": round(float(wrench[3]), 6),
                "wrench_ty": round(float(wrench[4]), 6),
                "wrench_tz": round(float(wrench[5]), 6),
                "joint_tau_max": round(joint_torque_mag, 6),
            }
            log_callback(row)
            last_log_time = elapsed
    zero_cmd = CartesianVelocities([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    zero_cmd.motion_finished = True
    control.writeOnce(zero_cmd)


def rotate_outbound_with_softstop(
    robot: Robot,
    axis: str,
    signed_angle_deg: float,
    outbound_time: float,
    safety_monitor: SafetyMonitor,
    log_callback: Optional[Callable] = None,
    base_context: Optional[Dict[str, object]] = None,
) -> CycleResult:
    control = robot.start_cartesian_velocity_control(ControllerMode.CartesianImpedance)
    theta_total = deg2rad(abs(signed_angle_deg))
    sign = 1.0 if signed_angle_deg >= 0.0 else -1.0
    time_elapsed = 0.0
    last_omega = 0.0
    last_log_time = -1e9
    peak_force_n = 0.0
    peak_torque_nm = 0.0
    peak_joint_torque_nm = 0.0
    samples_logged = 0

    try:
        while True:
            robot_state, period = control.readOnce()
            dt = period.to_sec()
            time_elapsed += dt
            force_mag, torque_mag, joint_torque_mag, wrench, joint_torque = extract_interaction_metrics(robot_state)
            peak_force_n = max(peak_force_n, force_mag)
            peak_torque_nm = max(peak_torque_nm, torque_mag)
            peak_joint_torque_nm = max(peak_joint_torque_nm, joint_torque_mag)

            hit, msg = safety_monitor.check(robot_state, time_elapsed)
            if log_callback is not None and time_elapsed - last_log_time >= TIME_SERIES_LOG_INTERVAL_S:
                T = pose_from_state(robot_state)
                row = {
                    **(base_context or {}),
                    "phase": "outbound",
                    "elapsed_s": round(float(time_elapsed), 4),
                    "commanded_omega_rad_s": round(float(last_omega), 6),
                    "ee_x_m": round(float(T[0, 3]), 6),
                    "ee_y_m": round(float(T[1, 3]), 6),
                    "ee_z_m": round(float(T[2, 3]), 6),
                    "force_mag_n": round(force_mag, 6),
                    "torque_mag_nm": round(torque_mag, 6),
                    "joint_torque_mag_nm": round(joint_torque_mag, 6),
                    "wrench_fx": round(float(wrench[0]), 6),
                    "wrench_fy": round(float(wrench[1]), 6),
                    "wrench_fz": round(float(wrench[2]), 6),
                    "wrench_tx": round(float(wrench[3]), 6),
                    "wrench_ty": round(float(wrench[4]), 6),
                    "wrench_tz": round(float(wrench[5]), 6),
                    "joint_tau_max": round(joint_torque_mag, 6),
                    "safety_counter": safety_monitor.counter,
                    "safety_hit": int(hit),
                }
                log_callback(row)
                last_log_time = time_elapsed
                samples_logged += 1

            if hit:
                print(msg)
                print("Soft stop triggered. Ramping down motion smoothly...")
                try:
                    ramp_down_velocity(
                        control,
                        axis=axis,
                        omega_prev=last_omega,
                        log_callback=log_callback,
                        base_context=base_context,
                        duration_s=1.0,
                    )
                except Exception as exc:
                    print(f"Ramp-down during soft stop failed: {exc}")
                return CycleResult(False, msg, peak_force_n, peak_torque_nm, peak_joint_torque_nm, samples_logged)

            s = min(time_elapsed / outbound_time, 1.0)
            sdot = minimum_jerk_derivative(s) / outbound_time
            omega_mag = theta_total * sdot * sign
            last_omega = omega_mag
            R = pose_from_state(robot_state)[:3, :3]
            axis_base = local_axis_in_base(R, axis)
            omega_vec = axis_base * omega_mag
            cmd = CartesianVelocities([0.0, 0.0, 0.0, omega_vec[0], omega_vec[1], omega_vec[2]])
            if time_elapsed >= outbound_time:
                cmd.motion_finished = True
                control.writeOnce(cmd)
                return CycleResult(True, "", peak_force_n, peak_torque_nm, peak_joint_torque_nm, samples_logged)
            control.writeOnce(cmd)
    finally:
        pass


def run_one_cycle_safe(
    robot: Robot,
    home_T: np.ndarray,
    axis: str,
    signed_angle_deg: float,
    outbound_time: float,
    return_time: float,
    dwell_time: float,
    soft_return_time: float,
    safety_monitor: SafetyMonitor,
    log_callback: Optional[Callable] = None,
    base_context: Optional[Dict[str, object]] = None,
) -> CycleResult:
    result = rotate_outbound_with_softstop(
        robot=robot,
        axis=axis,
        signed_angle_deg=signed_angle_deg,
        outbound_time=outbound_time,
        safety_monitor=safety_monitor,
        log_callback=log_callback,
        base_context=base_context,
    )
    if not result.completed_normally:
        print("Starting fresh slow return-to-home after soft stop...")
        try:
            robot.automatic_error_recovery()
        except Exception as exc:
            print(f"Automatic error recovery failed: {exc}")
            return result
        returned = move_to_home(robot, home_T, move_time=soft_return_time, dwell_time=dwell_time)
        print("Returned to home after soft stop." if returned else "Could not return home after soft stop.")
        return result

    returned = move_to_home(robot, home_T, move_time=soft_return_time, dwell_time=dwell_time)
    if not returned:
        print("Could not return home after normal outbound motion.")
        return CycleResult(False, "Home return failed after outbound motion.", result.peak_force_n, result.peak_torque_nm, result.peak_joint_torque_nm, result.samples_logged)
    return result


# ---------------------------
# UI helpers
# ---------------------------
def ask_int(prompt: str, min_value: int = 1, max_value: Optional[int] = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit():
            value = int(raw)
            if value >= min_value and (max_value is None or value <= max_value):
                return value
        print(f"Please enter an integer from {min_value} to {max_value}." if max_value is not None else f"Please enter an integer >= {min_value}.")


def ask_float(prompt: str, min_value: Optional[float] = None) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if min_value is not None and value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return value


def ask_text(prompt: str, default: str = "") -> str:
    raw = input(prompt).strip()
    return raw if raw else default


def ask_yes_no(prompt: str) -> bool:
    while True:
        reply = input(prompt).strip().lower()
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        print("Please enter y or n.")


def choose_from_menu(title: str, options: List[str]) -> int:
    print("\n" + title)
    for i, item in enumerate(options, start=1):
        print(f"  {i}. {item}")
    return ask_int("Enter number: ", 1, len(options)) - 1


def select_patient_side() -> str:
    idx = choose_from_menu("Select patient side for this exercise:", ["Left", "Right"])
    return ["Left", "Right"][idx]


def select_motion() -> MotionSpec:
    idx = choose_from_menu("Select exercise:", [m.name for m in MOTIONS])
    return MOTIONS[idx]


def select_rom_level(motion: MotionSpec) -> float:
    idx = choose_from_menu(f"Select ROM for {motion.name}:", [f"{deg:.1f} deg" for deg in motion.levels_deg])
    return motion.levels_deg[idx]


def select_cycles() -> int:
    return ask_int("Enter number of cycles: ", 1)


def estimate_supported_load(robot: Robot, sample_count: int = 120, sample_period_s: float = 0.02) -> Tuple[float, float, float, float]:
    force_vectors = []
    torque_vectors = []
    joint_torque_max_values = []

    for _ in range(sample_count):
        state = robot.read_once()
        wrench = np.array(state.O_F_ext_hat_K, dtype=float)
        joint_torque = np.array(state.tau_ext_hat_filtered, dtype=float)
        force_vectors.append(wrench[:3])
        torque_vectors.append(wrench[3:])
        joint_torque_max_values.append(float(np.max(np.abs(joint_torque))))
        time.sleep(sample_period_s)

    mean_force_vector = np.mean(np.array(force_vectors), axis=0)
    mean_torque_vector = np.mean(np.array(torque_vectors), axis=0)
    avg_force_n = float(np.linalg.norm(mean_force_vector))
    avg_torque_nm = float(np.linalg.norm(mean_torque_vector))
    avg_joint_torque_nm = float(np.mean(joint_torque_max_values))
    effective_mass_kg = avg_force_n / 9.81
    return effective_mass_kg, avg_force_n, avg_torque_nm, avg_joint_torque_nm



def build_auto_calibrated_profile(effective_mass_kg: float) -> RehabProfile:
    if effective_mass_kg < 1.5:
        base = CALIBRATION_PROFILE_LIGHT
        bucket = "lighter"
    elif effective_mass_kg < 3.0:
        base = CALIBRATION_PROFILE_MEDIUM
        bucket = "medium"
    else:
        base = CALIBRATION_PROFILE_HEAVY
        bucket = "heavier"

    return RehabProfile(
        label=f"Auto-calibrated {bucket} supported load ({effective_mass_kg:.2f} kg est.)",
        outbound_time_scale=base.outbound_time_scale,
        return_time_scale=base.return_time_scale,
        soft_return_time_scale=base.soft_return_time_scale,
        force_scale=base.force_scale,
        torque_scale=base.torque_scale,
        joint_torque_scale=base.joint_torque_scale,
    )



def run_startup_calibration_with_retry(robot: Robot) -> LoadSettings:
    while True:
        try:
            return run_startup_calibration(robot)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print("\nCalibration failed.")
            print(f"Reason: {exc}")
            print("Try again after checking that the arm and brace are settled and the network connection is stable.")
            try:
                robot.automatic_error_recovery()
            except Exception as recovery_exc:
                print(f"Automatic error recovery failed: {recovery_exc}")
            if not ask_yes_no("Redo startup calibration? [y/n]: "):
                raise RuntimeError(f"Startup calibration failed: {exc}") from exc
            print("Retrying startup calibration...")


def run_startup_calibration(robot: Robot) -> LoadSettings:
    brace_mass_kg = 0.53
    brace_com_m = (0.0, 0.0, 0.0)

    print("\n" + "=" * 72)
    print("STARTUP LOAD CALIBRATION")
    print("This step measures the supported load with the brace and arm attached.")
    print("1. Strap the arm into the brace.")
    print("2. Put the robot and brace in the normal starting setup.")
    print("3. Ask the person to relax and not push or pull.")
    print("4. Keep the robot still during measurement.")
    print("=" * 72)
    run_guided_adjustment_mode(robot, "Press ENTER when the brace and arm are ready for measurement... ")

    print("Measuring for about 2 to 3 seconds with the robot in a soft static hold. Please keep everything still...")
    effective_mass_kg, avg_force_n, avg_torque_nm, avg_joint_torque_nm = estimate_supported_load_while_holding(robot)
    profile = build_auto_calibrated_profile(effective_mass_kg)

    print("Calibration complete.")
    print(f"Estimated supported load: {effective_mass_kg:.2f} kg")
    print(f"Average external force:   {avg_force_n:.2f} N")
    print(f"Average wrench torque:    {avg_torque_nm:.2f} Nm")
    print(f"Average joint torque:     {avg_joint_torque_nm:.2f} Nm")
    print(f"Using profile:           {profile.label}")
    print(f"Using brace payload:     {brace_mass_kg:.2f} kg preset")

    return LoadSettings(
        label=f"Auto-calibrated brace + arm setup ({effective_mass_kg:.2f} kg est.)",
        use_payload_compensation=True,
        payload_mass_kg=brace_mass_kg,
        payload_com_m=brace_com_m,
        payload_inertia=ZERO_INERTIA,
        profile=profile,
        baseline_force_n=avg_force_n,
        baseline_torque_nm=avg_torque_nm,
        baseline_joint_torque_nm=avg_joint_torque_nm,
    )


def confirm_reposition(robot: Robot, previous_label: str, next_label: str) -> None:
    print("\n" + "!" * 72)
    print("SAFETY WARNING")
    print(f"Transitioning from '{previous_label}' to '{next_label}'.")
    print("Unstrap and reposition the user's arm before continuing.")
    print("Verify straps, alignment, comfort, and clearance.")
    run_guided_adjustment_mode(robot, "Press ENTER only after repositioning is complete... ")





def scaled_safety_limits(base: SoftSafetyLimits, profile: RehabProfile) -> SoftSafetyLimits:
    return SoftSafetyLimits(
        max_force_n=max(4.0, base.max_force_n * profile.force_scale),
        max_torque_nm=max(2.0, base.max_torque_nm * profile.torque_scale),
        max_joint_torque_nm=max(2.0, base.max_joint_torque_nm * profile.joint_torque_scale),
        consecutive_samples=base.consecutive_samples,
    )


def apply_load_if_supported(robot: Robot, load_settings: LoadSettings) -> bool:
    if not load_settings.use_payload_compensation:
        print("Payload compensation: not applied.")
        return False
    try:
        robot.set_load(load_settings.payload_mass_kg, list(load_settings.payload_com_m), list(load_settings.payload_inertia))
        print(f"Payload compensation applied: mass={load_settings.payload_mass_kg:.3f} kg, COM={load_settings.payload_com_m} m")
        return True
    except AttributeError:
        print("WARNING: robot.set_load(...) is not available in this pylibfranka build. Continuing without payload compensation.")
        return False
    except Exception as exc:
        print(f"WARNING: could not apply payload compensation ({exc}). Continuing without it.")
        return False


# ---------------------------
# Logging helpers
# ---------------------------
def sanitize_for_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "session"


def create_session_info(patient_id: str, patient_profile: RehabProfile, load_settings: LoadSettings) -> SessionInfo:
    session_id = time.strftime("%Y%m%d_%H%M%S")
    patient_stub = sanitize_for_filename(patient_id)[:24]
    session_dir = Path.cwd() / "session_logs" / f"session_{session_id}_{patient_stub}"
    session_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = str(session_dir / "session_summary.csv")
    timeseries_csv_path = str(session_dir / "session_timeseries.csv")
    return SessionInfo(
        session_id=session_id,
        patient_id=patient_id,
        patient_side="Per exercise",
        patient_profile_label=patient_profile.label,
        load_label=load_settings.label,
        session_dir=str(session_dir),
        summary_csv_path=summary_csv_path,
        timeseries_csv_path=timeseries_csv_path,
    )


def init_summary_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session_id", "patient_id", "patient_side", "exercise_label", "exercise_name",
            "position", "motion_family", "cycle_number", "cycles_requested", "rom_deg",
            "patient_profile_label", "load_label", "payload_comp_applied", "outbound_time_s", "return_time_s",
            "soft_return_time_s", "soft_stop", "soft_stop_reason", "peak_force_n", "peak_torque_nm",
            "peak_joint_torque_nm", "completed_normally", "timestamp_unix"
        ])
        writer.writeheader()


def init_timeseries_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session_id", "patient_id", "patient_side", "exercise_label", "cycle_number",
            "phase", "elapsed_s", "commanded_omega_rad_s", "ee_x_m", "ee_y_m", "ee_z_m", "force_mag_n",
            "torque_mag_nm", "joint_torque_mag_nm", "wrench_fx", "wrench_fy", "wrench_fz", "wrench_tx",
            "wrench_ty", "wrench_tz", "joint_tau_max", "safety_counter", "safety_hit"
        ])
        writer.writeheader()


def append_summary_row(path: str, row: Dict[str, object]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)


def append_timeseries_row(path: str, fieldnames: List[str], row: Dict[str, object]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def print_session_summary_header(session_info: SessionInfo) -> None:
    print("\n" + "=" * 72)
    print("SESSION SETUP")
    print(f"Patient ID:          {session_info.patient_id}")
    print("Patient side:        Selected for each exercise")
    print(f"Auto load setup:     {session_info.patient_profile_label}")
    print(f"Load mode:           {session_info.load_label}")
    print(f"Logs folder:         {session_info.session_dir}")
    print("=" * 72)


def print_run_summary(
    session_info: SessionInfo,
    patient_side: str,
    display_label: str,
    motion: MotionSpec,
    home_dict: Dict[str, float],
    rom_deg: float,
    cycles: int,
    load_settings: LoadSettings,
    outbound_time: float,
    return_time: float,
    soft_return_time: float,
    safety_limits: SoftSafetyLimits,
) -> None:
    print("\n" + "=" * 72)
    print("RUN SUMMARY")
    print(f"Patient:            {session_info.patient_id}")
    print(f"Side:               {patient_side}")
    print(f"Exercise:           {display_label}")
    print(f"ROM:                {rom_deg:.1f} deg")
    print(f"Cycles:             {cycles}")
    print(f"Load mode:          {load_settings.label}")
    print(f"Axis:               local {motion.axis.upper()} axis")
    print(f"Home type:          {motion.position.upper()}")
    print(f"Home XYZ mm:        ({home_dict['x_mm']:.1f}, {home_dict['y_mm']:.1f}, {home_dict['z_mm']:.1f})")
    print(f"Home rot deg:       ({home_dict['x_rot_deg']:.1f}, {home_dict['y_rot_deg']:.1f}, {home_dict['z_rot_deg']:.1f})")
    print(f"Outbound time:      {outbound_time:.2f} s")
    print(f"Return time:        {return_time:.2f} s")
    print(f"Soft return time:   {soft_return_time:.2f} s")
    print(f"Soft-stop force:    {safety_limits.max_force_n:.2f} N")
    print(f"Soft-stop wrench:   {safety_limits.max_torque_nm:.2f} Nm")
    print(f"Soft-stop joint:    {safety_limits.max_joint_torque_nm:.2f} Nm")
    print(f"Consecutive hits:   {safety_limits.consecutive_samples}")
    print(f"Summary CSV:        {os.path.basename(session_info.summary_csv_path)}")
    print(f"Timeseries CSV:     {os.path.basename(session_info.timeseries_csv_path)}")
    print("=" * 72)


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Franka rehab ROM program with side-aware UI")
    parser.add_argument("--ip", type=str, required=True, help="Robot IP address")
    parser.add_argument("--home-time", type=float, default=6.0, help="Seconds to move to each home pose")
    parser.add_argument("--outbound-time", type=float, default=4.0, help="Seconds for rehab outbound rotation")
    parser.add_argument("--return-time", type=float, default=6.0, help="Seconds to move home after normal outbound")
    parser.add_argument("--soft-return-time", type=float, default=8.0, help="Seconds to move home after soft stop")
    parser.add_argument("--dwell-time", type=float, default=0.5, help="Seconds to pause at endpoints")
    parser.add_argument("--max-force-n", type=float, default=8.0, help="Base soft stop force threshold")
    parser.add_argument("--max-torque-nm", type=float, default=5.0, help="Base soft stop wrench torque threshold")
    parser.add_argument("--max-joint-torque-nm", type=float, default=5.0, help="Base soft stop joint torque threshold")
    parser.add_argument("--consecutive-samples", type=int, default=2, help="Samples above threshold before soft stop")
    args = parser.parse_args()

    base_safety = SoftSafetyLimits(args.max_force_n, args.max_torque_nm, args.max_joint_torque_nm, args.consecutive_samples)

    print("WARNING: This program will move the robot.")
    print("Keep the user stop button within reach.")
    input("Press Enter to continue... ")

    # Session-level prompts: asked once.
    patient_id = ask_text("Enter patient name or ID: ", "UnknownPatient")

    robot = None
    previous_label: Optional[str] = None
    session_info: Optional[SessionInfo] = None
    timeseries_fields = [
        "session_id", "patient_id", "patient_side", "exercise_label", "cycle_number",
        "phase", "elapsed_s", "commanded_omega_rad_s", "ee_x_m", "ee_y_m", "ee_z_m", "force_mag_n",
        "torque_mag_nm", "joint_torque_mag_nm", "wrench_fx", "wrench_fy", "wrench_fz", "wrench_tx",
        "wrench_ty", "wrench_tz", "joint_tau_max", "safety_counter", "safety_hit"
    ]

    try:
        robot = Robot(args.ip, RealtimeConfig.kIgnore)
        robot.automatic_error_recovery()
        set_default_behaviour(robot)

        load_settings = run_startup_calibration_with_retry(robot)
        patient_profile = load_settings.profile
        session_info = create_session_info(patient_id, patient_profile, load_settings)
        init_summary_csv(session_info.summary_csv_path)
        init_timeseries_csv(session_info.timeseries_csv_path)
        print_session_summary_header(session_info)

        payload_applied = apply_load_if_supported(robot, load_settings)

        while True:
            motion = select_motion()
            patient_side = select_patient_side()
            display_label = format_motion_label(motion, patient_side)
            rom_deg = select_rom_level(motion)
            cycles = select_cycles()

            if previous_label is not None:
                confirm_reposition(robot, previous_label, display_label)

            combined_profile = load_settings.profile
            safety_limits = scaled_safety_limits(base_safety, combined_profile)
            safety_monitor = SafetyMonitor(
                safety_limits,
                baseline_force_n=load_settings.baseline_force_n,
                baseline_torque_nm=load_settings.baseline_torque_nm,
                baseline_joint_torque_nm=load_settings.baseline_joint_torque_nm,
                grace_time_s=0.20,
                clear_ratio=0.85,
            )

            outbound_time = args.outbound_time * combined_profile.outbound_time_scale
            return_time = args.return_time * combined_profile.return_time_scale
            soft_return_time = args.soft_return_time * combined_profile.soft_return_time_scale

            set_default_behaviour(robot)
            if payload_applied:
                apply_load_if_supported(robot, load_settings)

            home_dict = get_home_dict_for_motion(motion, patient_side)
            home_T = build_home_pose(home_dict)
            signed_angle_deg = get_signed_angle_deg(motion, patient_side, rom_deg)

            print_run_summary(
                session_info=session_info,
                patient_side=patient_side,
                display_label=display_label,
                motion=motion,
                home_dict=home_dict,
                rom_deg=rom_deg,
                cycles=cycles,
                load_settings=load_settings,
                outbound_time=outbound_time,
                return_time=return_time,
                soft_return_time=soft_return_time,
                safety_limits=safety_limits,
            )

            print("Moving to home...")
            homed = move_to_home(robot, home_T, move_time=args.home_time, dwell_time=0.35)
            if not homed:
                print("Could not reach home.")
                if session_info is not None:
                    print(f"Logs saved in: {session_info.session_dir}")
                return 1

            print(f"At home for {display_label} ({rom_deg:.1f} deg).")
            run_guided_adjustment_mode(robot, "Press Enter to start movement... ")
            set_default_behaviour(robot)
            if payload_applied:
                apply_load_if_supported(robot, load_settings)
            safety_monitor.reset()

            completed_cycles = 0
            for cycle in range(1, cycles + 1):
                print(f"Cycle {cycle}/{cycles}")
                print("outbound")
                base_context = {
                    "session_id": session_info.session_id,
                    "patient_id": session_info.patient_id,
                    "patient_side": patient_side,
                    "exercise_label": display_label,
                    "cycle_number": cycle,
                }
                cycle_result = run_one_cycle_safe(
                    robot=robot,
                    home_T=home_T,
                    axis=motion.axis,
                    signed_angle_deg=signed_angle_deg,
                    outbound_time=outbound_time,
                    return_time=return_time,
                    dwell_time=args.dwell_time,
                    soft_return_time=soft_return_time,
                    safety_monitor=safety_monitor,
                    log_callback=lambda row: append_timeseries_row(session_info.timeseries_csv_path, timeseries_fields, row),
                    base_context=base_context,
                )

                append_summary_row(session_info.summary_csv_path, {
                    "session_id": session_info.session_id,
                    "patient_id": session_info.patient_id,
                    "patient_side": patient_side,
                    "exercise_label": display_label,
                    "exercise_name": motion.name,
                    "position": motion.position,
                    "motion_family": motion.motion_family,
                    "cycle_number": cycle,
                    "cycles_requested": cycles,
                    "rom_deg": round(rom_deg, 3),
                    "patient_profile_label": patient_profile.label,
                    "load_label": load_settings.label,
                    "payload_comp_applied": int(payload_applied),
                    "outbound_time_s": round(outbound_time, 4),
                    "return_time_s": round(return_time, 4),
                    "soft_return_time_s": round(soft_return_time, 4),
                    "soft_stop": int(not cycle_result.completed_normally),
                    "soft_stop_reason": cycle_result.soft_stop_reason,
                    "peak_force_n": round(cycle_result.peak_force_n, 6),
                    "peak_torque_nm": round(cycle_result.peak_torque_nm, 6),
                    "peak_joint_torque_nm": round(cycle_result.peak_joint_torque_nm, 6),
                    "completed_normally": int(cycle_result.completed_normally),
                    "timestamp_unix": round(time.time(), 6),
                })

                if not cycle_result.completed_normally:
                    print("Stopped due to safety event.")
                    break
                completed_cycles += 1

            previous_label = display_label
            print(f"Completed {completed_cycles} of {cycles} requested cycles for {display_label}.")

            if not ask_yes_no("Run another exercise? [y/n]: "):
                print("Program complete.")
                if session_info is not None:
                    print(f"Logs saved in: {session_info.session_dir}")
                return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if session_info is not None:
            print(f"Logs saved in: {session_info.session_dir}")
        if robot is not None:
            robot.stop()
        return 1
    except Exception as exc:
        print(f"\nError: {exc}")
        if session_info is not None:
            print(f"Logs saved in: {session_info.session_dir}")
        if robot is not None:
            robot.stop()
        return 1


if __name__ == "__main__":
    sys.exit(main())
