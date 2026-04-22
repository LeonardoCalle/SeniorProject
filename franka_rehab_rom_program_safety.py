#!/usr/bin/env python3
"""
Franka rehab ROM program — Cartesian impedance control.

Architecture
------------
- Uses start_cartesian_pose_control(ControllerMode.JointImpedance).
  The robot's internal joint impedance controller handles gravity
  compensation, stiffness, and damping — we only send it a desired
  Cartesian pose every tick.
- One controller session covers the initial home move AND all cycles with
  no gaps between them, so the arm never sags under the brace weight.
- Joint impedance is configured once via set_joint_impedance before the
  exercise starts.  Note: set_cartesian_impedance has no effect in
  JointImpedance mode (confirmed libfranka issue #180) and is not called.
- Motion is planned in Cartesian space (minimum-jerk rotation around a
  local EE axis) using a state machine.
- Resistance detection uses the SafetyMonitor (external wrench + filtered
  joint torques).  On soft stop the desired pose is frozen at the last
  commanded position and the impedance controller pulls the arm back home.

Real-time safety notes
-----------------------
- print() and file I/O are queued from the control callback and flushed
  from the outer loop after writeOnce().  The FCI docs explicitly warn that
  printing inside a control loop adds unacceptable latency.
- Initial robot state is read from control.readOnce() AFTER opening the
  controller, matching the libfranka sample code pattern.

State machine phases
---------------------
  MOVE_TO_HOME → DWELL_HOME → OUTBOUND → DWELL_END → RETURN  (loops per cycle)
  Any phase → SOFT_STOP → SOFT_RETURN → DONE
  DONE → motion_finished, controller exits
"""

import argparse
import collections
import csv
import math
import os
import re
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from pylibfranka import CartesianPose, ControllerMode, RealtimeConfig, Robot

# ---------------------------
# Rehab geometry
# ---------------------------
LOWER_HOME_EXT_LEFT_INT_RIGHT = {
    "x_rot_deg": -110.0, "y_rot_deg": 0.0, "z_rot_deg": 0.0,
    "x_mm": 150.0, "y_mm": 580.0, "z_mm": 280.0,
}
LOWER_HOME_EXT_RIGHT_INT_LEFT = {
    "x_rot_deg": -110.0, "y_rot_deg": 0.0, "z_rot_deg": 90.0,
    "x_mm": 150.0, "y_mm": 580.0, "z_mm": 280.0,
}
UPPER_HOME = {
    "x_rot_deg": 180.0, "y_rot_deg": 0.0, "z_rot_deg": 90.0,
    "x_mm": 0.0, "y_mm": 600.0, "z_mm": 485.0,
}

ROM_LEVELS_DEG = {
    "External Rotation Lower": [25.0, 40.0, 55.0, 70.0, 85.0],
    "External Rotation Upper": [30.0, 45.0, 60.0, 75.0, 90.0],
    "Internal Rotation Lower": [20.0, 30.0, 40.0, 50.0, 60.0],
    "Internal Rotation Upper": [15.0, 25.0, 35.0, 40.0, 45.0],
}

TIME_SERIES_LOG_INTERVAL_S = 0.05

# Hardware collision thresholds — kept wide so the software SafetyMonitor
# always gets first chance to stop motion before a hardware reflex fires.
LOWER_TORQUE_THRESHOLDS = [70.0, 70.0, 60.0, 60.0, 45.0, 40.0, 35.0]
UPPER_TORQUE_THRESHOLDS = [70.0, 70.0, 60.0, 60.0, 45.0, 40.0, 35.0]
LOWER_FORCE_THRESHOLDS  = [55.0, 55.0, 55.0, 45.0, 45.0, 45.0]
UPPER_FORCE_THRESHOLDS  = [55.0, 55.0, 55.0, 45.0, 45.0, 45.0]

# Joint impedance — controls how stiffly each joint resists displacement
# from the commanded pose.  These values give a firm but compliant hold
# appropriate for guiding a patient's arm through ROM exercises.
# Note: set_cartesian_impedance() has NO EFFECT in JointImpedance mode
# (libfranka issue #180) and is therefore not used.
DEFAULT_JOINT_IMPEDANCE = [3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0]  # Nm/rad


# ---------------------------
# Data classes
# ---------------------------
@dataclass
class MotionSpec:
    name: str
    axis: str          # "x" or "y" — local EE axis to rotate around
    sign: float
    levels_deg: List[float]
    position: str      # "upper" or "lower"
    motion_family: str # "external" or "internal"


@dataclass
class SoftSafetyLimits:
    max_force_n: float
    max_torque_nm: float
    max_joint_torque_nm: float
    consecutive_samples: int
    clear_samples: int = 4


@dataclass
class SessionInfo:
    session_id: str
    patient_id: str
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


# ---------------------------
# Motion catalogue
# ---------------------------
MOTIONS = [
    MotionSpec("External Rotation Lower", axis="y", sign=+1.0,
               levels_deg=ROM_LEVELS_DEG["External Rotation Lower"],
               position="lower", motion_family="external"),
    MotionSpec("External Rotation Upper", axis="x", sign=+1.0,
               levels_deg=ROM_LEVELS_DEG["External Rotation Upper"],
               position="upper", motion_family="external"),
    MotionSpec("Internal Rotation Lower", axis="y", sign=-1.0,
               levels_deg=ROM_LEVELS_DEG["Internal Rotation Lower"],
               position="lower", motion_family="internal"),
    MotionSpec("Internal Rotation Upper", axis="x", sign=-1.0,
               levels_deg=ROM_LEVELS_DEG["Internal Rotation Upper"],
               position="upper", motion_family="internal"),
]


# ---------------------------
# Math helpers
# ---------------------------
def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_xyz_deg_to_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    return rot_z(deg2rad(z_deg)) @ rot_y(deg2rad(y_deg)) @ rot_x(deg2rad(x_deg))


def matrix_to_pose_16(R: np.ndarray, t: np.ndarray) -> List[float]:
    """Pack a 3x3 rotation and 3-vector translation into a 16-element
    column-major list as expected by pylibfranka CartesianPose."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
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
    """Derivative of the minimum-jerk profile w.r.t. normalised time s."""
    s = max(0.0, min(1.0, s))
    return 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4


# Duration of the velocity ramp-down when a soft stop fires mid-motion.
# During this window the robot decelerates smoothly to zero velocity using
# the same minimum-jerk profile, preventing the velocity / acceleration
# discontinuity that triggers Franka's cartesian_motion_generator reflex.
SOFT_DECEL_RAMP_S = 0.4  # seconds

# How long to hold at home and measure the resting arm+brace load after
# strap-in before starting the first cycle.  The mean readings become the
# SafetyMonitor baseline so only *changes* above resting trigger a soft stop.
BASELINE_MEASURE_TIME_S = 2.0  # seconds


def pose_from_state(robot_state) -> np.ndarray:
    """Read O_T_EE from robot state into a 4x4 homogeneous matrix."""
    return np.array(robot_state.O_T_EE, dtype=float).reshape((4, 4)).T


def interp_pose(T_a: np.ndarray, T_b: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly interpolate between two poses; orthogonalise rotation via SVD.
    Short-circuits at boundaries to avoid unnecessary computation."""
    if alpha <= 0.0:
        return T_a
    if alpha >= 1.0:
        return T_b
    p = (1.0 - alpha) * T_a[:3, 3] + alpha * T_b[:3, 3]
    R_blend = (1.0 - alpha) * T_a[:3, :3] + alpha * T_b[:3, :3]
    U, _, Vt = np.linalg.svd(R_blend)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def local_axis_in_base(R: np.ndarray, axis: str) -> np.ndarray:
    v = R[:, 0] if axis == "x" else R[:, 1]
    n = np.linalg.norm(v)
    return np.array([1.0, 0.0, 0.0]) if n < 1e-12 else v / n


def apply_rotation_to_pose(T: np.ndarray, axis: str, angle_rad: float) -> np.ndarray:
    """Return T with its rotation component rotated by angle_rad around its local axis."""
    R_cur = T[:3, :3]
    ax = local_axis_in_base(R_cur, axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    K = np.array([
        [0,      -ax[2],  ax[1]],
        [ax[2],   0,     -ax[0]],
        [-ax[1],  ax[0],  0    ],
    ])
    R_delta = np.eye(3) + s * K + (1 - c) * (K @ K)
    T_new = T.copy()
    T_new[:3, :3] = R_delta @ R_cur
    return T_new


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
        return LOWER_HOME_EXT_LEFT_INT_RIGHT if side == "left" else LOWER_HOME_EXT_RIGHT_INT_LEFT
    return LOWER_HOME_EXT_LEFT_INT_RIGHT if side == "right" else LOWER_HOME_EXT_RIGHT_INT_LEFT


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
# Safety monitor
# ---------------------------
class SafetyMonitor:
    def __init__(
        self,
        limits: SoftSafetyLimits,
        grace_time_s: float = 0.20,
        clear_ratio: float = 0.85,
    ):
        self.limits = limits
        self.grace_time_s = max(0.0, grace_time_s)
        self.clear_ratio = min(max(clear_ratio, 0.0), 1.0)
        self.counter = 0
        self._clear_counter = 0
        # Baseline readings measured after strap-in at rest.  The check()
        # method compares the *delta above baseline* against the thresholds so
        # that the resting arm+brace load never triggers a soft stop.
        self.baseline_force_n = 0.0
        self.baseline_torque_nm = 0.0
        self.baseline_joint_torque_nm = 0.0

    def set_baselines(self, force_n: float, torque_nm: float, joint_torque_nm: float) -> None:
        """Update resting baselines measured after strap-in."""
        self.baseline_force_n = max(0.0, force_n)
        self.baseline_torque_nm = max(0.0, torque_nm)
        self.baseline_joint_torque_nm = max(0.0, joint_torque_nm)

    def reset(self) -> None:
        self.counter = 0
        self._clear_counter = 0

    def check(self, robot_state, elapsed_s: float = 0.0) -> Tuple[bool, str]:
        force_mag, torque_mag, joint_torque_mag, _, _ = extract_interaction_metrics(robot_state)

        if elapsed_s < self.grace_time_s:
            self.counter = 0
            return False, ""

        # Compare delta above resting baseline so arm+brace weight is ignored
        force_delta = max(0.0, force_mag - self.baseline_force_n)
        torque_delta = max(0.0, torque_mag - self.baseline_torque_nm)
        joint_delta  = max(0.0, joint_torque_mag - self.baseline_joint_torque_nm)

        exceeded = []
        if force_delta >= self.limits.max_force_n:
            exceeded.append(f"d|F|={force_delta:.2f} N (raw {force_mag:.2f})")
        if torque_delta >= self.limits.max_torque_nm:
            exceeded.append(f"d|Tau_wrench|={torque_delta:.2f} Nm (raw {torque_mag:.2f})")
        if joint_delta >= self.limits.max_joint_torque_nm:
            exceeded.append(f"dmax|Tau_joint|={joint_delta:.2f} Nm (raw {joint_torque_mag:.2f})")

        if exceeded:
            self.counter += 1
            self._clear_counter = 0
        else:
            if (force_delta < self.limits.max_force_n * self.clear_ratio and
                    torque_delta < self.limits.max_torque_nm * self.clear_ratio and
                    joint_delta < self.limits.max_joint_torque_nm * self.clear_ratio):
                self._clear_counter += 1
                if self._clear_counter >= self.limits.clear_samples:
                    self.counter = 0
                    self._clear_counter = 0

        if self.counter >= self.limits.consecutive_samples:
            return True, "SOFT SAFETY EVENT: " + ", ".join(exceeded)
        return False, ""


# ---------------------------
# Robot setup
# ---------------------------
def set_robot_behaviour(robot: Robot) -> None:
    """Apply collision thresholds and joint impedance parameters.
    NOTE: set_cartesian_impedance() is intentionally NOT called here —
    it has no effect when using ControllerMode.JointImpedance (libfranka #180).
    """
    robot.set_collision_behavior(
        LOWER_TORQUE_THRESHOLDS, UPPER_TORQUE_THRESHOLDS,
        LOWER_FORCE_THRESHOLDS,  UPPER_FORCE_THRESHOLDS,
    )
    robot.set_joint_impedance(DEFAULT_JOINT_IMPEDANCE)


# ---------------------------
# ---------------------------
# Exercise block — single continuous impedance controller
# ---------------------------
def run_exercise_block(
    robot: Robot,
    home_T: np.ndarray,
    axis: str,
    signed_angle_deg: float,
    home_time: float,
    outbound_time: float,
    return_time: float,
    soft_return_time: float,
    dwell_time: float,
    cycles: int,
    safety_monitor: SafetyMonitor,
    log_callback: Optional[Callable],
    base_context_template: Dict[str, object],
    timeseries_fields: List[str],
) -> List[CycleResult]:
    """
    Run all cycles for one exercise as a single continuous Cartesian impedance
    controller session.  The controller never stops between cycles, so the
    robot's impedance law always holds the arm — no sag, no controller gaps.

    Commands are CartesianPose objects (desired EE pose).  The robot's internal
    JointImpedance controller handles gravity compensation, stiffness, and
    damping from the set_joint_impedance settings.

    Real-time safety
    ----------------
    print() and log_callback are NEVER called from inside the control callback.
    Messages and log rows are pushed onto deques and flushed from the outer
    loop after writeOnce(), keeping all I/O out of the 1 kHz path.

    State machine
    -------------
    MOVE_TO_HOME  — interpolate from actual start pose to home_T over home_time
    WAIT_STRAP    — hold home_T; background thread prompts therapist to strap arm
                    in; transitions to MEASURE_BASELINE when Enter is pressed
    MEASURE_BASELINE — hold home_T for BASELINE_MEASURE_TIME_S while accumulating
                    resting force/torque readings; sets SafetyMonitor baselines so
                    the arm+brace weight is never mistaken for patient resistance
    DWELL_HOME    — hold home_T for dwell_time between cycles
    OUTBOUND      — rotate to T_end over outbound_time  (safety monitored)
    DWELL_END     — hold T_end for dwell_time
    RETURN        — rotate back to home_T over return_time  (safety monitored)
    SOFT_DECEL    — smoothly decelerates to zero velocity over SOFT_DECEL_RAMP_S
                    using a min-jerk fade.  This prevents the velocity /
                    acceleration discontinuity that causes Franka's
                    cartesian_motion_generator reflex.
    SOFT_RETURN   — slow interpolation from decel end-pose back to home_T
    DONE          — send motion_finished, controller exits
    """
    theta_total = deg2rad(abs(signed_angle_deg))
    sign = 1.0 if signed_angle_deg >= 0.0 else -1.0
    T_end = apply_rotation_to_pose(home_T, axis, theta_total * sign)

    # Pre-compute constant pose lists to avoid repackaging every tick
    home_pose_16: List[float] = matrix_to_pose_16(home_T[:3, :3], home_T[:3, 3])
    T_end_pose_16: List[float] = matrix_to_pose_16(T_end[:3, :3], T_end[:3, 3])

    results: List[CycleResult] = []

    # Queues for print messages and log rows — flushed from outer loop, NOT
    # from inside the realtime callback (FCI docs: printing in a control loop
    # adds unacceptable delay).
    msg_queue: Deque[str] = collections.deque()
    log_queue: Deque[Dict] = collections.deque()

    # Event set by the background strap-in prompt thread when the user presses
    # Enter.  The WAIT_STRAP phase polls this every tick without blocking the
    # 1 kHz control loop.
    strap_confirmed = threading.Event()

    state = {
        "phase":                "MOVE_TO_HOME",
        "phase_elapsed":        0.0,
        "total_elapsed":        0.0,
        "cycle":                1,
        "peak_force_n":         0.0,
        "peak_torque_nm":       0.0,
        "peak_joint_torque_nm": 0.0,
        "samples_logged":       0,
        "last_log_time":        -1e9,
        "soft_stop_msg":        "",
        "T_frozen":             home_T.copy(),
        "frozen_pose_16":       home_pose_16[:],
        "T_move_start":         home_T.copy(),
        # Set True by control_callback when the home move finishes; the outer
        # loop sees this flag, spawns the strap-in prompt thread, then clears it.
        "show_strap_prompt":    False,
        # Baseline measurement accumulators — populated during MEASURE_BASELINE,
        # DWELL_HOME, and DWELL_END so the safety threshold continuously adapts
        # to the pose-dependent gravity loading of the arm+brace.
        "bl_f_sum":   0.0,
        "bl_t_sum":   0.0,
        "bl_j_sum":   0.0,
        "bl_count":   0,
        # Stored baseline values at each endpoint — used to interpolate the
        # expected gravity loading throughout the arc of motion.
        "home_bl_f":  0.0,  "home_bl_t":  0.0,  "home_bl_j":  0.0,
        "end_bl_f":   0.0,  "end_bl_t":   0.0,  "end_bl_j":   0.0,
        "has_end_bl": False,  # True after the first DWELL_END measurement
        # Fields populated when soft stop fires — used by SOFT_DECEL
        "decel_alpha_fire":  0.0,  # trajectory alpha at moment of soft stop
        "decel_v_fire":      0.0,  # d(alpha)/dt at moment of soft stop
        "decel_T_start":     home_T.copy(),  # trajectory start pose
        "decel_T_end":       T_end.copy(),   # trajectory end pose
    }

    def _transition(new_phase: str, robot_state=None) -> None:
        state["phase"] = new_phase
        state["phase_elapsed"] = 0.0
        if new_phase == "MOVE_TO_HOME" and robot_state is not None:
            state["T_move_start"] = pose_from_state(robot_state)

    def _reset_peaks() -> None:
        state["peak_force_n"] = 0.0
        state["peak_torque_nm"] = 0.0
        state["peak_joint_torque_nm"] = 0.0
        state["samples_logged"] = 0

    def control_callback(robot_state, period) -> CartesianPose:
        dt = period.to_sec()
        state["phase_elapsed"] += dt
        state["total_elapsed"] += dt
        t     = state["phase_elapsed"]
        phase = state["phase"]

        force_mag, torque_mag, joint_torque_mag, wrench, _ = extract_interaction_metrics(robot_state)
        state["peak_force_n"]          = max(state["peak_force_n"],          force_mag)
        state["peak_torque_nm"]        = max(state["peak_torque_nm"],        torque_mag)
        state["peak_joint_torque_nm"]  = max(state["peak_joint_torque_nm"],  joint_torque_mag)

        # T_des defaults to home_T so it is always bound even in DONE
        T_des = home_T
        pose_16: Optional[List[float]] = None  # use pre-computed when possible

        if phase == "MOVE_TO_HOME":
            alpha = minimum_jerk_scalar(min(t / home_time, 1.0))
            T_des = interp_pose(state["T_move_start"], home_T, alpha)
            if t >= home_time + dwell_time:
                # Signal the outer loop to spawn the strap-in prompt thread.
                # Do NOT print or call input() here — we are in the realtime path.
                state["show_strap_prompt"] = True
                _transition("WAIT_STRAP")

        elif phase == "WAIT_STRAP":
            # Hold home pose rigidly while the therapist straps the arm in.
            pose_16 = home_pose_16
            if strap_confirmed.is_set():
                msg_queue.append(
                    f"Measuring resting load for {BASELINE_MEASURE_TIME_S:.0f} s — keep the arm still..."
                )
                _transition("MEASURE_BASELINE")

        elif phase == "MEASURE_BASELINE":
            # Hold home and accumulate force/torque readings to establish the
            # resting baseline for the arm+brace load.  The SafetyMonitor will
            # then only trigger on *increases* above this resting value.
            pose_16 = home_pose_16
            state["bl_f_sum"] += force_mag
            state["bl_t_sum"] += torque_mag
            state["bl_j_sum"] += joint_torque_mag
            state["bl_count"] += 1
            if t >= BASELINE_MEASURE_TIME_S:
                n = max(state["bl_count"], 1)
                hf = state["bl_f_sum"] / n
                ht = state["bl_t_sum"] / n
                hj = state["bl_j_sum"] / n
                safety_monitor.set_baselines(hf, ht, hj)
                safety_monitor.reset()
                # Store home baselines — used to interpolate during motion
                state["home_bl_f"] = hf
                state["home_bl_t"] = ht
                state["home_bl_j"] = hj
                # Initialise end baseline to home so interpolation is safe
                # before the first DWELL_END measurement
                state["end_bl_f"] = hf
                state["end_bl_t"] = ht
                state["end_bl_j"] = hj
                # Reset accumulator for reuse in DWELL phases
                state["bl_f_sum"] = 0.0
                state["bl_t_sum"] = 0.0
                state["bl_j_sum"] = 0.0
                state["bl_count"] = 0
                msg_queue.append(
                    f"Baseline set — resting load: "
                    f"{hf:.1f} N force, {ht:.1f} Nm wrench, {hj:.1f} Nm joint. "
                    f"Starting cycle 1/{cycles}"
                )
                _transition("OUTBOUND")

        elif phase == "DWELL_HOME":
            pose_16 = home_pose_16
            # Accumulate readings at home pose so the baseline stays fresh
            # each cycle (the arm angle at home is constant so this is stable)
            state["bl_f_sum"] += force_mag
            state["bl_t_sum"] += torque_mag
            state["bl_j_sum"] += joint_torque_mag
            state["bl_count"] += 1
            if t >= dwell_time:
                n = max(state["bl_count"], 1)
                hf = state["bl_f_sum"] / n
                ht = state["bl_t_sum"] / n
                hj = state["bl_j_sum"] / n
                state["home_bl_f"] = hf
                state["home_bl_t"] = ht
                state["home_bl_j"] = hj
                safety_monitor.set_baselines(hf, ht, hj)
                safety_monitor.reset()
                state["bl_f_sum"] = 0.0
                state["bl_t_sum"] = 0.0
                state["bl_j_sum"] = 0.0
                state["bl_count"] = 0
                msg_queue.append(f"Cycle {state['cycle']}/{cycles} — outbound")
                _transition("OUTBOUND")

        elif phase == "OUTBOUND":
            alpha = minimum_jerk_scalar(min(t / outbound_time, 1.0))
            T_des = interp_pose(home_T, T_end, alpha)
            # Interpolate expected gravity baseline between home and end pose.
            # This accounts for the changing gravity loading as the arm sweeps
            # through the arc, preventing false soft stops from pose-dependent
            # force increases.  Once DWELL_END has been measured (has_end_bl),
            # interpolation is fully accurate; before that, home baseline is used
            # uniformly (conservative but safe for the first outbound pass).
            if state["has_end_bl"]:
                safety_monitor.set_baselines(
                    state["home_bl_f"] + alpha * (state["end_bl_f"] - state["home_bl_f"]),
                    state["home_bl_t"] + alpha * (state["end_bl_t"] - state["home_bl_t"]),
                    state["home_bl_j"] + alpha * (state["end_bl_j"] - state["home_bl_j"]),
                )
            hit, msg = safety_monitor.check(robot_state, t)
            if hit:
                msg_queue.append(msg)
                msg_queue.append("Soft stop — decelerating, then returning to home.")
                state["soft_stop_msg"] = msg
                # Record trajectory state at the exact moment of soft stop so
                # SOFT_DECEL can continue the motion smoothly to zero velocity.
                s = min(t / outbound_time, 1.0)
                state["decel_alpha_fire"] = minimum_jerk_scalar(s)
                state["decel_v_fire"]     = minimum_jerk_derivative(s) / outbound_time
                state["decel_T_start"]    = home_T
                state["decel_T_end"]      = T_end
                _transition("SOFT_DECEL")
            elif t >= outbound_time:
                _transition("DWELL_END")

        elif phase == "DWELL_END":
            pose_16 = T_end_pose_16
            # Accumulate readings at the end position — gravity acts on the arm
            # differently here so the baseline is different from home.
            state["bl_f_sum"] += force_mag
            state["bl_t_sum"] += torque_mag
            state["bl_j_sum"] += joint_torque_mag
            state["bl_count"] += 1
            if t >= dwell_time:
                n = max(state["bl_count"], 1)
                ef = state["bl_f_sum"] / n
                et = state["bl_t_sum"] / n
                ej = state["bl_j_sum"] / n
                state["end_bl_f"] = ef
                state["end_bl_t"] = et
                state["end_bl_j"] = ej
                state["has_end_bl"] = True
                safety_monitor.set_baselines(ef, et, ej)
                safety_monitor.reset()
                state["bl_f_sum"] = 0.0
                state["bl_t_sum"] = 0.0
                state["bl_j_sum"] = 0.0
                state["bl_count"] = 0
                msg_queue.append(f"Cycle {state['cycle']}/{cycles} — return")
                _transition("RETURN")

        elif phase == "RETURN":
            alpha = minimum_jerk_scalar(min(t / return_time, 1.0))
            T_des = interp_pose(T_end, home_T, alpha)
            # Return sweep: baseline goes from end-position reading back to home.
            if state["has_end_bl"]:
                safety_monitor.set_baselines(
                    state["end_bl_f"] + alpha * (state["home_bl_f"] - state["end_bl_f"]),
                    state["end_bl_t"] + alpha * (state["home_bl_t"] - state["end_bl_t"]),
                    state["end_bl_j"] + alpha * (state["home_bl_j"] - state["end_bl_j"]),
                )
            hit, msg = safety_monitor.check(robot_state, t)
            if hit:
                msg_queue.append(msg)
                msg_queue.append("Soft stop on return — decelerating, then returning to home.")
                state["soft_stop_msg"] = "Return path soft stop: " + msg
                s = min(t / return_time, 1.0)
                state["decel_alpha_fire"] = minimum_jerk_scalar(s)
                state["decel_v_fire"]     = minimum_jerk_derivative(s) / return_time
                state["decel_T_start"]    = T_end   # RETURN goes T_end → home_T
                state["decel_T_end"]      = home_T
                _transition("SOFT_DECEL")
            elif t >= return_time:
                results.append(CycleResult(
                    completed_normally=True, soft_stop_reason="",
                    peak_force_n=state["peak_force_n"],
                    peak_torque_nm=state["peak_torque_nm"],
                    peak_joint_torque_nm=state["peak_joint_torque_nm"],
                    samples_logged=state["samples_logged"],
                ))
                _reset_peaks()
                state["cycle"] += 1
                if state["cycle"] > cycles:
                    _transition("DONE")
                else:
                    _transition("DWELL_HOME")

        elif phase == "SOFT_DECEL":
            # Smoothly decelerate from the velocity at soft-stop to zero.
            #
            # We continue the original trajectory but with a fading speed.
            # If the velocity at soft-stop was v_fire (alpha/sec) and time
            # within this phase is t, the fade function f(t) goes from 1 to 0
            # using a min-jerk profile so that acceleration is also continuous:
            #
            #   f(u) = 1 - min_jerk(u)       where u = t / SOFT_DECEL_RAMP_S
            #
            # The effective alpha is:
            #   alpha(t) = alpha_fire + v_fire * SOFT_DECEL_RAMP_S
            #              * integral_0^u (1 - min_jerk(s)) ds
            #            = alpha_fire + v_fire * SOFT_DECEL_RAMP_S
            #              * (u - 2.5u^4 + 3u^5 - u^6)
            #
            # Velocity at t=0 equals v_fire (continuous with previous phase).
            # Velocity at t=RAMP equals 0 (no jerk into SOFT_RETURN).
            # Acceleration at both endpoints is 0 (no reflex).
            u = min(t / SOFT_DECEL_RAMP_S, 1.0)
            mj_integral = 2.5 * u**4 - 3.0 * u**5 + u**6
            alpha_now = state["decel_alpha_fire"] + state["decel_v_fire"] * SOFT_DECEL_RAMP_S * (u - mj_integral)
            alpha_now = max(0.0, min(1.0, alpha_now))
            T_des = interp_pose(state["decel_T_start"], state["decel_T_end"], alpha_now)
            if t >= SOFT_DECEL_RAMP_S:
                # Decel complete — record where we stopped and begin slow return
                state["T_frozen"] = T_des.copy()
                state["frozen_pose_16"] = matrix_to_pose_16(T_des[:3, :3], T_des[:3, 3])
                _transition("SOFT_RETURN")

        elif phase == "SOFT_RETURN":
            alpha = minimum_jerk_scalar(min(t / soft_return_time, 1.0))
            T_des = interp_pose(state["T_frozen"], home_T, alpha)
            if t >= soft_return_time + dwell_time:
                results.append(CycleResult(
                    completed_normally=False,
                    soft_stop_reason=state["soft_stop_msg"],
                    peak_force_n=state["peak_force_n"],
                    peak_torque_nm=state["peak_torque_nm"],
                    peak_joint_torque_nm=state["peak_joint_torque_nm"],
                    samples_logged=state["samples_logged"],
                ))
                _transition("DONE")

        # DONE falls through to T_des=home_T set above

        if pose_16 is None:
            pose_16 = matrix_to_pose_16(T_des[:3, :3], T_des[:3, 3])

        cmd = CartesianPose(pose_16)
        if phase == "DONE":
            cmd.motion_finished = True
            return cmd

        # Queue log row — NO file I/O here, flushed from outer loop
        if (log_callback is not None and
                state["total_elapsed"] - state["last_log_time"] >= TIME_SERIES_LOG_INTERVAL_S):
            T_cur = pose_from_state(robot_state)
            log_queue.append({
                **base_context_template,
                "cycle_number":          state["cycle"],
                "phase":                 phase,
                "elapsed_s":             round(state["total_elapsed"], 4),
                "commanded_omega_rad_s": 0.0,
                "ee_x_m":               round(float(T_cur[0, 3]), 6),
                "ee_y_m":               round(float(T_cur[1, 3]), 6),
                "ee_z_m":               round(float(T_cur[2, 3]), 6),
                "force_mag_n":          round(force_mag, 6),
                "torque_mag_nm":        round(torque_mag, 6),
                "joint_torque_mag_nm":  round(joint_torque_mag, 6),
                "wrench_fx":            round(float(wrench[0]), 6),
                "wrench_fy":            round(float(wrench[1]), 6),
                "wrench_fz":            round(float(wrench[2]), 6),
                "wrench_tx":            round(float(wrench[3]), 6),
                "wrench_ty":            round(float(wrench[4]), 6),
                "wrench_tz":            round(float(wrench[5]), 6),
                "joint_tau_max":        round(joint_torque_mag, 6),
                "safety_counter":       safety_monitor.counter,
                "safety_hit":           0,
            })
            state["last_log_time"] = state["total_elapsed"]
            state["samples_logged"] += 1

        return cmd

    # Set impedance parameters before starting the controller
    set_robot_behaviour(robot)

    control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
    try:
        # Read initial state from controller AFTER switching — matches libfranka
        # sample code pattern.  robot.read_once() before start risks a stale or
        # slightly different state causing a jerk on the first command.
        robot_state_0, _ = control.readOnce()
        state["T_move_start"] = pose_from_state(robot_state_0)
        # Send current pose immediately to satisfy the first write cycle
        control.writeOnce(CartesianPose(list(robot_state_0.O_T_EE)))

        while True:
            robot_state, period = control.readOnce()
            cmd = control_callback(robot_state, period)
            control.writeOnce(cmd)

            # Flush message and log queues outside the realtime path
            while msg_queue:
                print(msg_queue.popleft())
            if log_callback is not None:
                while log_queue:
                    log_callback(log_queue.popleft())

            # When the home move finishes, spawn a background thread to prompt
            # the therapist to strap in.  We cannot call input() on the main
            # thread here because that would block the control loop.
            if state["show_strap_prompt"]:
                state["show_strap_prompt"] = False
                def _strap_prompt():
                    print("\n" + "=" * 72)
                    print("Robot is at the home position and holding firmly.")
                    print("It is now safe to strap the patient's arm into the brace.")
                    print("Take your time — the robot will not move until you press Enter.")
                    print("=" * 72)
                    input("Press Enter when the arm is strapped in and ready to begin... ")
                    strap_confirmed.set()
                threading.Thread(target=_strap_prompt, daemon=True).start()

            if getattr(cmd, "motion_finished", False):
                break

    except Exception as exc:
        print(f"Impedance controller exited with error: {exc}")
        # Close controller cleanly using last desired pose (smooth, not noisy measured)
        try:
            last_state = robot.read_once()
            abort = CartesianPose(list(last_state.O_T_EE_d))
            abort.motion_finished = True
            control.writeOnce(abort)
        except Exception:
            pass
        if state["phase"] != "DONE":
            results.append(CycleResult(
                completed_normally=False,
                soft_stop_reason=f"Controller error: {exc}",
                peak_force_n=state["peak_force_n"],
                peak_torque_nm=state["peak_torque_nm"],
                peak_joint_torque_nm=state["peak_joint_torque_nm"],
                samples_logged=state["samples_logged"],
            ))

    # Flush any remaining messages/logs after controller exits
    while msg_queue:
        print(msg_queue.popleft())
    if log_callback is not None:
        while log_queue:
            log_callback(log_queue.popleft())

    return results


# ---------------------------
# Logging helpers
# ---------------------------
def sanitize_for_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "session"


def create_session_info(patient_id: str) -> SessionInfo:
    session_id = time.strftime("%Y%m%d_%H%M%S")
    patient_stub = sanitize_for_filename(patient_id)[:24]
    session_dir = Path.cwd() / "session_logs" / f"session_{session_id}_{patient_stub}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return SessionInfo(
        session_id=session_id,
        patient_id=patient_id,
        session_dir=str(session_dir),
        summary_csv_path=str(session_dir / "session_summary.csv"),
        timeseries_csv_path=str(session_dir / "session_timeseries.csv"),
    )


SUMMARY_FIELDS = [
    "session_id", "patient_id", "patient_side", "exercise_label", "exercise_name",
    "position", "motion_family", "cycle_number", "cycles_requested", "rom_deg",
    "home_time_s", "outbound_time_s", "return_time_s", "soft_return_time_s",
    "soft_stop", "soft_stop_reason",
    "peak_force_n", "peak_torque_nm", "peak_joint_torque_nm",
    "completed_normally", "timestamp_unix",
]

TIMESERIES_FIELDS = [
    "session_id", "patient_id", "patient_side", "exercise_label", "cycle_number",
    "phase", "elapsed_s", "commanded_omega_rad_s",
    "ee_x_m", "ee_y_m", "ee_z_m",
    "force_mag_n", "torque_mag_nm", "joint_torque_mag_nm",
    "wrench_fx", "wrench_fy", "wrench_fz", "wrench_tx", "wrench_ty", "wrench_tz",
    "joint_tau_max", "safety_counter", "safety_hit",
]


def init_csv(path: str, fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_row(path: str, fieldnames: List[str], row: Dict[str, object]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


# ---------------------------
# UI helpers
# ---------------------------
def ask_int(prompt: str, min_value: int = 1, max_value: Optional[int] = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit():
            v = int(raw)
            if v >= min_value and (max_value is None or v <= max_value):
                return v
        bound = f"{min_value} to {max_value}" if max_value is not None else f">= {min_value}"
        print(f"Please enter an integer {bound}.")


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
    return ["Left", "Right"][choose_from_menu("Select patient side:", ["Left", "Right"])]


def select_motion() -> MotionSpec:
    return MOTIONS[choose_from_menu("Select exercise:", [m.name for m in MOTIONS])]


def select_rom_level(motion: MotionSpec) -> float:
    idx = choose_from_menu(f"Select ROM for {motion.name}:", [f"{d:.1f} deg" for d in motion.levels_deg])
    return motion.levels_deg[idx]


def select_cycles() -> int:
    return ask_int("Enter number of cycles (1–10): ", 1, 10)


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Franka rehab ROM — Cartesian impedance control")
    parser.add_argument("--ip",                   type=str,   required=True,  help="Robot IP address")
    parser.add_argument("--home-time",            type=float, default=6.0,   help="Seconds to move to home pose at start of each exercise")
    parser.add_argument("--outbound-time",         type=float, default=4.0,   help="Seconds for outbound rotation")
    parser.add_argument("--return-time",           type=float, default=6.0,   help="Seconds for return rotation")
    parser.add_argument("--soft-return-time",      type=float, default=4.0,   help="Seconds for soft-stop return to home")
    parser.add_argument("--dwell-time",            type=float, default=0.5,   help="Seconds to pause at endpoints")
    parser.add_argument("--max-force-n",           type=float, default=12.0,  help="Soft-stop force threshold [N]")
    parser.add_argument("--max-torque-nm",         type=float, default=8.0,   help="Soft-stop wrench torque threshold [Nm]")
    parser.add_argument("--max-joint-torque-nm",   type=float, default=8.0,   help="Soft-stop joint torque threshold [Nm]")
    parser.add_argument("--consecutive-samples",   type=int,   default=3,     help="Samples above threshold before soft stop")
    parser.add_argument("--clear-samples",         type=int,   default=4,     help="Sustained samples below threshold before counter resets")
    args = parser.parse_args()

    base_safety = SoftSafetyLimits(
        args.max_force_n, args.max_torque_nm, args.max_joint_torque_nm,
        args.consecutive_samples, args.clear_samples,
    )

    print("WARNING: This program will move the robot.")
    print("Keep the user stop button within reach.")
    input("Press Enter to continue... ")

    patient_id = ask_text("Enter patient name or ID: ", "UnknownPatient")

    robot = None
    previous_label: Optional[str] = None
    session_info: Optional[SessionInfo] = None

    try:
        robot = Robot(args.ip, RealtimeConfig.kIgnore)
        robot.automatic_error_recovery()
        set_robot_behaviour(robot)

        session_info = create_session_info(patient_id)
        init_csv(session_info.summary_csv_path, SUMMARY_FIELDS)
        init_csv(session_info.timeseries_csv_path, TIMESERIES_FIELDS)


        while True:
            motion        = select_motion()
            patient_side  = select_patient_side()
            display_label = format_motion_label(motion, patient_side)
            rom_deg       = select_rom_level(motion)
            cycles        = select_cycles()

            if previous_label is not None:
                print("\n" + "!" * 72)
                print(f"Transitioning from '{previous_label}' to '{display_label}'.")
                print("Unstrap and reposition the user's arm before continuing.")
                input("Press ENTER only after repositioning is complete... ")

            safety_monitor = SafetyMonitor(base_safety, grace_time_s=0.20, clear_ratio=0.85)

            home_dict        = get_home_dict_for_motion(motion, patient_side)
            home_T           = build_home_pose(home_dict)
            signed_angle_deg = get_signed_angle_deg(motion, patient_side, rom_deg)

            print("\n" + "=" * 72)
            print(f"Exercise:        {display_label}")
            print(f"ROM:             {rom_deg:.1f} deg")
            print(f"Cycles:          {cycles}")
            print(f"Axis:            local {motion.axis.upper()}")
            print(f"Home time:       {args.home_time:.2f} s")
            print(f"Outbound time:   {args.outbound_time:.2f} s")
            print(f"Return time:     {args.return_time:.2f} s")
            print(f"Soft-stop force: {base_safety.max_force_n:.2f} N  "
                  f"wrench: {base_safety.max_torque_nm:.2f} Nm  "
                  f"joint: {base_safety.max_joint_torque_nm:.2f} Nm")
            print("=" * 72)

            print("\n" + "!" * 72)
            print("WARNING: The robot will now move to the home position.")
            print("Do NOT strap the patient in yet — keep the arm free.")
            print("Once the robot reaches home it will stop and prompt you to strap in.")
            print("!" * 72)
            input("Press Enter to begin home move... ")

            base_context = {
                "session_id":     session_info.session_id,
                "patient_id":     session_info.patient_id,
                "patient_side":   patient_side,
                "exercise_label": display_label,
            }

            cycle_results = run_exercise_block(
                robot=robot,
                home_T=home_T,
                axis=motion.axis,
                signed_angle_deg=signed_angle_deg,
                home_time=args.home_time,
                outbound_time=args.outbound_time,
                return_time=args.return_time,
                soft_return_time=args.soft_return_time,
                dwell_time=args.dwell_time,
                cycles=cycles,
                safety_monitor=safety_monitor,
                log_callback=lambda row: append_row(
                    session_info.timeseries_csv_path, TIMESERIES_FIELDS, row
                ),
                base_context_template=base_context,
                timeseries_fields=TIMESERIES_FIELDS,
            )

            for i, cr in enumerate(cycle_results, start=1):
                append_row(session_info.summary_csv_path, SUMMARY_FIELDS, {
                    "session_id":           session_info.session_id,
                    "patient_id":           session_info.patient_id,
                    "patient_side":         patient_side,
                    "exercise_label":       display_label,
                    "exercise_name":        motion.name,
                    "position":             motion.position,
                    "motion_family":        motion.motion_family,
                    "cycle_number":         i,
                    "cycles_requested":     cycles,
                    "rom_deg":              round(rom_deg, 3),
                    "home_time_s":          round(args.home_time, 4),
                    "outbound_time_s":      round(args.outbound_time, 4),
                    "return_time_s":        round(args.return_time, 4),
                    "soft_return_time_s":   round(args.soft_return_time, 4),
                    "soft_stop":            int(not cr.completed_normally),
                    "soft_stop_reason":     cr.soft_stop_reason,
                    "peak_force_n":         round(cr.peak_force_n, 6),
                    "peak_torque_nm":       round(cr.peak_torque_nm, 6),
                    "peak_joint_torque_nm": round(cr.peak_joint_torque_nm, 6),
                    "completed_normally":   int(cr.completed_normally),
                    "timestamp_unix":       round(time.time(), 6),
                })

            completed = sum(1 for r in cycle_results if r.completed_normally)
            print(f"Completed {completed} of {cycles} cycles for {display_label}.")

            previous_label = display_label

            if not ask_yes_no("Run another exercise? [y/n]: "):
                print("Program complete.")
                return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        if robot is not None:
            robot.stop()
        return 1
    except Exception as exc:
        print(f"\nError: {exc}")
        if robot is not None:
            robot.stop()
        return 1
    finally:
        if session_info is not None:
            print(f"Logs saved in: {session_info.session_dir}")


if __name__ == "__main__":
    sys.exit(main())
    
