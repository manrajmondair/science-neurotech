"""PyBullet-based inverse kinematics support for high-level arm control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import importlib.util
import logging
import math
from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

from .commands import EndEffectorDeltaCommand
from .controller import extract_joint_positions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmPose:
    """Cartesian end-effector pose in the robot base frame."""

    x: float
    y: float
    z: float
    tool_roll: float


@dataclass(frozen=True)
class ArmKinematics:
    """Approximate geometric dimensions for the SO-101 arm."""

    base_height: float = 0.075
    upper_arm_length: float = 0.16
    forearm_length: float = 0.16
    tool_length: float = 0.1

    @classmethod
    def from_urdf(cls, urdf_path: str | Path, joint_names: list[str]) -> "ArmKinematics":
        root = ET.parse(urdf_path).getroot()
        joint_origins = {
            joint.attrib["name"]: _parse_origin_xyz(joint)
            for joint in root.findall("joint")
        }

        missing = [name for name in joint_names if name not in joint_origins]
        if missing:
            raise ValueError(
                "URDF does not contain the configured IK joint names. "
                f"Missing: {missing}. Available joints: {sorted(joint_origins)}"
            )

        return cls(
            base_height=abs(joint_origins[joint_names[0]][2]),
            upper_arm_length=_vector_length(joint_origins[joint_names[2]]),
            forearm_length=_vector_length(joint_origins[joint_names[3]]),
            tool_length=_vector_length(joint_origins[joint_names[4]]),
        )


@dataclass
class SO101IKTranslator:
    """Translate end-effector deltas into joint targets using PyBullet IK."""

    urdf_path: str | None = None
    use_degrees: bool = True
    kinematics: ArmKinematics = field(default_factory=ArmKinematics)
    joint_names: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
    )
    end_effector_link: str = "tool_tip"
    _backend: "_PyBulletIKBackend" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not _pybullet_available():
            raise ImportError("pybullet is required for IK. Install dependencies and run `uv sync`.")
        if self.urdf_path:
            self.kinematics = ArmKinematics.from_urdf(self.urdf_path, self.joint_names)
        self._backend = _PyBulletIKBackend(
            kinematics=self.kinematics,
            urdf_path=self.urdf_path,
            use_degrees=self.use_degrees,
            joint_names=self.joint_names,
            end_effector_link=self.end_effector_link,
        )

    def forward(self, observation: Mapping[str, Any]) -> ArmPose:
        return self._backend.forward(observation)

    def translate(
        self,
        command: EndEffectorDeltaCommand,
        observation: Mapping[str, Any],
        last_targets: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return self._backend.translate(command, observation, last_targets)


def _pybullet_available() -> bool:
    return importlib.util.find_spec("pybullet") is not None


@dataclass
class _PyBulletIKBackend:
    """PyBullet-backed FK/IK wrapper."""

    kinematics: ArmKinematics
    urdf_path: str | None
    use_degrees: bool
    joint_names: list[str]
    end_effector_link: str
    _tempdir: tempfile.TemporaryDirectory[str] | None = field(default=None, init=False, repr=False)
    _client: int = field(default=0, init=False, repr=False)
    _pybullet: Any = field(default=None, init=False, repr=False)
    _robot_id: int = field(default=0, init=False, repr=False)
    _joint_name_to_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _joint_indices: list[int] = field(default_factory=list, init=False, repr=False)
    _end_effector_index: int = field(default=0, init=False, repr=False)
    # Full DOF structure for building correctly-sized IK arrays.
    _dof_joint_names: list[str] = field(default_factory=list, init=False, repr=False)
    _dof_urdf_limits: list[tuple[float, float]] = field(default_factory=list, init=False, repr=False)
    _dof_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        import pybullet as pybullet

        self._pybullet = pybullet
        self._client = pybullet.connect(pybullet.DIRECT)
        urdf_path = self.urdf_path or self._build_generated_urdf()
        self._robot_id = pybullet.loadURDF(urdf_path, useFixedBase=True, physicsClientId=self._client)

        dof_idx = 0
        for joint_index in range(pybullet.getNumJoints(self._robot_id, physicsClientId=self._client)):
            info = pybullet.getJointInfo(self._robot_id, joint_index, physicsClientId=self._client)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            link_name = info[12].decode("utf-8")
            lower_limit = float(info[8])
            upper_limit = float(info[9])

            if joint_type != pybullet.JOINT_FIXED:
                self._dof_joint_names.append(joint_name)
                self._dof_urdf_limits.append((lower_limit, upper_limit))
                self._dof_index[joint_name] = dof_idx
                dof_idx += 1

            if joint_type == pybullet.JOINT_REVOLUTE:
                self._joint_name_to_index[joint_name] = joint_index
            if link_name == self.end_effector_link:
                self._end_effector_index = joint_index

        missing = [name for name in self.joint_names if name not in self._joint_name_to_index]
        if missing:
            raise ValueError(
                "URDF does not expose the configured IK joints. "
                f"Missing: {missing}. Available revolute joints: {sorted(self._joint_name_to_index)}"
            )
        self._joint_indices = [self._joint_name_to_index[name] for name in self.joint_names]
        if self._end_effector_index == 0:
            raise ValueError(
                f"URDF does not expose the configured end-effector link '{self.end_effector_link}'."
            )

    def forward(self, observation: Mapping[str, Any]) -> ArmPose:
        joints = extract_joint_positions(observation)
        q = [
            self._to_radians(joints["shoulder_pan"]),
            self._to_radians(joints["shoulder_lift"]),
            self._to_radians(joints["elbow_flex"]),
            self._to_radians(joints["wrist_flex"]),
            self._to_radians(joints["wrist_roll"]),
        ]
        self._reset_state(q)
        state = self._pybullet.getLinkState(
            self._robot_id,
            self._end_effector_index,
            computeForwardKinematics=True,
            physicsClientId=self._client,
        )
        position = state[4]
        return ArmPose(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            tool_roll=float(joints["wrist_roll"]),
        )

    def translate(
        self,
        command: EndEffectorDeltaCommand,
        observation: Mapping[str, Any],
        last_targets: dict[str, float] | None = None,
    ) -> dict[str, float]:
        joints = extract_joint_positions(observation)
        current_pan_rad = self._to_radians(joints["shoulder_pan"])
        target_pan_rad = current_pan_rad + command.dy

        # For joints that accumulate deltas (wrist_roll, gripper), prefer last
        # commanded target as the base to prevent drift from chasing the observation.
        base = last_targets if last_targets else joints
        gripper = max(0.0, min(100.0, base.get("gripper", joints["gripper"]) + command.d_jaw))
        wrist_roll_base = base.get("wrist_roll", joints["wrist_roll"])

        # Only x/radial reach and z/height commands require IK. Pan, roll, and
        # gripper updates can pass through directly without perturbing the arm chain.
        if command.dx == 0.0 and command.dz == 0.0:
            targets = {
                "shoulder_pan": self._from_radians(target_pan_rad),
                "shoulder_lift": float(base.get("shoulder_lift", joints["shoulder_lift"])),
                "elbow_flex": float(base.get("elbow_flex", joints["elbow_flex"])),
                "wrist_flex": float(base.get("wrist_flex", joints["wrist_flex"])),
                "wrist_roll": float(wrist_roll_base + command.d_rot),
                "gripper": float(gripper),
            }
            return targets

        current_pose = self.forward(observation)
        current_reach = math.hypot(current_pose.x, current_pose.y)
        target_reach = max(0.0, current_reach + command.dx)
        target_position = [
            target_reach * math.cos(target_pan_rad),
            target_reach * math.sin(target_pan_rad),
            current_pose.z + command.dz,
        ]
        logger.info(
            "IK target_position=%s command=%s current_pose=%s target_pan_rad=%s target_reach=%s",
            target_position,
            command,
            current_pose,
            target_pan_rad,
            target_reach,
        )
        # Build IK arrays sized for every DOF in the model so the null-space
        # solver receives correctly aligned constraints. Arrays sized smaller
        # than numDofs cause PyBullet to apply constraints to the wrong joints.
        wrist_roll_rad = self._to_radians(joints["wrist_roll"])
        lower_limits: list[float] = []
        upper_limits: list[float] = []
        joint_ranges: list[float] = []
        rest_poses: list[float] = []

        for dof_name, (urdf_lower, urdf_upper) in zip(
            self._dof_joint_names, self._dof_urdf_limits
        ):
            if dof_name == "shoulder_pan":
                lower_limits.append(target_pan_rad)
                upper_limits.append(target_pan_rad)
                joint_ranges.append(0.0)
                rest_poses.append(target_pan_rad)
            elif dof_name == "wrist_roll":
                lower_limits.append(wrist_roll_rad)
                upper_limits.append(wrist_roll_rad)
                joint_ranges.append(0.0)
                rest_poses.append(wrist_roll_rad)
            elif dof_name in joints:
                # Arm joint: use [-π, π] so the null-space is not artificially
                # clamped below the URDF limits PyBullet enforces as hard bounds.
                lower_limits.append(-math.pi)
                upper_limits.append(math.pi)
                joint_ranges.append(2.0 * math.pi)
                rest_poses.append(self._to_radians(float(joints[dof_name])))
            else:
                # Non-arm DOF (e.g. gripper jaw): constrain within URDF limits,
                # rest at midpoint so the solver doesn't chase arbitrary values.
                j_range = max(0.0, urdf_upper - urdf_lower)
                lower_limits.append(urdf_lower)
                upper_limits.append(urdf_upper)
                joint_ranges.append(j_range)
                rest_poses.append((urdf_lower + urdf_upper) / 2.0)

        self._reset_state([rest_poses[self._dof_index[n]] for n in self.joint_names])

        solution = self._pybullet.calculateInverseKinematics(
            self._robot_id,
            self._end_effector_index,
            targetPosition=target_position,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=500,
            residualThreshold=1e-4,
            physicsClientId=self._client,
        )

        targets = {
            "shoulder_pan": self._from_radians(target_pan_rad),
            "shoulder_lift": self._from_radians(solution[self._dof_index["shoulder_lift"]]),
            "elbow_flex": self._from_radians(solution[self._dof_index["elbow_flex"]]),
            "wrist_flex": self._from_radians(solution[self._dof_index["wrist_flex"]]),
            "wrist_roll": float(wrist_roll_base + command.d_rot),
            "gripper": float(gripper),
        }
        logger.info("IK solved targets=%s", targets)
        return targets

    def _reset_state(self, joint_positions_rad: list[float]) -> None:
        for idx, joint_index in enumerate(self._joint_indices[:5]):
            self._pybullet.resetJointState(
                self._robot_id,
                joint_index,
                joint_positions_rad[idx],
                physicsClientId=self._client,
            )

    def _build_generated_urdf(self) -> str:
        self._tempdir = tempfile.TemporaryDirectory(prefix="so101-pybullet-")
        urdf_path = Path(self._tempdir.name) / "so101_generated.urdf"
        urdf_path.write_text(_generated_so101_urdf(self.kinematics), encoding="utf-8")
        return str(urdf_path)

    def _to_radians(self, angle: float) -> float:
        if self.use_degrees:
            return math.radians(angle)
        return float(angle)

    def _from_radians(self, angle: float) -> float:
        if self.use_degrees:
            return math.degrees(angle)
        return float(angle)


def _generated_so101_urdf(kinematics: ArmKinematics) -> str:
    return f"""<?xml version="1.0"?>
<robot name="so101_generated">
  <link name="base"/>
  <link name="pan_link"/>
  <link name="upper_link"/>
  <link name="forearm_link"/>
  <link name="tool_link"/>
  <link name="tool_tip"/>

  <joint name="1" type="revolute">
    <parent link="base"/>
    <child link="pan_link"/>
    <origin xyz="0 0 {kinematics.base_height}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>

  <joint name="2" type="revolute">
    <parent link="pan_link"/>
    <child link="upper_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>

  <joint name="3" type="revolute">
    <parent link="upper_link"/>
    <child link="forearm_link"/>
    <origin xyz="{kinematics.upper_arm_length} 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>

  <joint name="4" type="revolute">
    <parent link="forearm_link"/>
    <child link="tool_link"/>
    <origin xyz="{kinematics.forearm_length} 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>

  <joint name="5" type="revolute">
    <parent link="tool_link"/>
    <child link="tool_tip"/>
    <origin xyz="{kinematics.tool_length} 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _parse_origin_xyz(joint: ET.Element) -> tuple[float, float, float]:
    origin = joint.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0)
    return _parse_xyz(origin.attrib.get("xyz", "0 0 0"))


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = value.split()
    if len(parts) != 3:
        raise ValueError(f"Expected three xyz components, got: {value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _vector_length(vector: tuple[float, float, float]) -> float:
    x, y, z = vector
    return math.sqrt(x * x + y * y + z * z)
