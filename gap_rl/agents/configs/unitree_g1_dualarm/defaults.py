from collections import OrderedDict

from gap_rl import DESCRIPTION_DIR
from gap_rl.agents.controllers import (
    PDEEPoseEulerControllerConfig,
    PDJointPosMimicControllerConfig,
    deepcopy_dict,
)
from gap_rl.sensors.camera import CameraConfig


class UnitreeG1DualArmDefaultConfig:
    """Default configuration for the Unitree G1 dual-arm platform."""

    def __init__(self) -> None:
        self.urdf_path = f"{DESCRIPTION_DIR}/unitree_g1/unitree_g1_dualarm.urdf"
        self.urdf_config = {}

        # ------------------------------------------------------------------ #
        # Robot structure metadata
        # ------------------------------------------------------------------ #
        self.hands = ("left", "right")
        self.primary_hand = "right"
        self.ee_link_names = OrderedDict(
            left="left_grasp_link",
            right="right_grasp_link",
        )
        # Backward compatibility with single-tool pipelines
        self.ee_link_name = self.ee_link_names[self.primary_hand]

        self.finger_link_names = OrderedDict(
            left=("left_inner_finger_pad", "left_inner_finger_pad_2"),
            right=("right_inner_finger_pad", "right_inner_finger_pad_2"),
        )
        self.finger_sizes = OrderedDict(
            left=(0.04, 0.01, 0.02),
            right=(0.04, 0.01, 0.02),
        )

        # Joint definitions (ordered to match physical kinematics)
        self.arm_joint_map = OrderedDict(
            left=[
                "left_shoulder_yaw_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_elbow_pitch_joint",
                "left_wrist_pitch_joint",
                "left_wrist_roll_joint",
            ],
            right=[
                "right_shoulder_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_elbow_pitch_joint",
                "right_wrist_pitch_joint",
                "right_wrist_roll_joint",
            ],
        )
        self.arm_joint_names = [
            joint for joints in self.arm_joint_map.values() for joint in joints
        ]

        self.gripper_joint_map = OrderedDict(
            left=[
                "left_gripper_finger_left_joint",
                "left_gripper_finger_right_joint",
            ],
            right=[
                "right_gripper_finger_left_joint",
                "right_gripper_finger_right_joint",
            ],
        )
        self.gripper_joint_names = [
            joint for joints in self.gripper_joint_map.values() for joint in joints
        ]

        self.gripper_widths = OrderedDict(left=0.085, right=0.085)

        # ------------------------------------------------------------------ #
        # Controller parameters
        # ------------------------------------------------------------------ #
        self.arm_stiffness = 800.0
        self.arm_damping = 40.0
        self.arm_force_limit = 120.0
        self.ee_delta = 0.01
        self.rot_euler_bound = 0.05

        self.gripper_stiffness = 600.0
        self.gripper_damping = 30.0
        self.gripper_force_limit = 40.0

        # On-board sensing: keep a wrist camera aligned with the primary hand
        self.cameras = [
            CameraConfig(
                uid="right_hand_realsense",
                p=[0.0, 0.0, 0.0],
                q=[1, 0, 0, 0],
                width=320,
                height=180,
                fov=0.758,
                near=0.01,
                far=5.0,
                actor_uid="right_camera_hand_link",
                hide_link=False,
            )
        ]

    @property
    def controllers(self):
        # Cartesian pose controller for each arm
        left_arm_pose = PDEEPoseEulerControllerConfig(
            self.arm_joint_map["left"],
            -self.ee_delta,
            self.ee_delta,
            self.rot_euler_bound,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            cache_size=3,
            frame="ee",
            smooth=False,
            ee_link=self.ee_link_names["left"],
            normalize_action=False,
        )
        right_arm_pose = PDEEPoseEulerControllerConfig(
            self.arm_joint_map["right"],
            -self.ee_delta,
            self.ee_delta,
            self.rot_euler_bound,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            cache_size=3,
            frame="ee",
            smooth=False,
            ee_link=self.ee_link_names["right"],
            normalize_action=False,
        )

        left_gripper_pd = PDJointPosMimicControllerConfig(
            self.gripper_joint_map["left"],
            0.0,
            self.gripper_widths["left"],
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=0.2,
            normalize_action=False,
        )
        right_gripper_pd = PDJointPosMimicControllerConfig(
            self.gripper_joint_map["right"],
            0.0,
            self.gripper_widths["right"],
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=0.2,
            normalize_action=False,
        )

        controller_configs = dict(
            dual_arm_cartesian=dict(
                left_arm=left_arm_pose,
                right_arm=right_arm_pose,
                left_gripper=left_gripper_pd,
                right_gripper=right_gripper_pd,
            ),
            right_arm_cartesian=dict(
                right_arm=right_arm_pose,
                left_gripper=left_gripper_pd,
                right_gripper=right_gripper_pd,
            ),
        )

        return deepcopy_dict(controller_configs)
