from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sapien.core as sapien
from gap_rl import DESCRIPTION_DIR
from gap_rl.agents.base_agent import BaseAgent, parse_urdf_config
from gap_rl.agents.configs.unitree_g1_dualarm import defaults
from gap_rl.utils.common import compute_angle_between
from gap_rl.utils.geometry import transform_points
from gap_rl.utils.sapien_utils import (
    get_entity_by_name,
    get_multi_pairwise_contact_impulse,
    get_pairwise_contact_impulse,
)
from gap_rl.utils.trimesh_utils import get_actor_mesh


class UnitreeG1DualArm(BaseAgent):
    """Dual-arm Unitree G1 platform with independent grippers."""

    _config: defaults.UnitreeG1DualArmDefaultConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hands: List[str] = list(self.config.hands)
        self.primary_hand: str = self.config.primary_hand
        self.ee_links: Dict[str, sapien.Link] = {
            hand: get_entity_by_name(self.robot.get_links(), link_name)
            for hand, link_name in self.config.ee_link_names.items()
        }
        self.primary_ee_link: sapien.Link = self.ee_links[self.primary_hand]
        self.secondary_ee_links: Dict[str, sapien.Link] = {
            hand: link
            for hand, link in self.ee_links.items()
            if hand != self.primary_hand
        }
        self.finger_links: Dict[str, Tuple[sapien.LinkBase, sapien.LinkBase]] = {}
        self.finger_meshes: Dict[str, Tuple] = {}
        for hand, link_names in self.config.finger_link_names.items():
            finger1 = get_entity_by_name(self.robot.get_links(), link_names[0])
            finger2 = get_entity_by_name(self.robot.get_links(), link_names[1])
            self.finger_links[hand] = (finger1, finger2)
            mesh1 = get_actor_mesh(finger1, False)
            mesh2 = get_actor_mesh(finger2, False)
            self.finger_meshes[hand] = (mesh1, mesh2)
        self.finger_sizes = dict(self.config.finger_sizes)
        self.num_grippers: int = len(self.hands)
        self.gripper_widths: Dict[str, float] = dict(self.config.gripper_widths)
        self.primary_gripper_width: float = self.gripper_widths[self.primary_hand]

    @classmethod
    def get_default_config(cls):
        return defaults.UnitreeG1DualArmDefaultConfig()

    # ------------------------------------------------------------------ #
    # BaseAgent hooks
    # ------------------------------------------------------------------ #
    def _load_articulation(self):
        loader = self.scene.create_urdf_loader()
        urdf_path = str(self.urdf_path)
        urdf_path = urdf_path.format(description=DESCRIPTION_DIR)
        urdf_config = parse_urdf_config(self.urdf_config, self.scene)

        builder = loader.load_file_as_articulation_builder(urdf_path, urdf_config)
        for link_builder in builder.get_link_builders():
            link_builder.set_collision_groups(1, 1, 2, 0)
        self.robot = builder.build(fix_root_link=self.fix_root_link)
        assert self.robot is not None, f"Fail to load URDF from {urdf_path}"
        self.robot.set_name(Path(urdf_path).stem)

        self.robot_link_ids = [link.get_id() for link in self.robot.get_links()]
        self.robot_collision_actors = [
            actor
            for actor in self.robot.get_links()
            if actor.get_collision_shapes()
        ]
        active_joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.gripper_joint_ids = [
            active_joint_names.index(joint_name)
            for joint_name in self.config.gripper_joint_names
        ]
        self.gripper_joint_ids_map = {
            hand: [active_joint_names.index(name) for name in joint_names]
            for hand, joint_names in self.config.gripper_joint_map.items()
        }

    def _after_init(self):
        pass

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _get_hand_fingers(self, hand: Optional[str] = None):
        hand = hand or self.primary_hand
        if hand not in self.finger_links:
            raise ValueError(f"Unknown hand '{hand}'. Available hands: {self.hands}")
        return self.finger_links[hand]

    def _get_hand_width(self, hand: Optional[str] = None) -> float:
        hand = hand or self.primary_hand
        return self.gripper_widths[hand]

    # ------------------------------------------------------------------ #
    # Grasp & contact checks
    # ------------------------------------------------------------------ #
    def check_grasp(
        self,
        actor: sapien.ActorBase,
        min_impulse: float = 1e-6,
        max_angle: float = 85,
        hand: Optional[str] = None,
    ) -> bool:
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()
        finger1_link, finger2_link = self._get_hand_fingers(hand)

        limpulse = get_pairwise_contact_impulse(contacts, finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, finger2_link, actor)

        ldirection = -finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = -finger2_link.pose.to_transformation_matrix()[:3, 1]

        langle = compute_angle_between(ldirection, limpulse)
        rangle = compute_angle_between(rdirection, rimpulse)

        lflag = (
            np.linalg.norm(limpulse) >= min_impulse
            and np.rad2deg(langle) <= max_angle
        )
        rflag = (
            np.linalg.norm(rimpulse) >= min_impulse
            and np.rad2deg(rangle) <= max_angle
        )
        return lflag and rflag

    def check_contact_fingers(
        self,
        actor: sapien.ActorBase,
        min_impulse: float = 1e-6,
        hand: Optional[str] = None,
    ) -> Tuple[bool, bool]:
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()
        finger1_link, finger2_link = self._get_hand_fingers(hand)

        limpulse = get_pairwise_contact_impulse(contacts, finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, finger2_link, actor)

        return (
            np.linalg.norm(limpulse) >= min_impulse,
            np.linalg.norm(rimpulse) >= min_impulse,
        )

    def check_contact(
        self,
        actor: sapien.ActorBase,
        min_impulse: float = 1e-6,
    ) -> Tuple[bool, np.ndarray]:
        contacts = self.scene.get_contacts()
        multi_impluse = np.linalg.norm(
            get_multi_pairwise_contact_impulse(contacts, self.robot_collision_actors, actor),
            axis=-1,
        )
        is_agent_contact = any(multi_impluse >= min_impulse)
        return is_agent_contact, multi_impluse

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #
    def sample_ee_coords(
        self, num_sample: int = 10, hand: Optional[str] = None
    ) -> np.ndarray:
        """Uniformly sample points on the finger meshes for dense reward."""

        finger1_link, finger2_link = self._get_hand_fingers(hand)
        mesh1, mesh2 = self.finger_meshes[hand or self.primary_hand]

        finger1_points = transform_points(
            finger1_link.get_pose().to_transformation_matrix(), mesh1.sample(num_sample)
        )
        finger2_points = transform_points(
            finger2_link.get_pose().to_transformation_matrix(), mesh2.sample(num_sample)
        )
        return np.stack((finger1_points, finger2_points))
