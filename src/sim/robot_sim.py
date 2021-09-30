import queue

import numpy as np

from ..command import Command
from ..controller import Controller


class XArmSim:
    def __init__(
        self,
        bullet_client,
        controller: Controller,
        joint_positions: list = None,
    ):
        self.bullet_client = bullet_client
        self.controller = controller

        if joint_positions is not None:
            self.controller.joint_positions = joint_positions

        # default load robot_sim model
        pybullet_flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.robot_sim = self.bullet_client.loadURDF(
            "xarm/xarm6_robot.urdf",
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=True,
            flags=pybullet_flags,
        )

        # set robot_sim to base position
        for joint_num in range(1, self.controller.DOFs + 1):
            self.bullet_client.changeDynamics(
                self.robot_sim, joint_num, linearDamping=0, angularDamping=0
            )
            info = self.bullet_client.getJointInfo(self.robot_sim, joint_num)

            jointType = info[2]
            if jointType == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(
                    self.robot_sim,
                    joint_num,
                    self.controller.joint_positions[joint_num - 1],
                )
            elif jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(
                    self.robot_sim,
                    joint_num,
                    self.controller.joint_positions[joint_num - 1],
                )

        # get cartesian position
        self.controller.cartesian_pos = self.get_cartesian_pos()
        self.controller.future_cartesian_pos = self.controller.cartesian_pos

        self.move(self.controller.joint_positions)

    def get_cartesian_pos(self, compute=True):
        end_effector_state = self.bullet_client.getLinkState(
            self.robot_sim,
            self.controller.end_effector_index,
            computeForwardKinematics=compute,
        )
        end_effector_xyz = end_effector_state[4]
        end_effector_rpy = self.bullet_client.getEulerFromQuaternion(
            end_effector_state[5]
        )
        cartesian_pos = end_effector_xyz + end_effector_rpy
        # print(cartesian_pos)
        return Command(*cartesian_pos, is_radian=True)

    def format_cartesian_pos(self, in_radians: bool = True) -> str:
        xyz = self.controller.cartesian_pos[:3]
        rpy = self.controller.cartesian_pos[3:]

        ret = ""
        ret += ", ".join([f"{coord * 1000:.2f}" for coord in xyz])
        ret += ", "
        ret += ", ".join(
            [
                f"{coord:.2f}" if in_radians else f"{np.rad2deg(coord):.2f}"
                for coord in rpy
            ]
        )

        return ret

    def run_ik(self, xyzrpy: list):
        xyz = xyzrpy[:3]
        rpy = xyzrpy[3:]

        rpy_quaternion = self.bullet_client.getQuaternionFromEuler(rpy)

        if self.controller.use_null_space:
            # target_joint_positions = self.bullet_client.calculateInverseKinematics(
            #     self.robot_sim,
            #     self.controller.end_effector_index,
            #     xyz,
            #     rpy_quaternion,
            #     lowerLimits=self.controller.ll,
            #     upperLimits=self.controller.ul,
            #     jointRanges=self.controller.jr,
            #     restPoses=self.controller.rp,
            #     # restPoses=np.array(self.controller.joint_positions).tolist(),
            #     residualThreshold=1e-5,
            #     maxNumIterations=50,
            # )
            pass
        else:
            target_joint_positions = self.bullet_client.calculateInverseKinematics(
                self.robot_sim,
                self.controller.end_effector_index,
                xyz,
                rpy_quaternion,
                maxNumIterations=50,
            )

        return target_joint_positions

    def move(self, target_joint_positions: list):
        if self.controller.use_dynamics:
            for i in range(self.controller.DOFs):
                self.bullet_client.setJointMotorControl2(
                    self.robot_sim,
                    i + 1,
                    self.bullet_client.POSITION_CONTROL,
                    target_joint_positions[i],
                    force=5 * 240.0,
                )
        else:
            for i in range(self.controller.DOFs):
                self.bullet_client.resetJointState(
                    self.robot_sim, i + 1, target_joint_positions[i]
                )

        if self.controller.move_real:
            self.controller.arm_real.set_servo_angle_j(  # type: ignore[union-attr]
                target_joint_positions, is_radian=True
            )

    def reset(self):
        pass

    def step(self):
        try:
            target_position = self.controller.decomposed_command_queue.get(block=False)
            # print(target_joint_positions)

            self.controller.decomposed_command_queue.task_done()
        except queue.Empty:
            return

        # xyzrpy = [
        #     0.2069,  # in meters
        #     0.0,  # in meters
        #     0.2587,  # in meters
        #     math.pi,  # in radians
        #     0.0,  # in radians
        #     0.0,  # in radians
        # ]
        # jointPoses = self.run_ik(xyzrpy)
        # print("jointPoses=",jointPoses)

        target_joint_positions = self.run_ik(target_position)

        self.move(target_joint_positions)

        self.controller.joint_positions = target_joint_positions
        self.controller.cartesian_pos = self.get_cartesian_pos(compute=True)

        self.bullet_client.addUserDebugLine(
            self.controller.cartesian_pos[:3],
            target_position[:3],
        )

        # print(old_cartesian_pos[:3], "\n", self.cartesian_pos[:3])
