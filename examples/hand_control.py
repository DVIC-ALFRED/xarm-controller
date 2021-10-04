import os
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import pybullet as p
import pybullet_data as pd
import xarm_hand_control.processing.process as xhcpp

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.sim.robot_sim as xarm_sim
from src.command import Command
from src.controller import Controller
from src.real.robot_real import XArmReal
from src.event import subscribe, post_event

VIDEO_PATH = "/dev/video0"
MOVE_ARM = False
ARM_IP = "172.21.72.200"


def worker(robot: Any, time_step: float) -> None:
    """Thread to run simulation."""

    while 1:
        robot.step()
        p.stepSimulation()
        time.sleep(time_step)


def send_command(command: Command) -> None:
    """Send extracted command to robot."""

    post_event("new_command", command)


def coords_extracter(controller: Controller):
    """Exctract coords to send command to robot.
    To be executed inside of xarm_hand_control module."""

    SKIPPED_COMMANDS = 5
    COEFF = 22

    current = [0]

    def coords_to_command(data: Any):
        current[0] += 1
        if current[0] < SKIPPED_COMMANDS:
            return

        current[0] = 0

        x = data[0] * COEFF / 1000
        z = data[1] * COEFF / 1000

        # speed = np.linalg.norm(data, ord=2) * COEFF * 50
        # speed = int(speed)
        # # speed = np.log(speed) * COEFF
        # mvacc = speed * 10
        speed = 500
        mvacc = speed * 10

        curr_pos = controller.future_cartesian_pos
        # print(f"{curr_pos=}")

        command = Command(
            x=curr_pos.x + x,
            y=curr_pos.y,
            z=curr_pos.z + z,
            roll=curr_pos.roll,
            pitch=curr_pos.pitch,
            yaw=curr_pos.yaw,
            speed=speed,
            acc=mvacc,
            is_radian=curr_pos.is_radian,
            is_cartesian=True,
            is_relative=False,
        )

        # print(command)

        send_command(command)

    return coords_to_command


def setup_pybullet() -> None:
    """Setup PyBullet environment."""

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())


def main():
    """Example: move xArm with hand."""

    time_step = 1.0 / 60.0
    setup_pybullet()

    cap = cv2.VideoCapture(VIDEO_PATH)  # pylint: disable=no-member

    if MOVE_ARM:
        xarm_real = XArmReal(ARM_IP)
        xarm_real.connect_loop()
    else:
        xarm_real = None

    controller = Controller(move_real=MOVE_ARM, arm_real=xarm_real)
    robot_sim = xarm_sim.XArmSim(
        p,
        controller,
        joint_positions=np.deg2rad(np.array([-90, -65.1, -20.8, -0.6, -4.1, 0.6])),
    )

    worker_thread = threading.Thread(
        target=worker, args=[robot_sim, time_step], daemon=True
    )
    worker_thread.start()

    hand_control_thread = threading.Thread(
        target=xhcpp.loop,
        args=[
            cap,
        ],
        kwargs={"coords_extracter_func": coords_extracter(controller)},
        daemon=True,
    )
    hand_control_thread.start()

    # send_command(controller, Command(
    #     0,
    #     -0.2278,
    #     0.6439,
    #     0,
    #     -90,
    #     90,
    #     speed=10,
    #     is_radian=False,
    #     is_cartesian=True,
    #     is_relative=False,
    # ))

    hand_control_thread.join()


if __name__ == "__main__":
    main()
