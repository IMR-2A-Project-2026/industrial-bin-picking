#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
import time
from geometry_msgs.msg import PoseStamped, Pose
from pymoveit2 import MoveIt2
import math
import subprocess

def main():
    rclpy.init()
    node = Node("ur3e_commander_node")
    logger = get_logger("ur3e_commander_node")
    logger.info("Várakozás a MoveIt indítására...")
    time.sleep(5.0)

    arm = MoveIt2(
        node=node,
        joint_names=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        base_link_name="base_link",
        end_effector_name="tool0",
        group_name="ur3e",
    )

    gripper = MoveIt2(
        node=node,
        joint_names=["robotiq_85_left_knuckle_joint"],
        base_link_name="base_link",
        end_effector_name="tool0",
        group_name="robotiq_gripper",
    )

    touch_links = [
        "robotiq_85_right_finger_tip_link", "robotiq_85_right_finger_link",
        "robotiq_85_right_knuckle_link", "robotiq_85_right_inner_knuckle_link",
        "robotiq_85_left_finger_tip_link", "robotiq_85_left_finger_link",
        "robotiq_85_left_knuckle_link", "robotiq_85_left_inner_knuckle_link",
        "robotiq_85_base_link"
    ]

    CUBE_1_NAME = "box1"
    cube_1_size = [0.06, 0.06, 0.06]
    cube_1_pos = Pose()
    cube_1_pos.position.x = 0.58
    cube_1_pos.position.y = 0.38
    cube_1_pos.position.z = 0.03
    cube_1_pos.orientation.x = 0.0
    cube_1_pos.orientation.y = 0.0
    cube_1_pos.orientation.z = math.sin(0.5)
    cube_1_pos.orientation.w = math.cos(0.5)

    CUBE_2_NAME = "box2"
    cube_2_size = [0.06, 0.06, 0.06]
    cube_2_pos = Pose()
    cube_2_pos.position.x = 0.1
    cube_2_pos.position.y = 0.6
    cube_2_pos.position.z = 0.03
    cube_2_pos.orientation.x = 0.0
    cube_2_pos.orientation.y = 0.0
    cube_2_pos.orientation.z = math.sin(0.1)
    cube_2_pos.orientation.w = math.cos(0.5)

    CUBE_3_NAME = "box3"
    cube_3_size = [0.06, 0.06, 0.06]
    cube_3_pos = Pose()
    cube_3_pos.position.x = -0.2
    cube_3_pos.position.y = 0.3
    cube_3_pos.position.z = 0.03
    cube_3_pos.orientation.x = 0.0
    cube_3_pos.orientation.y = 0.0
    cube_3_pos.orientation.z = math.sin(-0.25)
    cube_3_pos.orientation.w = math.cos(0.5)

    position_list_1 = [cube_1_pos.position.x,
                       cube_1_pos.position.y,
                       cube_1_pos.position.z]

    orientation_list_1 = [cube_1_pos.orientation.x,
                          cube_1_pos.orientation.y,
                          cube_1_pos.orientation.z,
                          cube_1_pos.orientation.w]

    arm.add_collision_box(
        CUBE_1_NAME,
        cube_1_size,
        "box",
        position_list_1,
        orientation_list_1
    )

    position_list_2 = [cube_2_pos.position.x,
                       cube_2_pos.position.y,
                       cube_2_pos.position.z]

    orientation_list_2 = [cube_2_pos.orientation.x,
                          cube_2_pos.orientation.y,
                          cube_2_pos.orientation.z,
                          cube_2_pos.orientation.w]

    arm.add_collision_box(
        CUBE_2_NAME,
        cube_2_size,
        "box",
        position_list_2,
        orientation_list_2
    )

    position_list_3 = [cube_3_pos.position.x,
                       cube_3_pos.position.y,
                       cube_3_pos.position.z]

    orientation_list_3 = [cube_3_pos.orientation.x,
                          cube_3_pos.orientation.y,
                          cube_3_pos.orientation.z,
                          cube_3_pos.orientation.w]

    arm.add_collision_box(
        CUBE_3_NAME,
        cube_3_size,
        "box",
        position_list_3,
        orientation_list_3
    )

    cube1_1 = [-2.46, -3.14, 0.26, -3.23, -1.83, 0.0]
    arm.move_to_configuration(cube1_1)
    arm.wait_until_executed()

    cube1_2 = [-2.46, -3.14, 0.0, -3.23, -1.83, 0.0]
    arm.move_to_configuration(cube1_2)
    arm.wait_until_executed()

    time.sleep(1.0)
    arm.attach_collision_object(CUBE_1_NAME, "tool0", touch_links)
    time.sleep(1.0)

    close = [0.5]
    gripper.move_to_configuration(close)
    gripper.wait_until_executed()

    cube1_3 = [-2.46, -3.14, 0.26, -3.23, -1.83, 0.0]
    arm.move_to_configuration(cube1_3)
    arm.wait_until_executed()

    base1_1 = [-4.71, -2.39, -1.57, -2.39, -1.57, 0.0]
    arm.move_to_configuration(base1_1)
    arm.wait_until_executed()

    open = [0.0]
    gripper.move_to_configuration(open)
    gripper.wait_until_executed()

    time.sleep(1.0)
    arm.detach_collision_object(CUBE_1_NAME)
    time.sleep(1.0)

    base1_2 = [-4.71, -2.22, -1.92, -2.22, -1.57, 0.0]
    arm.move_to_configuration(base1_2)
    arm.wait_until_executed()

    cube2_1 = [-1.6, -2.35, -1.57, -2.35, -1.74, 0.0]
    arm.move_to_configuration(cube2_1)
    arm.wait_until_executed()

    cube2_2 = [-1.6, -2.55, -1.37, -2.35, -1.74, 0.0]
    arm.move_to_configuration(cube2_2)
    arm.wait_until_executed()

    time.sleep(1.0)
    arm.attach_collision_object(CUBE_2_NAME, "tool0", touch_links)
    time.sleep(1.0)

    gripper.move_to_configuration(close)
    gripper.wait_until_executed()

    cube2_3 = [-1.6, -2.55, -1.37, -2.09, -1.74, 0.0]
    arm.move_to_configuration(cube2_3)
    arm.wait_until_executed()

    base2_1 = [-4.73, -2.21, -1.55, -2.53, -1.59, 0.0]
    arm.move_to_configuration(base2_1)
    arm.wait_until_executed()

    gripper.move_to_configuration(open)
    gripper.wait_until_executed()

    time.sleep(1.0)
    arm.detach_collision_object(CUBE_2_NAME)
    time.sleep(1.0)

    base2_2 = [-4.73, -2.21, -1.55, -2.2, -1.59, 0.0]
    arm.move_to_configuration(base2_2)
    arm.wait_until_executed()

    cube3_1 = [1.76, -1.48, 1.69, -1.76, -1.6, 2.3]
    arm.move_to_configuration(cube3_1)
    arm.wait_until_executed()

    cube3_2 = [1.75, -1.38, 1.94, -2.13, -1.6, 2.3]
    arm.move_to_configuration(cube3_2)
    arm.wait_until_executed()

    time.sleep(1.0)
    arm.attach_collision_object(CUBE_3_NAME, "tool0", touch_links)
    time.sleep(1.0)

    gripper.move_to_configuration(close)
    gripper.wait_until_executed()

    cube3_3 = [1.76, -1.48, 1.69, -1.76, -1.6, 2.3]
    arm.move_to_configuration(cube3_3)
    arm.wait_until_executed()

    base3_1 = [1.57, -2.18, -1.52, 3.75, -1.57, 0.0]
    arm.move_to_configuration(base3_1)
    arm.wait_until_executed()

    gripper.move_to_configuration(open)
    gripper.wait_until_executed()

    time.sleep(1.0)
    arm.detach_collision_object(CUBE_3_NAME)
    time.sleep(1.0)

    gripper.move_to_configuration(open)
    gripper.wait_until_executed()

    base3_2 = [1.57, -2.0, -1.83, 3.87, -1.57, 0.0]
    arm.move_to_configuration(base3_2)
    arm.wait_until_executed()

    stand = [0.0, -1.57, 0.0, 0.0, 0.0, 1.57]
    arm.move_to_configuration(stand)
    arm.wait_until_executed()

    rclpy.shutdown()

if __name__ == "__main__":
    main()

