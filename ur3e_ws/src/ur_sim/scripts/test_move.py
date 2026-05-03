#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from rclpy.executors import MultiThreadedExecutor
import time
import threading
from pymoveit2 import MoveIt2

def main():
    rclpy.init()
    node = Node("test_move_node")
    logger = get_logger("test_move_node")
    logger.info("Initializing MoveIt 2 interface...")

    # Spin the node in a separate thread
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()

    # Wait for MoveIt to be ready
    time.sleep(2.0)

    arm = MoveIt2(
        node=node,
        joint_names=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        base_link_name="base_link",
        end_effector_name="tool0",
        group_name="ur3e",
    )

    # Simplified 'Home' configuration (all zeros or a safe pose)
    # Using a safe pose similar to 'up' or 'home'
    # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    # Values in radians
    home_config = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0] 

    logger.info("Moving to HOME configuration...")
    arm.move_to_configuration(home_config)
    arm.wait_until_executed()
    logger.info("Reached HOME configuration.")

    time.sleep(1.0)

    # Move to a second arbitrary position to verify control
    target_config = [1.57, -1.57, 1.57, -1.57, 1.57, 0.0]
    logger.info("Moving to TARGET configuration...")
    arm.move_to_configuration(target_config)
    arm.wait_until_executed()
    logger.info("Reached TARGET configuration.")

    rclpy.shutdown()
    spin_thread.join()

if __name__ == "__main__":
    main()
