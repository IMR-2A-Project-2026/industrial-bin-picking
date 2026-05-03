#!/usr/bin/env python3
"""
pick_and_place.py — Single-point pick + place for UR3e (REAL ROBOT)

Safety:
  - Table collision object prevents planning through the surface.
  - NO ceiling — UR3e links exceed 0.35 m in normal configs; ceiling
    blocks all valid start states.
  - Elbow-flip prevention via Cartesian path on descend/retreat.

Fixes vs previous version:
  1. Controller rejection (STATUS_ABORTED on every move after the first):
       pymoveit2 was sending goals in MERGE/continuation mode.
       scaled_joint_trajectory_controller rejects those when no
       trajectory is actively running.  Fix: cancel any live goal
       before each new move so the controller always sees REPLACE.

  2. Robot hit the ground:
       PICK_Z = 0.03 m is the cube *centre*.  tool0 (no gripper, just
       a 5 mm camera bracket) at Z = 0.03 is essentially at the table
       surface.  PICK_Z is now raised to SAFE_PICK_Z = 0.10 m until
       you verify the TCP position with:
           ros2 topic echo /tcp_pose_broadcaster/pose --once
       after the approach step, then lower it in small increments.

  3. allowed_start_tolerance raised to 0.05 rad via ROS2 parameter
     override so a small amount of joint drift between retries does
     not abort the trajectory validation.

Place sequence (added):
  - STEP 6 : Place approach-1  — joint-space move to a safe intermediate
             posture [0, -1.57, -1.57, -1.57, 1.57, 0.0] (radians).
             This moves the arm away from the pick side before swinging
             toward the drop zone, avoiding any table/object collision.
  - STEP 7 : Place approach-2  — joint-space move to
             [0°, -140°, -50°, -75°, 90°, 0°] converted to radians.
             The arm is now lined up above the drop location.
  - STEP 8 : Place (lower)     — joint-space move to the final drop
             posture [0°, -155°, -54°, -58°, 90°, 0°] in radians.
             Gripper open goes here.
  - STEP 9 : Retreat to approach-2 — reverse back to the pre-place
             posture before swinging home, keeping the same arm config.
  - STEP 10: Return home (joint space).

  All joint moves use move_joints() which calls cancel_and_wait() first,
  matching the pattern already used for pick.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
import time
import threading
from geometry_msgs.msg import Pose
from pymoveit2 import MoveIt2
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these before each run
# ═══════════════════════════════════════════════════════════════════════

GROUP_NAME = "ur3e"

# Pick point in robot base_link frame
PICK_X =  0.3 - 0.4       # -0.10 m
PICK_Y = -0.3 - (-0.68)   #  0.38 m

# ⚠️  SAFE HEIGHT — do NOT lower this until you have confirmed the TCP
#     position at approach with:
#         ros2 topic echo /tcp_pose_broadcaster/pose --once
#
#     Previous value was 0.03 m (cube centre), which drove tool0 into
#     the table surface.  Start at 0.10 m and lower in 2 cm steps.
PICK_Z = 0.10

APPROACH_CLEARANCE = 0.20  # metres above PICK_Z

# Camera mount: 11.5 cm × 6.5 cm × 0.5 cm, offset 2.5 mm below tool0
CAMERA_MOUNT_SIZE   = [0.115, 0.065, 0.005]
CAMERA_MOUNT_OFFSET = [0.0,   0.0,   -0.0025]

VELOCITY_SCALE     = 0.05
ACCELERATION_SCALE = 0.05
PLANNING_TIME_SEC  = 15.0
POSITION_TOL       = 0.01
ORIENTATION_TOL    = 0.1

# Table surface at Z=0 in base_link frame
TABLE_Z = 0.00

# ── Place joint configurations ─────────────────────────────────────────
# Intermediate safe posture between pick side and drop zone.
# Given directly in radians.
PLACE_APPROACH1_JOINTS = [0.0, -1.57, -1.57, -1.57, 1.57, 0.0]

# Pre-place posture: [0°, -140°, -50°, -75°, 90°, 0°] → radians
PLACE_APPROACH2_JOINTS = [
    math.radians(0),
    math.radians(-140),
    math.radians(-50),
    math.radians(-75),
    math.radians(90),
    math.radians(0),
]

# Final drop posture: [0°, -155°, -54°, -58°, 90°, 0°] → radians
PLACE_DROP_JOINTS = [
    math.radians(0),
    math.radians(-155),
    math.radians(-54),
    math.radians(-58),
    math.radians(90),
    math.radians(0),
]


# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
BASE_LINK    = "base_link"
END_EFFECTOR = "tool0"
QUAT_DOWN    = dict(x=1.0, y=0.0, z=0.0, w=0.0)
HOME_JOINTS  = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def make_pose(x: float, y: float, z: float) -> Pose:
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = float(z)
    p.orientation.x = QUAT_DOWN["x"]
    p.orientation.y = QUAT_DOWN["y"]
    p.orientation.z = QUAT_DOWN["z"]
    p.orientation.w = QUAT_DOWN["w"]
    return p


def cancel_and_wait(arm: MoveIt2) -> None:
    """
    Cancel any active trajectory goal and wait for the controller to
    settle before sending the next goal.

    Why this is needed
    ------------------
    pymoveit2 sends FollowJointTrajectory goals with goal_time_tolerance
    set, which causes the UR driver to keep the action handle "open"
    briefly after the last point is reached.  When the next goal arrives
    during that window, ros_control sees an active goal and rejects the
    new one as a MERGE/continuation — which scaled_joint_trajectory_
    controller refuses when not in blending mode.

    Explicitly cancelling first guarantees the controller is idle and
    treats the incoming goal as a fresh REPLACE.
    """
    try:
        arm.cancel_execution()
    except Exception:
        pass   # no active goal — fine
    time.sleep(0.3)   # let the controller process the cancel


def setup_collision_scene(arm: MoveIt2) -> None:
    """
    Only a table slab — no ceiling (see module docstring).
    """
    for name in ["table", "ceiling", "floor"]:
        arm.remove_collision_object(name)
    time.sleep(0.4)

    # Table: 5 cm thick slab, top face at Z=TABLE_Z
    arm.add_collision_box(
        id="table",
        size=[2.0, 2.0, 0.05],
        frame_id=BASE_LINK,
        position=[0.0, 0.0, TABLE_Z - 0.025],
        quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    print(f"   🗺️   Table slab at Z={TABLE_Z:.2f}  (no ceiling)")
    time.sleep(1.0)


def attach_camera_mount(node: Node) -> None:
    pub = node.create_publisher(
        AttachedCollisionObject, "/attached_collision_object", 10
    )
    time.sleep(0.5)

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [float(d) for d in CAMERA_MOUNT_SIZE]

    box_pose = Pose()
    box_pose.position.x = float(CAMERA_MOUNT_OFFSET[0])
    box_pose.position.y = float(CAMERA_MOUNT_OFFSET[1])
    box_pose.position.z = float(CAMERA_MOUNT_OFFSET[2])
    box_pose.orientation.w = 1.0

    co = CollisionObject()
    co.header = Header()
    co.header.frame_id = END_EFFECTOR
    co.id = "camera_mount"
    co.primitives.append(box)
    co.primitive_poses.append(box_pose)
    co.operation = CollisionObject.ADD

    aco = AttachedCollisionObject()
    aco.link_name   = END_EFFECTOR
    aco.object      = co
    aco.touch_links = [END_EFFECTOR, "wrist_3_link"]

    pub.publish(aco)
    print(f"   📷  Camera mount attached  {CAMERA_MOUNT_SIZE}")
    time.sleep(1.0)


def move_safe(arm: MoveIt2, pose: Pose, label: str, retries: int = 3) -> bool:
    """Joint-space move with cancel-before-send to prevent goal rejection."""
    for attempt in range(1, retries + 1):
        print(f"   [{label}] attempt {attempt}/{retries} ...")
        cancel_and_wait(arm)   # ← key fix: clear any residual active goal
        try:
            arm.move_to_pose(pose)
            result = arm.wait_until_executed()
            if result is False:
                print(f"   [{label}] ❌  execution reported failure")
                time.sleep(2.0)
                continue
            print(f"   [{label}] ✅  reached")
            return True
        except Exception as exc:
            print(f"   [{label}] ❌  {str(exc)[:80]}")
            time.sleep(2.0)
    print(f"   [{label}] ⚠️   all retries exhausted")
    return False


def move_cartesian(arm: MoveIt2, pose: Pose, label: str, retries: int = 3) -> bool:
    """
    Cartesian straight-line move with cancel-before-send.

    Forces a linear TCP path → IK solver keeps current arm configuration
    → prevents elbow-flip wild arcs on descend/retreat.
    """
    for attempt in range(1, retries + 1):
        print(f"   [{label}] attempt {attempt}/{retries} (Cartesian) ...")
        cancel_and_wait(arm)   # ← key fix
        try:
            try:
                arm.move_to_pose(pose, cartesian=True)
            except TypeError:
                print(f"   [{label}] ℹ️   cartesian= kwarg not supported; using standard planner")
                arm.move_to_pose(pose)

            result = arm.wait_until_executed()
            if result is False:
                print(f"   [{label}] ❌  execution reported failure")
                time.sleep(2.0)
                continue
            print(f"   [{label}] ✅  reached")
            return True
        except Exception as exc:
            print(f"   [{label}] ❌  {str(exc)[:80]}")
            time.sleep(2.0)
    print(f"   [{label}] ⚠️   all retries exhausted")
    return False


def move_joints(arm: MoveIt2, joints: list, label: str, retries: int = 3) -> bool:
    """
    Joint-space move to an explicit joint configuration, with
    cancel-before-send to prevent goal rejection.

    Parameters
    ----------
    joints : list[float]
        Target joint angles in radians, ordered as JOINT_NAMES.
    """
    deg_str = "  ".join(f"{math.degrees(j):+.1f}°" for j in joints)
    for attempt in range(1, retries + 1):
        print(f"   [{label}] attempt {attempt}/{retries}  [{deg_str}]")
        cancel_and_wait(arm)
        try:
            arm.move_to_configuration(joints)
            result = arm.wait_until_executed()
            if result is False:
                print(f"   [{label}] ❌  execution reported failure")
                time.sleep(2.0)
                continue
            print(f"   [{label}] ✅  reached")
            return True
        except Exception as exc:
            print(f"   [{label}] ❌  {str(exc)[:80]}")
            time.sleep(2.0)
    print(f"   [{label}] ⚠️   all retries exhausted")
    return False


def move_home(arm: MoveIt2, label: str = "home") -> bool:
    print(f"   [{label}] moving to home ...")
    cancel_and_wait(arm)   # ← key fix
    arm.move_to_configuration(HOME_JOINTS)
    result = arm.wait_until_executed()
    if result is False:
        print(f"   [{label}] ❌  home move failed")
        return False
    print(f"   [{label}] ✅  home reached")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    rclpy.init()

    node = Node(
        "ur3e_pick_place",
        parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, False),
        ],
    )

    # ── MoveIt2 must be created BEFORE the executor starts spinning ──────
    # Avoids: RCLError: wait set index for status subscription is out of bounds
    print("🔧  Initialising MoveIt2...")
    arm = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=BASE_LINK,
        end_effector_name=END_EFFECTOR,
        group_name=GROUP_NAME,
    )
    arm.max_velocity_scaling_factor     = VELOCITY_SCALE
    arm.max_acceleration_scaling_factor = ACCELERATION_SCALE
    arm.planning_time                   = PLANNING_TIME_SEC
    arm.goal_position_tolerance         = POSITION_TOL
    arm.goal_orientation_tolerance      = ORIENTATION_TOL

    # Raise allowed_start_tolerance from default 0.01 to 0.05 rad.
    # The trajectory execution manager validates that the robot's current
    # joint state matches the trajectory start point within this tolerance.
    # With the 2 s retry delay, gravity/compliance can move joints ~0.02–0.04 rad,
    # which was causing "start point deviates" aborts on every retry.
    try:
        node.set_parameters([
            Parameter(
                "move_group.trajectory_execution.allowed_start_tolerance",
                Parameter.Type.DOUBLE,
                0.05,
            )
        ])
    except Exception:
        pass  # parameter may not exist on this distro — not fatal

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("⏳  Waiting for ROS2 / MoveIt2 to settle... (5 s)")
    time.sleep(5.0)

    print("🗺️   Setting up collision scene...")
    setup_collision_scene(arm)
    attach_camera_mount(node)
    time.sleep(1.0)

    pose_approach = make_pose(PICK_X, PICK_Y, PICK_Z + APPROACH_CLEARANCE)
    pose_pick     = make_pose(PICK_X, PICK_Y, PICK_Z)

    print(f"\n📍  Pick point (base_link frame):")
    print(f"    approach  X:{PICK_X:.3f}  Y:{PICK_Y:.3f}  Z:{PICK_Z + APPROACH_CLEARANCE:.3f}")
    print(f"    pick      X:{PICK_X:.3f}  Y:{PICK_Y:.3f}  Z:{PICK_Z:.3f}  ← verify before lowering further")
    print(f"    table     Z:{TABLE_Z:.3f}")
    print(f"\n⚠️   PICK_Z = {PICK_Z:.2f} m  (raised from 0.03 — verify TCP then lower in 2 cm steps)\n")

    print(f"📍  Place joint configs (degrees):")
    print(f"    approach-1  {[round(math.degrees(j),1) for j in PLACE_APPROACH1_JOINTS]}")
    print(f"    approach-2  {[round(math.degrees(j),1) for j in PLACE_APPROACH2_JOINTS]}")
    print(f"    drop        {[round(math.degrees(j),1) for j in PLACE_DROP_JOINTS]}")

    # ── STEP 1 ──────────────────────────────────────────────────────────
    print("\n▶  STEP 1 — Home (joint space)")
    move_home(arm, "home")
    time.sleep(0.5)

    # ── STEP 2 ──────────────────────────────────────────────────────────
    print(f"\n▶  STEP 2 — Approach ({APPROACH_CLEARANCE*100:.0f} cm above pick)")
    ok = move_safe(arm, pose_approach, "approach")
    if not ok:
        print("⛔  Approach failed — aborting safely.")
        move_home(arm, "abort-home")
        rclpy.shutdown()
        spin_thread.join()
        return

    print("\n   📡  Check TCP now — in another terminal:")
    print("       ros2 topic echo /tcp_pose_broadcaster/pose --once")
    print("   Expected Z ≈", round(PICK_Z + APPROACH_CLEARANCE, 3))
    time.sleep(0.5)

    # ── STEP 3 ──────────────────────────────────────────────────────────
    print("\n▶  STEP 3 — Descend (Cartesian straight line → no elbow flip)")
    ok = move_cartesian(arm, pose_pick, "descend")
    if not ok:
        print("⚠️   Descend failed — retreating.")
        move_cartesian(arm, pose_approach, "retreat-after-fail")
        move_home(arm, "abort-home")
        rclpy.shutdown()
        spin_thread.join()
        return

    print("\n   ⏸   At pick position — GRIPPER CLOSE goes here")
    time.sleep(1.5)

    # ── STEP 4 ──────────────────────────────────────────────────────────
    print("\n▶  STEP 4 — Retreat (Cartesian straight line back up)")
    ok = move_cartesian(arm, pose_approach, "retreat")
    if not ok:
        print("⚠️   Retreat failed — attempting emergency home.")
        move_home(arm, "abort-home")
        rclpy.shutdown()
        spin_thread.join()
        return
    time.sleep(0.5)

    # ── STEP 5 ──────────────────────────────────────────────────────────
    # Swing the arm to the safe intermediate posture before moving toward
    # the drop zone.  This avoids swinging a loaded arm through the pick
    # side objects.
    print("\n▶  STEP 5 — Place approach-1 (safe intermediate posture)")
    ok = move_joints(arm, PLACE_APPROACH1_JOINTS, "place-approach-1")
    if not ok:
        print("⚠️   Place approach-1 failed — retreating to home.")
        move_home(arm, "abort-home")
        rclpy.shutdown()
        spin_thread.join()
        return
    time.sleep(0.5)

    # ── STEP 6 ──────────────────────────────────────────────────────────
    # Move to the pre-place posture directly above the drop location.
    print("\n▶  STEP 6 — Place approach-2 (above drop location)")
    ok = move_joints(arm, PLACE_APPROACH2_JOINTS, "place-approach-2")
    if not ok:
        print("⚠️   Place approach-2 failed — retreating to home.")
        move_home(arm, "abort-home")
        rclpy.shutdown()
        spin_thread.join()
        return
    time.sleep(0.5)

    # ── STEP 7 ──────────────────────────────────────────────────────────
    # Lower into the final drop posture.
    print("\n▶  STEP 7 — Place / drop")
    ok = move_joints(arm, PLACE_DROP_JOINTS, "place-drop")
    if not ok:
        print("⚠️   Place drop failed — retreating to approach-2.")
        move_joints(arm, PLACE_APPROACH2_JOINTS, "place-retreat-fail")
        move_home(arm, "abort-home")
        rclpy.shutdown()
        spin_thread.join()
        return

    print("\n   ⏸   At place position — GRIPPER OPEN goes here")
    time.sleep(1.5)

    # ── STEP 8 ──────────────────────────────────────────────────────────
    # Retreat back to approach-2 before swinging home — keeps the same
    # arm configuration so no unexpected flips.
    print("\n▶  STEP 8 — Retreat to place approach-2")
    move_joints(arm, PLACE_APPROACH2_JOINTS, "place-retreat")
    time.sleep(0.5)

    # ── STEP 9 ──────────────────────────────────────────────────────────
    print("\n▶  STEP 9 — Return home (joint space)")
    move_home(arm, "return-home")

    print("\n🎉  Pick-and-place complete!\n")
    rclpy.shutdown()
    spin_thread.join()


if __name__ == "__main__":
    main()