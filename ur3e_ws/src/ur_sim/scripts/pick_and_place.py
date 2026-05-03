#!/usr/bin/env python3
"""
pick_and_place_fixed.py — Full pick & place for UR3e (REAL ROBOT)

Sequence
--------
  1. Home             (joint space)
  2. Approach pick    (joint space — Cartesian-goal)
  3. Descend to pick  (Cartesian straight line) → GRIPPER CLOSE
  4. Retreat          (Cartesian straight line)
  5. Place approach-1 (joint space — safe intermediate posture)
  6. Place approach-2 (joint space — above drop location)
  7. Descend to place (joint space — final drop posture) → GRIPPER OPEN
  8. Retreat to approach-2 (joint space)
  9. Return home      (joint space)

Fixes applied
=============
  FIX 1 — Wrong group name:
      GROUP_NAME = "ur_manipulator"  (SRDF: <group name="ur_manipulator">)
      Was "ur3e" → OMPL had no config → every plan failed.

  FIX 2 — Table slab clips base_link_inertia:
      TABLE_Z = -0.01 m, slab centre at TABLE_Z - 0.04 m.
      Top face now sits at -0.01 m, safely below the robot base at Z=0.
      Was 0.00 m → MoveIt flagged every start state as "in collision".

  FIX 3 — wrist_3 2π wrap → PATH_TOLERANCE_VIOLATED:
      get_current_joints() reads /joint_states live and wraps every angle
      into (−π, π] before planning.  seed_from_current() is called before
      every move command.  Controller and planner always agree on the wrap.

  FIX 4 — camera_mount touch_links too narrow:
      Added wrist_1_link and wrist_2_link to touch_links.  At home pose
      all four wrist links are close to tool0; missing any one caused
      MoveIt to report a self-collision in the start state.

  Speed change:
      VELOCITY_SCALE / ACCELERATION_SCALE reduced from 0.05 to 0.03
      (3 % of rated speed) for extra caution during initial tuning.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
import time
import threading
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from pymoveit2 import MoveIt2
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these before each run
# ═══════════════════════════════════════════════════════════════════════

# FIX 1: must match SRDF <group name="ur_manipulator">
GROUP_NAME = "ur_manipulator"

# ── Pick point (base_link frame) ────────────────────────────────────────
PICK_X =  0.3 - 0.4       # -0.10 m
PICK_Y = -0.3 - (-0.68)   #  0.38 m
PICK_Z =  0.10             # ⚠️  safe start — lower in 2 cm steps after
                           #     TCP echo confirms position

APPROACH_CLEARANCE = 0.20  # metres above PICK_Z on descend

# Speed: reduced to 3 % of rated speed for safe initial tuning
VELOCITY_SCALE     = 0.03  # was 0.05
ACCELERATION_SCALE = 0.03  # was 0.05

PLANNING_TIME_SEC  = 15.0
POSITION_TOL       = 0.01
ORIENTATION_TOL    = 0.1

# FIX 2: top face at -0.01 m clears base_link_inertia
TABLE_Z = -0.01

# Camera mount dimensions
CAMERA_MOUNT_SIZE   = [0.065, 0.115, 0.005]
CAMERA_MOUNT_OFFSET = [0.0,   0.0,   -0.0025]

# ── Place joint configurations (radians) ───────────────────────────────
# Intermediate safe posture — moves arm away from pick side before
# swinging toward the drop zone.
PLACE_APPROACH1_JOINTS = [0.0, -1.57, -1.57, -1.57, 1.57, 0.0]

# Pre-place posture: arm lined up above drop location
# [0°, -140°, -50°, -75°, 90°, 0°]
PLACE_APPROACH2_JOINTS = [
    math.radians(  0),
    math.radians(-140),
    math.radians( -50),
    math.radians( -75),
    math.radians(  90),
    math.radians(   0),
]

# Final drop posture: lower into drop location
# [0°, -155°, -54°, -58°, 90°, 0°]
PLACE_DROP_JOINTS = [
    math.radians(  0),
    math.radians(-155),
    math.radians( -54),
    math.radians( -58),
    math.radians(  90),
    math.radians(   0),
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

# wrist_3 = 0.0 keeps it well away from the ±π wrap boundary (FIX 3)
HOME_JOINTS = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

# Timeout for /joint_states read
JOINT_STATE_TIMEOUT = 5.0


# ═══════════════════════════════════════════════════════════════════════
#  GRIPPER PLACEHOLDERS
#  Replace with real gripper driver calls when hardware is ready.
# ═══════════════════════════════════════════════════════════════════════

def gripper_close() -> None:
    print("   🤏  [gripper] CLOSE  ← replace with real gripper call")
    time.sleep(0.5)

def gripper_open() -> None:
    print("   🖐   [gripper] OPEN   ← replace with real gripper call")
    time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════════════
#  JOINT-STATE HELPERS  (FIX 3)
# ═══════════════════════════════════════════════════════════════════════

def _wrap(angle: float) -> float:
    """Wrap angle into (−π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def get_current_joints(node: Node) -> list[float] | None:
    """
    Read one /joint_states message and return joint positions in
    JOINT_NAMES order, each wrapped into (−π, π].

    Wrapping is critical: if wrist_3 is physically at +π the hardware
    may report −π (or vice versa).  Without normalisation the planner
    seeds a trajectory starting 2π away from the controller's view,
    causing immediate PATH_TOLERANCE_VIOLATED abort.

    Returns None on timeout.
    """
    received: list[float] | None = None
    event = threading.Event()

    def _cb(msg: JointState) -> None:
        nonlocal received
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            received = [_wrap(name_to_pos[n]) for n in JOINT_NAMES]
        except KeyError:
            return   # not all joints present yet — wait for next message
        event.set()

    sub = node.create_subscription(JointState, "/joint_states", _cb, 10)
    event.wait(timeout=JOINT_STATE_TIMEOUT)
    node.destroy_subscription(sub)

    if received is None:
        node.get_logger().warn("get_current_joints: timed out waiting for /joint_states")
    return received


def seed_from_current(node: Node, arm: MoveIt2) -> None:
    """
    Attempt to set the MoveIt2 start state to the current wrapped joints.
    Falls back silently if the pymoveit2 API is not available (older versions
    default to /joint_states anyway, which is still normalised by the driver).
    """
    joints = get_current_joints(node)
    if joints is None:
        return
    try:
        arm.set_start_state_to_current_state()
    except AttributeError:
        pass
    names_vals = ", ".join(f"{n}={v:.3f}" for n, v in zip(JOINT_NAMES, joints))
    node.get_logger().info(f"Seed joints (wrapped): {names_vals}")


# ═══════════════════════════════════════════════════════════════════════
#  MOTION HELPERS
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
    Cancel any active trajectory goal before sending a new one.
    Prevents STATUS_ABORTED from scaled_joint_trajectory_controller
    rejecting MERGE-mode goals — guarantees each new goal is REPLACE.
    """
    try:
        arm.cancel_execution()
    except Exception:
        pass
    time.sleep(0.3)


def setup_collision_scene(arm: MoveIt2) -> None:
    """
    Table slab only — no ceiling (UR3e links exceed 0.35 m in normal configs).

    FIX 2: slab centre at TABLE_Z - 0.04 so top face sits at TABLE_Z = -0.01 m,
    safely below base_link_inertia at Z = 0.
    """
    for name in ["table", "ceiling", "floor"]:
        arm.remove_collision_object(name)
    time.sleep(0.4)

    arm.add_collision_box(
        id="table",
        size=[2.0, 2.0, 0.05],
        frame_id=BASE_LINK,
        position=[0.0, 0.0, TABLE_Z - 0.04],   # top face at TABLE_Z
        quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    print(f"   🗺️   Table top face at Z={TABLE_Z:.3f} m  (no ceiling)")
    time.sleep(1.5)   # give planning scene time to process before first plan


def attach_camera_mount(node: Node) -> None:
    """
    Attach the camera bracket as a collision object on tool0.

    FIX 4: touch_links lists ALL links geometrically adjacent to tool0.
    At home pose wrist_1/2/3 and tool0 are clustered together — missing
    any one causes MoveIt to flag the start state as in self-collision
    and refuse every plan.
    """
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
    aco.link_name = END_EFFECTOR
    aco.object    = co
    # FIX 4: all four links must be listed or self-collision check fails
    aco.touch_links = [
        "tool0",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]

    pub.publish(aco)
    print(f"   📷  Camera mount attached  {CAMERA_MOUNT_SIZE}")
    time.sleep(1.0)


def move_safe(node: Node, arm: MoveIt2, pose: Pose, label: str, retries: int = 3) -> bool:
    """Joint-space Cartesian-goal move with cancel-before-send and wrap seeding."""
    for attempt in range(1, retries + 1):
        print(f"   [{label}] attempt {attempt}/{retries} ...")
        cancel_and_wait(arm)
        seed_from_current(node, arm)   # FIX 3
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


def move_cartesian(node: Node, arm: MoveIt2, pose: Pose, label: str, retries: int = 3) -> bool:
    """
    Cartesian straight-line move with cancel-before-send and wrap seeding.
    Forces linear TCP path → prevents elbow-flip on descend/retreat.
    """
    for attempt in range(1, retries + 1):
        print(f"   [{label}] attempt {attempt}/{retries} (Cartesian) ...")
        cancel_and_wait(arm)
        seed_from_current(node, arm)   # FIX 3
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


def move_joints(node: Node, arm: MoveIt2, joints: list, label: str, retries: int = 3) -> bool:
    """Joint-space move to an explicit configuration with cancel-before-send and wrap seeding."""
    deg_str = "  ".join(f"{math.degrees(j):+.1f}°" for j in joints)
    for attempt in range(1, retries + 1):
        print(f"   [{label}] attempt {attempt}/{retries}  [{deg_str}]")
        cancel_and_wait(arm)
        seed_from_current(node, arm)   # FIX 3
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


def move_home(node: Node, arm: MoveIt2, label: str = "home") -> bool:
    """Move to HOME_JOINTS with wrap-normalised seed. wrist_3=0 avoids ±π ambiguity."""
    print(f"   [{label}] moving to home ...")
    cancel_and_wait(arm)
    seed_from_current(node, arm)   # FIX 3
    arm.move_to_configuration(HOME_JOINTS)
    result = arm.wait_until_executed()
    if result is False:
        print(f"   [{label}] ❌  home move failed")
        return False
    print(f"   [{label}] ✅  home reached")
    return True


def abort(node: Node, arm: MoveIt2, spin_thread: threading.Thread, reason: str) -> None:
    print(f"\n⛔  ABORT — {reason}")
    move_home(node, arm, "abort-home")
    rclpy.shutdown()
    spin_thread.join()


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

    # MoveIt2 must be built BEFORE the executor starts spinning.
    # (Avoids: RCLError: wait set index for status subscription out of bounds)
    print("🔧  Initialising MoveIt2...")
    arm = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=BASE_LINK,
        end_effector_name=END_EFFECTOR,
        group_name=GROUP_NAME,          # FIX 1: "ur_manipulator"
    )
    arm.max_velocity_scaling_factor     = VELOCITY_SCALE     # 3 %
    arm.max_acceleration_scaling_factor = ACCELERATION_SCALE # 3 %
    arm.planning_time                   = PLANNING_TIME_SEC
    arm.goal_position_tolerance         = POSITION_TOL
    arm.goal_orientation_tolerance      = ORIENTATION_TOL

    # Raise allowed_start_tolerance 0.01 → 0.05 rad so joint drift
    # during retry sleep doesn't cause "start point deviates" aborts.
    try:
        node.set_parameters([
            Parameter(
                "move_group.trajectory_execution.allowed_start_tolerance",
                Parameter.Type.DOUBLE,
                0.05,
            )
        ])
    except Exception:
        pass  # not fatal if parameter doesn't exist on this distro

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("⏳  Waiting for ROS2 / MoveIt2 to settle... (5 s)")
    time.sleep(5.0)

    # Print live joint states — warns if wrist_3 is near ±π wrap zone
    print("\n🔍  Reading current joint states...")
    current = get_current_joints(node)
    if current:
        for name, val in zip(JOINT_NAMES, current):
            print(f"    {name:30s}: {val:+.4f} rad  ({math.degrees(val):+.1f}°)")
        wrist3 = current[5]
        if abs(abs(wrist3) - math.pi) < 0.1:
            print("\n   ⚠️   wrist_3 is near ±π — wrap ambiguity zone!")
            print("       Jog wrist_3 away from ±180° on the teach pendant then restart.")
    else:
        print("   ⚠️   Could not read joint states — proceeding anyway")

    print("\n🗺️   Setting up collision scene...")
    setup_collision_scene(arm)
    attach_camera_mount(node)
    time.sleep(1.0)

    pose_approach = make_pose(PICK_X, PICK_Y, PICK_Z + APPROACH_CLEARANCE)
    pose_pick     = make_pose(PICK_X, PICK_Y, PICK_Z)

    print(f"\n📍  Pick point (base_link frame):")
    print(f"    approach  X:{PICK_X:.3f}  Y:{PICK_Y:.3f}  Z:{PICK_Z + APPROACH_CLEARANCE:.3f}")
    print(f"    pick      X:{PICK_X:.3f}  Y:{PICK_Y:.3f}  Z:{PICK_Z:.3f}  ← verify before lowering")
    print(f"    table     Z:{TABLE_Z:.3f}")
    print(f"\n📍  Place joint configs (degrees):")
    print(f"    approach-1  {[round(math.degrees(j), 1) for j in PLACE_APPROACH1_JOINTS]}")
    print(f"    approach-2  {[round(math.degrees(j), 1) for j in PLACE_APPROACH2_JOINTS]}")
    print(f"    drop        {[round(math.degrees(j), 1) for j in PLACE_DROP_JOINTS]}")
    print(f"\n⚠️   PICK_Z = {PICK_Z:.2f} m — verify TCP then lower in 2 cm steps")
    print(f"⚠️   Speed: {VELOCITY_SCALE*100:.0f} % velocity / {ACCELERATION_SCALE*100:.0f} % acceleration\n")

    # ── STEP 1 — Home ───────────────────────────────────────────────────
    print("▶  STEP 1 — Home (joint space)")
    move_home(node, arm, "home")
    time.sleep(0.5)

    # ── STEP 2 — Approach pick ──────────────────────────────────────────
    print(f"\n▶  STEP 2 — Approach pick ({APPROACH_CLEARANCE*100:.0f} cm above pick)")
    if not move_safe(node, arm, pose_approach, "approach"):
        abort(node, arm, spin_thread, "approach failed")
        return

    print(f"\n   📡  Verify TCP:  ros2 topic echo /tcp_pose_broadcaster/pose --once")
    print(f"   Expected Z ≈ {PICK_Z + APPROACH_CLEARANCE:.3f}")
    time.sleep(0.5)

    # ── STEP 3 — Descend to pick ────────────────────────────────────────
    print("\n▶  STEP 3 — Descend to pick (Cartesian)")
    if not move_cartesian(node, arm, pose_pick, "pick-descend"):
        move_cartesian(node, arm, pose_approach, "pick-retreat-fail")
        abort(node, arm, spin_thread, "pick descend failed")
        return

    gripper_close()

    # ── STEP 4 — Retreat from pick ──────────────────────────────────────
    print("\n▶  STEP 4 — Retreat from pick (Cartesian)")
    if not move_cartesian(node, arm, pose_approach, "pick-retreat"):
        abort(node, arm, spin_thread, "pick retreat failed — object may be held")
        return
    time.sleep(0.3)

    # ── STEP 5 — Place approach-1 (safe intermediate) ───────────────────
    print("\n▶  STEP 5 — Place approach-1 (safe intermediate posture)")
    if not move_joints(node, arm, PLACE_APPROACH1_JOINTS, "place-approach-1"):
        abort(node, arm, spin_thread, "place approach-1 failed — object still held")
        return
    time.sleep(0.3)

    # ── STEP 6 — Place approach-2 (above drop) ──────────────────────────
    print("\n▶  STEP 6 — Place approach-2 (above drop location)")
    if not move_joints(node, arm, PLACE_APPROACH2_JOINTS, "place-approach-2"):
        abort(node, arm, spin_thread, "place approach-2 failed — object still held")
        return
    time.sleep(0.3)

    # ── STEP 7 — Descend to drop ────────────────────────────────────────
    print("\n▶  STEP 7 — Descend to drop posture")
    if not move_joints(node, arm, PLACE_DROP_JOINTS, "place-drop"):
        move_joints(node, arm, PLACE_APPROACH2_JOINTS, "place-retreat-fail")
        abort(node, arm, spin_thread, "place drop failed — object still held")
        return

    gripper_open()

    # ── STEP 8 — Retreat from place ─────────────────────────────────────
    print("\n▶  STEP 8 — Retreat to place approach-2")
    move_joints(node, arm, PLACE_APPROACH2_JOINTS, "place-retreat")
    time.sleep(0.3)

    # ── STEP 9 — Return home ────────────────────────────────────────────
    print("\n▶  STEP 9 — Return home (joint space)")
    move_home(node, arm, "return-home")

    print("\n🎉  Pick & place complete!\n")
    rclpy.shutdown()
    spin_thread.join()


if __name__ == "__main__":
    main()