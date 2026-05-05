#!/usr/bin/env python3
"""
pick_and_place_grasp.py — Pick & place for UR3e driven by grasp_detector.

Same as pick_place_test.py, EXCEPT the pick X/Y come from the
grasp_detector node instead of being hard-coded.

Topic : /shoe_detector/grasp_point_base   (geometry_msgs/PointStamped)
Frame : base_link  → no TF transform needed (already in robot frame)

Strategy
--------
  1. Subscribe to the topic at start-up.
  2. Collect N=10 consecutive valid samples and take the MEDIAN of X/Y.
   The median filters out outlier frames from the detector (a single
   bad YOLO inference won't move the target).
  3. IGNORE the published Z. Depth from the RealSense has ±5-10 mm noise
   and a wrong Z would push the gripper into the shoe. Use a fixed
   PICK_Z (safe height) as in the original script — lower in 2 cm
   steps after TCP verification.
  4. Latch the pose: once the grasp is captured, the script no longer
   listens to the topic. The robot will not chase a moving target.

Sequence
--------
  0. Wait for grasp pose from /shoe_detector/grasp_point_base   <- NEW
  1. Home             (joint space)
  2. Approach pick    (joint space — Cartesian-goal)
  3. Descend to pick  (Cartesian straight line) -> GRIPPER CLOSE
  4. Retreat          (Cartesian straight line)
  5. Place approach-1 (joint space — safe intermediate posture)
  6. Place approach-2 (joint space — above drop location)
  7. Descend to place (joint space — final drop posture) -> GRIPPER OPEN
  8. Retreat to approach-2 (joint space)
  9. Return home      (joint space)
"""

import math
import statistics
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
import time
import threading
from geometry_msgs.msg import Pose, PointStamped
from sensor_msgs.msg import JointState
from pymoveit2 import MoveIt2
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header


# ===================================================================
#  CONFIGURATION — edit these before each run
# ===================================================================

GROUP_NAME = "ur_manipulator"

# -- Grasp source (NEW) ------------------------------------------------
GRASP_TOPIC          = "/shoe_detector/grasp_point_base"
GRASP_FRAME_EXPECTED = "base_link"
GRASP_SAMPLES        = 5     # collect this many before computing median
GRASP_TIMEOUT_SEC    = 60.0   # total time to gather GRASP_SAMPLES

# -- Pick height (kept fixed — depth from camera is too noisy) ----------
# X and Y come from the topic. Z is set here for safety.
PICK_Z = 0.10              # safe start — lower in 2 cm steps after TCP echo
APPROACH_CLEARANCE = 0.20  # metres above PICK_Z

# -- Workspace bounds — reject grasp poses outside this zone -----------
# UR3e reach is ~0.5 m. Loose sanity bounds in base_link frame.
PICK_X_MIN, PICK_X_MAX = -0.45, 0.45
PICK_Y_MIN, PICK_Y_MAX = -0.45, 0.45

# Speed: 3 % of rated for safe initial tuning
VELOCITY_SCALE     = 0.03
ACCELERATION_SCALE = 0.03

PLANNING_TIME_SEC  = 15.0
POSITION_TOL       = 0.01
ORIENTATION_TOL    = 0.1

# Top face at -0.01 m, slab 50 cm thick to act as no-go floor
TABLE_Z = -0.01

# Camera mount dimensions
CAMERA_MOUNT_SIZE   = [0.065, 0.115, 0.005]
CAMERA_MOUNT_OFFSET = [0.0,   0.0,   -0.0025]

# -- Place joint configurations (radians) ------------------------------
PLACE_APPROACH1_JOINTS = [0.0, -1.57, -1.57, -1.57, 1.57, 0.0]

PLACE_APPROACH2_JOINTS = [
   math.radians(  0),
   math.radians(-140),
   math.radians( -50),
   math.radians( -75),
   math.radians(  90),
   math.radians(   0),
]

PLACE_DROP_JOINTS = [
   math.radians(  0),
   math.radians(-155),
   math.radians( -54),
   math.radians( -58),
   math.radians(  90),
   math.radians(   0),
]


# ===================================================================
#  CONSTANTS
# ===================================================================

JOINT_NAMES = [
   "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
BASE_LINK    = "base_link"
END_EFFECTOR = "tool0"
QUAT_DOWN    = dict(x=1.0, y=0.0, z=0.0, w=0.0)

HOME_JOINTS = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

JOINT_STATE_TIMEOUT = 5.0


# ===================================================================
#  GRIPPER PLACEHOLDERS — replace with calls into your GripperController
# ===================================================================

def gripper_close() -> None:
   print(" [gripper] CLOSE  <- replace with real gripper call")
   time.sleep(0.5)

def gripper_open() -> None:
   print("  [gripper] OPEN   <- replace with real gripper call")
   time.sleep(0.5)


# ===================================================================
#  GRASP LISTENER  (NEW)
#  Collects GRASP_SAMPLES poses and returns the median X/Y in base_link.
# ===================================================================

def wait_for_grasp(node: Node):
   """
   Subscribe to GRASP_TOPIC, collect GRASP_SAMPLES poses (frame_id must
   be base_link), return the median (x, y, z) tuple. The Z is returned
   only for logging — the motion code uses the fixed PICK_Z.
   Returns None on timeout or wrong frame.
   """
   samples_x: list[float] = []
   samples_y: list[float] = []
   samples_z: list[float] = []
   bad_frames: list[str] = []
   done = threading.Event()

   def _cb(msg: PointStamped) -> None:
   if msg.header.frame_id != GRASP_FRAME_EXPECTED:
   bad_frames.append(msg.header.frame_id)
   return
   samples_x.append(msg.point.x)
   samples_y.append(msg.point.y)
   samples_z.append(msg.point.z)
   if len(samples_x) >= GRASP_SAMPLES:
   done.set()

   sub = node.create_subscription(PointStamped, GRASP_TOPIC, _cb, 10)
   print(f"   Listening on {GRASP_TOPIC} for {GRASP_SAMPLES} samples "
   f"(timeout {GRASP_TIMEOUT_SEC:.0f} s)...")
   done.wait(timeout=GRASP_TIMEOUT_SEC)
   node.destroy_subscription(sub)

   n = len(samples_x)
   if n < 3:
   if bad_frames:
   print(f"   Received messages but frame_id was '{bad_frames[-1]}', "
   f"expected '{GRASP_FRAME_EXPECTED}'")
   else:
   print(f"   Only got {n} sample(s) before timeout — is the "
   f"detector running and seeing the shoe?")
   return None

   if n < GRASP_SAMPLES:
   print(f"   WARNING: Got only {n}/{GRASP_SAMPLES} samples — proceeding anyway")

   mx = statistics.median(samples_x)
   my = statistics.median(samples_y)
   mz = statistics.median(samples_z)

   # Print spread so the operator can see how stable the detection was
   sx = max(samples_x) - min(samples_x)
   sy = max(samples_y) - min(samples_y)
   print(f"   Samples  : n={n}")
   print(f"   Median   : X={mx:+.4f}  Y={my:+.4f}  Z={mz:+.4f}")
   print(f"   Spread   : dX={sx*1000:.1f} mm  dY={sy*1000:.1f} mm")
   if sx > 0.02 or sy > 0.02:
   print(f"   WARNING: Spread > 2 cm — detection is noisy; check lighting / shoe pose")

   return mx, my, mz


def validate_grasp(x: float, y: float, z: float) -> bool:
   """Reject grasp poses outside the workspace bounds."""
   ok = True
   if not (PICK_X_MIN <= x <= PICK_X_MAX):
   print(f"   ERROR: Grasp X={x:+.3f} outside [{PICK_X_MIN}, {PICK_X_MAX}]")
   ok = False
   if not (PICK_Y_MIN <= y <= PICK_Y_MAX):
   print(f"   ERROR: Grasp Y={y:+.3f} outside [{PICK_Y_MIN}, {PICK_Y_MAX}]")
   ok = False
   if z < TABLE_Z - 0.05 or z > 0.30:
   print(f"   WARNING: Grasp Z={z:+.3f} unusual (table at {TABLE_Z:.2f}) — "
   f"continuing anyway since Z is overridden by PICK_Z={PICK_Z:.2f}")
   return ok


# ===================================================================
#  JOINT-STATE HELPERS  (FIX 3)
# ===================================================================

def _wrap(angle: float) -> float:
   return (angle + math.pi) % (2 * math.pi) - math.pi


def get_current_joints(node: Node):
   received: list[float] | None = None
   event = threading.Event()

   def _cb(msg: JointState) -> None:
   nonlocal received
   name_to_pos = dict(zip(msg.name, msg.position))
   try:
   received = [_wrap(name_to_pos[n]) for n in JOINT_NAMES]
   except KeyError:
   return
   event.set()

   sub = node.create_subscription(JointState, "/joint_states", _cb, 10)
   event.wait(timeout=JOINT_STATE_TIMEOUT)
   node.destroy_subscription(sub)

   if received is None:
   node.get_logger().warn("get_current_joints: timed out waiting for /joint_states")
   return received


def seed_from_current(node: Node, arm: MoveIt2) -> None:
   joints = get_current_joints(node)
   if joints is None:
   return
   try:
   arm.set_start_state_to_current_state()
   except AttributeError:
   pass
   names_vals = ", ".join(f"{n}={v:.3f}" for n, v in zip(JOINT_NAMES, joints))
   node.get_logger().info(f"Seed joints (wrapped): {names_vals}")


# ===================================================================
#  MOTION HELPERS
# ===================================================================

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
   try:
   arm.cancel_execution()
   except Exception:
   pass
   time.sleep(0.3)


def setup_collision_scene(arm: MoveIt2) -> None:
   for name in ["table", "ceiling", "floor"]:
   arm.remove_collision_object(name)
   time.sleep(0.4)

   arm.add_collision_box(
   id="table",
   size=[2.0, 2.0, 0.50],
   frame_id=BASE_LINK,
   position=[0.0, 0.0, TABLE_Z - 0.25],
   quat_xyzw=[0.0, 0.0, 0.0, 1.0],
   )
   print(f"   Table top face at Z={TABLE_Z:.3f} m, 50 cm thick (no-go floor)")
   time.sleep(1.5)


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
   aco.link_name = END_EFFECTOR
   aco.object    = co
   aco.touch_links = [
   "tool0",
   "wrist_1_link",
   "wrist_2_link",
   "wrist_3_link",
   ]

   pub.publish(aco)
   print(f"   Camera mount attached  {CAMERA_MOUNT_SIZE}")
   time.sleep(1.0)


def move_safe(node: Node, arm: MoveIt2, pose: Pose, label: str, retries: int = 3) -> bool:
   for attempt in range(1, retries + 1):
   print(f"   [{label}] attempt {attempt}/{retries} ...")
   cancel_and_wait(arm)
   seed_from_current(node, arm)
   try:
   arm.move_to_pose(pose)
   result = arm.wait_until_executed()
   if result is False:
   print(f"   [{label}] execution reported failure")
   time.sleep(2.0)
   continue
   print(f"   [{label}] reached")
   return True
   except Exception as exc:
   print(f"   [{label}] {str(exc)[:80]}")
   time.sleep(2.0)
   print(f"   [{label}] all retries exhausted")
   return False


def move_cartesian(node: Node, arm: MoveIt2, pose: Pose, label: str, retries: int = 3) -> bool:
   for attempt in range(1, retries + 1):
   print(f"   [{label}] attempt {attempt}/{retries} (Cartesian) ...")
   cancel_and_wait(arm)
   seed_from_current(node, arm)
   try:
   try:
   arm.move_to_pose(pose, cartesian=True)
   except TypeError:
   print(f"   [{label}] cartesian= kwarg not supported; using standard planner")
   arm.move_to_pose(pose)
   result = arm.wait_until_executed()
   if result is False:
   print(f"   [{label}] execution reported failure")
   time.sleep(2.0)
   continue
   print(f"   [{label}] reached")
   return True
   except Exception as exc:
   print(f"   [{label}] {str(exc)[:80]}")
   time.sleep(2.0)
   print(f"   [{label}] all retries exhausted")
   return False


def move_joints(node: Node, arm: MoveIt2, joints: list, label: str, retries: int = 3) -> bool:
   deg_str = "  ".join(f"{math.degrees(j):+.1f} deg" for j in joints)
   for attempt in range(1, retries + 1):
   print(f"   [{label}] attempt {attempt}/{retries}  [{deg_str}]")
   cancel_and_wait(arm)
   seed_from_current(node, arm)
   try:
   arm.move_to_configuration(joints)
   result = arm.wait_until_executed()
   if result is False:
   print(f"   [{label}] execution reported failure")
   time.sleep(2.0)
   continue
   print(f"   [{label}] reached")
   return True
   except Exception as exc:
   print(f"   [{label}] {str(exc)[:80]}")
   time.sleep(2.0)
   print(f"   [{label}] all retries exhausted")
   return False


def move_home(node: Node, arm: MoveIt2, label: str = "home") -> bool:
   print(f"   [{label}] moving to home ...")
   cancel_and_wait(arm)
   seed_from_current(node, arm)
   arm.move_to_configuration(HOME_JOINTS)
   result = arm.wait_until_executed()
   if result is False:
   print(f"   [{label}] home move failed")
   return False
   print(f"   [{label}] home reached")
   return True


def abort(node: Node, arm: MoveIt2, spin_thread: threading.Thread, reason: str) -> None:
   print(f"\nABORT — {reason}")
   move_home(node, arm, "abort-home")
   rclpy.shutdown()
   spin_thread.join()


def shutdown_clean(spin_thread: threading.Thread) -> None:
   """Clean shutdown when we abort BEFORE any motion (no need to go home)."""
   rclpy.shutdown()
   spin_thread.join()


# ===================================================================
#  MAIN
# ===================================================================

def main():
   rclpy.init()

   node = Node(
   "ur3e_pick_place_grasp",
   parameter_overrides=[
   Parameter("use_sim_time", Parameter.Type.BOOL, False),
   ],
   )

   print("Initialising MoveIt2...")
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

   try:
   node.set_parameters([
   Parameter(
   "move_group.trajectory_execution.allowed_start_tolerance",
   Parameter.Type.DOUBLE,
   0.05,
   )
   ])
   except Exception:
   pass

   executor = MultiThreadedExecutor()
   executor.add_node(node)
   spin_thread = threading.Thread(target=executor.spin, daemon=True)
   spin_thread.start()

   print("Waiting for ROS2 / MoveIt2 to settle... (5 s)")
   time.sleep(5.0)

   print("\nReading current joint states...")
   current = get_current_joints(node)
   if current:
   for name, val in zip(JOINT_NAMES, current):
   print(f"    {name:30s}: {val:+.4f} rad  ({math.degrees(val):+.1f} deg)")
   wrist3 = current[5]
   if abs(abs(wrist3) - math.pi) < 0.1:
   print("\n   WARNING: wrist_3 is near +/-pi — wrap ambiguity zone!")
   print("       Jog wrist_3 away from +/-180 deg on the teach pendant then restart.")
   else:
   print("   WARNING: Could not read joint states — proceeding anyway")

   print("\nSetting up collision scene...")
   setup_collision_scene(arm)
   attach_camera_mount(node)
   time.sleep(1.0)

   # -- STEP 0 — Wait for grasp pose from grasp_detector (NEW) ----------
   print(f"\n>  STEP 0 — Waiting for grasp pose from {GRASP_TOPIC}")
   grasp = wait_for_grasp(node)
   if grasp is None:
   print("\n   Make sure grasp_detector is running:")
   print("     ros2 run shoe_detector grasp_detector --ros-args \\")
   print("       -p model_path:=/path/to/best_kaagle_10mars.pt \\")
   print("       -p target_class:=hole -p conf_threshold:=0.5")
   shutdown_clean(spin_thread)
   return

   PICK_X, PICK_Y, grasp_z_published = grasp

   if not validate_grasp(PICK_X, PICK_Y, grasp_z_published):
   print("\n   Grasp rejected. Reposition the shoe and try again.")
   shutdown_clean(spin_thread)
   return

   # Build pick poses from the latched grasp X/Y + the SAFE fixed Z.
   pose_approach = make_pose(PICK_X, PICK_Y, PICK_Z + APPROACH_CLEARANCE)
   pose_pick     = make_pose(PICK_X, PICK_Y, PICK_Z)

   print(f"\nPick point (base_link frame):")
   print(f"    grasp Z (camera, ignored)  : {grasp_z_published:+.3f}")
   print(f"    PICK_Z (used, fixed safe)  : {PICK_Z:+.3f}")
   print(f"    approach  X:{PICK_X:+.3f}  Y:{PICK_Y:+.3f}  Z:{PICK_Z + APPROACH_CLEARANCE:.3f}")
   print(f"    pick      X:{PICK_X:+.3f}  Y:{PICK_Y:+.3f}  Z:{PICK_Z:.3f}  <- verify before lowering")
   print(f"    table     Z:{TABLE_Z:.3f}")
   print(f"\nPlace joint configs (degrees):")
   print(f"    approach-1  {[round(math.degrees(j), 1) for j in PLACE_APPROACH1_JOINTS]}")
   print(f"    approach-2  {[round(math.degrees(j), 1) for j in PLACE_APPROACH2_JOINTS]}")
   print(f"    drop        {[round(math.degrees(j), 1) for j in PLACE_DROP_JOINTS]}")
   print(f"\nWARNING: PICK_Z = {PICK_Z:.2f} m — verify TCP then lower in 2 cm steps")
   print(f"WARNING: Speed: {VELOCITY_SCALE*100:.0f} % velocity / {ACCELERATION_SCALE*100:.0f} % acceleration\n")

   # -- STEP 1 — Home ---------------------------------------------------
   print("> STEP 1 — Home (joint space)")
   move_home(node, arm, "home")
   time.sleep(0.5)

   # -- STEP 2 — Approach pick ------------------------------------------
   print(f"\n> STEP 2 — Approach pick ({APPROACH_CLEARANCE*100:.0f} cm above pick)")
   if not move_safe(node, arm, pose_approach, "approach"):
   abort(node, arm, spin_thread, "approach failed")
   return

   print(f"\n   Verify TCP:  ros2 topic echo /tcp_pose_broadcaster/pose --once")
   print(f"   Expected Z ~ {PICK_Z + APPROACH_CLEARANCE:.3f}")
   time.sleep(0.5)

   # -- STEP 3 — Descend to pick ----------------------------------------
   print("\n> STEP 3 — Descend to pick (Cartesian)")
   if not move_cartesian(node, arm, pose_pick, "pick-descend"):
   move_cartesian(node, arm, pose_approach, "pick-retreat-fail")
   abort(node, arm, spin_thread, "pick descend failed")
   return

   gripper_close()

   # -- STEP 4 — Retreat from pick --------------------------------------
   print("\n> STEP 4 — Retreat from pick (Cartesian)")
   if not move_cartesian(node, arm, pose_approach, "pick-retreat"):
   abort(node, arm, spin_thread, "pick retreat failed — object may be held")
   return
   time.sleep(0.3)

   # -- STEP 5 — Place approach-1 (safe intermediate) -------------------
   print("\n> STEP 5 — Place approach-1 (safe intermediate posture)")
   if not move_joints(node, arm, PLACE_APPROACH1_JOINTS, "place-approach-1"):
   abort(node, arm, spin_thread, "place approach-1 failed — object still held")
   return
   time.sleep(0.3)

   # -- STEP 6 — Place approach-2 (above drop) --------------------------
   print("\n> STEP 6 — Place approach-2 (above drop location)")
   if not move_joints(node, arm, PLACE_APPROACH2_JOINTS, "place-approach-2"):
   abort(node, arm, spin_thread, "place approach-2 failed — object still held")
   return
   time.sleep(0.3)

   # -- STEP 7 — Descend to drop ----------------------------------------
   print("\n> STEP 7 — Descend to drop posture")
   if not move_joints(node, arm, PLACE_DROP_JOINTS, "place-drop"):
   move_joints(node, arm, PLACE_APPROACH2_JOINTS, "place-retreat-fail")
   abort(node, arm, spin_thread, "place drop failed — object still held")
   return

   gripper_open()

   # -- STEP 8 — Retreat from place -------------------------------------
   print("\n> STEP 8 — Retreat to place approach-2")
   move_joints(node, arm, PLACE_APPROACH2_JOINTS, "place-retreat")
   time.sleep(0.3)

   # -- STEP 9 — Return home --------------------------------------------
   print("\n> STEP 9 — Return home (joint space)")
   move_home(node, arm, "return-home")

   print("\nPick & place complete!\n")
   rclpy.shutdown()
   spin_thread.join()


if __name__ == "__main__":
   main()
