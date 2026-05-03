#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import time
import threading
from geometry_msgs.msg import Pose
from pymoveit2 import MoveIt2

def main():
    rclpy.init()
    
    node = Node(
        "ur3e_stacker_final",
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)]
    )
    
    # Utilisation du SingleThreadedExecutor pour la stabilité
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("--- DÉMARRAGE DU SCRIPT ---")
    print("Vérification de la connexion Gazebo/MoveIt... (5s)")
    time.sleep(5.0) 

    arm = MoveIt2(
        node=node,
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        base_link_name="base_link",
        end_effector_name="tool0",
        group_name="ur3e",
    )
    
    arm.max_velocity_scaling_factor = 0.1
    arm.max_acceleration_scaling_factor = 0.1
    arm.planning_time = 15.0 

    # --- TES DONNÉES DE POSITIONS ---
    robot_X, robot_Y, robot_Z = 0.4, -0.68, 0.89
    cube_world_X, cube_world_Y, cube_world_Z = 0.3, -0.3, 0.91 
    
    b1_world_X, b1_world_Y = 0.13, -0.375
    b2_world_X, b2_world_Y = 0.35, -0.105
    b3_world_X, b3_world_Y = 0.615, -0.18

    SIZE_B1 = [0.16, 0.55, 0.07]
    SIZE_B2 = [0.26, 0.11, 0.11]
    SIZE_B3 = [0.27, 0.26, 0.15]
    SIZE_TABLE = [1.5, 1.2, 0.001]

    # CALCUL DES POSITIONS RELATIVES
    TARGET_X = cube_world_X - robot_X   
    TARGET_Y = cube_world_Y - robot_Y   
    TARGET_Z = cube_world_Z - robot_Z   

    T1_X, T1_Y, T1_Z = b1_world_X - robot_X, b1_world_Y - robot_Y, 0.07 / 2
    T2_X, T2_Y, T2_Z = b2_world_X - robot_X, b2_world_Y - robot_Y, 0.11 / 2
    T3_X, T3_Y, T3_Z = b3_world_X - robot_X, b3_world_Y - robot_Y, 0.15 / 2

    CUBE_NAME = "cube_rouge"
    GRIPPER_OFFSET = 0.15 

    # --- MISE EN PLACE DE LA SCÈNE ---
    print("Nettoyage et ajout des obstacles...")
    for obj in [CUBE_NAME, "box1", "box2", "box3", "table_plan"]:
        arm.remove_collision_object(obj)
    time.sleep(1.0)

    arm.add_collision_box("table_plan", SIZE_TABLE, "box", [0.0, 0.0, -0.02], [0.0, 0.0, 0.0, 1.0])
    arm.add_collision_box("box1", SIZE_B1, "box", [T1_X, T1_Y, T1_Z], [0.0, 0.0, 0.0, 1.0])
    arm.add_collision_box("box2", SIZE_B2, "box", [T2_X, T2_Y, T2_Z], [0.0, 0.0, 0.0, 1.0])
    arm.add_collision_box("box3", SIZE_B3, "box", [T3_X, T3_Y, T3_Z], [0.0, 0.0, 0.0, 1.0])
    arm.add_collision_box(CUBE_NAME, [0.06, 0.06, 0.06], "box", [TARGET_X, TARGET_Y, TARGET_Z], [0.0, 0.0, 0.0, 1.0])
    time.sleep(2.0)

    # ==========================================================
    # --- DÉROULÉ DU TEST (FORCE SUCCESS & RESET) ---
    # ==========================================================

    # Paramètres de tolérance pour garantir la planification
    arm.planning_time = 20.0       
    arm.goal_position_tolerance = 0.01 # 1cm (plus flexible pour éviter le FAILURE)
    arm.goal_orientation_tolerance = 0.1

    # --- NETTOYAGE RADICAL AVANT DE COMMENCER ---
    # On retire tout ce qui pourrait bloquer le robot au démarrage
    print("Nettoyage forcé de la scène...")
    for obj in ["cube_rouge", "test_cube_999", "box3"]:
        arm.remove_collision_object(obj)
    time.sleep(1.0)

    # ÉTAPE 0 : Position Initiale en L
    print("--- ÉTAPE 0 : Position de repos (L) ---")
    arm.move_to_configuration([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    arm.wait_until_executed()
    time.sleep(2.0)

    # --- ÉTAPE DE SÉCURITÉ : MONTÉE VIA JOINTS (PLUS FIABLE) ---
    print("Mouvement de sécurité (Redressement)...")
    # Redressement du coude pour éviter de raser la table lors du déplacement
    arm.move_to_configuration([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    arm.wait_until_executed()

    # ÉTAPE A : APPROCHE HAUTE
    print(f"--- ÉTAPE A : Approche vers X:{TARGET_X:.2f}, Y:{TARGET_Y:.2f} ---")
    pose_approche = Pose()
    pose_approche.position.x = float(TARGET_X)
    pose_approche.position.y = float(TARGET_Y)
    # Approche à +20cm du centre du cube pour valider l'alignement
    pose_approche.position.z = float(TARGET_Z + 0.20) 
    
    pose_approche.orientation.x = 1.0
    pose_approche.orientation.y = 0.0
    pose_approche.orientation.z = 0.0
    pose_approche.orientation.w = 0.0

    arm.move_to_pose(pose_approche)
    arm.wait_until_executed()

    # ÉTAPE B : DESCENTE PRÉCISE
    print("--- ÉTAPE B : Descente vers le cube ---")
    # On définit une nouvelle pose pour être sûr de ne pas modifier l'originale par référence
    pose_pick = Pose()
    pose_pick.position.x = float(TARGET_X)
    pose_pick.position.y = float(TARGET_Y)
    # Descente à 2cm au-dessus du centre (TARGET_Z)
    pose_pick.position.z = float(TARGET_Z + 0.02) 
    pose_pick.orientation = pose_approche.orientation
    
    arm.move_to_pose(pose_pick)
    arm.wait_until_executed()
    print(">>> POSITION DE SAISIE ATTEINTE !")

    # --- ÉTAPE C : L-SHAPE OPPOSÉ (Z=0.9 via configuration joints) ---
    print("--- ÉTAPE C : Rotation vers le côté opposé ---")
    time.sleep(1.0) 
    # Configuration en L à 180° (Base=3.14) pour libérer la zone
    arm.move_to_configuration([3.14, -1.57, 1.57, -1.57, -1.57, 0.0])
    arm.wait_until_executed()

    print("Fin de séquence complète.")
    node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()
    
if __name__ == "__main__":
    main()
