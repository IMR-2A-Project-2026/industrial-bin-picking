#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import time
import threading
import math
from geometry_msgs.msg import Pose
from pymoveit2 import MoveIt2
from tf2_ros import Buffer, TransformListener

# ==========================================================
# FONCTION : LE "JUGE DE PAIX" (Vérification physique avec TF2)
# ==========================================================
def wait_for_target_safe(tf_buffer, target_pose, position_tolerance=0.02, timeout=15.0):
    """
    Mesure en temps réel la position du robot dans Gazebo.
    Renvoie True dès que le robot est physiquement à la bonne position.
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # 1. Lire la position réelle du bout du bras par rapport à la base
            trans = tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            current_x = trans.transform.translation.x
            current_y = trans.transform.translation.y
            current_z = trans.transform.translation.z

            # 2. Calculer l'erreur de distance
            dx = current_x - target_pose.position.x
            dy = current_y - target_pose.position.y
            dz = current_z - target_pose.position.z
            distance = math.sqrt(dx**2 + dy**2 + dz**2)

            # 3. Si on est assez proche
            if distance <= position_tolerance:
                print(f"    [VERDICT] -> SUCCÈS : Le robot a physiquement atteint la cible ! (Erreur: {distance:.3f}m)")
                return True
                
        except Exception as e:
            # TF n'est pas encore prêt, on patiente
            pass

        time.sleep(0.1)
        
    print(f"    [VERDICT] -> ÉCHEC : Le robot n'est pas arrivé après {timeout} secondes.")
    return False


# ==========================================================
# SCRIPT PRINCIPAL
# ==========================================================
def main():
    rclpy.init()
    
    node = Node(
        "ur3e_stacker_final",
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)]
    )
    
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Initialisation du système de mesure (TF2)
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    print("--- DÉMARRAGE DU SCRIPT ---")
    time.sleep(3.0) 

    arm = MoveIt2(
        node=node,
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        base_link_name="base_link",
        end_effector_name="tool0",
        group_name="ur3e",
    )
    
    # Paramètres de fluidité et de calcul
    arm.max_velocity_scaling_factor = 0.1
    arm.max_acceleration_scaling_factor = 0.1
    arm.planning_time = 21.0  

    # --- DONNÉES DE POSITIONS ---
    robot_X, robot_Y, robot_Z = 0.4, -0.68, 0.89
    cube_world_X, cube_world_Y, cube_world_Z = 0.3, -0.3, 0.91 
    
    b1_world_X, b1_world_Y = 0.13, -0.375
    b2_world_X, b2_world_Y = 0.35, -0.105
    b3_world_X, b3_world_Y = 0.615, -0.18

    SIZE_B1 = [0.16, 0.55, 0.07]
    SIZE_B2 = [0.26, 0.11, 0.11]
    SIZE_B3 = [0.27, 0.26, 0.15]
    SIZE_TABLE = [1.5, 1.2, 0.001]

    TARGET_X = cube_world_X - robot_X   
    TARGET_Y = cube_world_Y - robot_Y   
    TARGET_Z = cube_world_Z - robot_Z   

    T1_X, T1_Y, T1_Z = b1_world_X - robot_X, b1_world_Y - robot_Y, 0.07 / 2
    T2_X, T2_Y, T2_Z = b2_world_X - robot_X, b2_world_Y - robot_Y, 0.11 / 2
    T3_X, T3_Y, T3_Z = b3_world_X - robot_X, b3_world_Y - robot_Y, 0.15 / 2

    CUBE_NAME = "cube_rouge"
    GRIPPER_OFFSET = 0.22

    # --- MISE EN PLACE DE LA SCÈNE ---
    print("Nettoyage et ajout des obstacles...")
    for obj in [CUBE_NAME, "box1", "box2", "box3", "table_plan"]:
        arm.remove_collision_object(obj)
    time.sleep(1.0)

    arm.add_collision_box("table_plan", SIZE_TABLE, "box", [0.0, 0.0, -0.02], [0.0, 0.0, 0.0, 1.0])
    arm.add_collision_box("box1", SIZE_B1, "box", [T1_X, T1_Y, T1_Z], [0.0, 0.0, 0.0, 1.0])
    arm.add_collision_box("box2", SIZE_B2, "box", [T2_X, T2_Y, T2_Z], [0.0, 0.0, 0.0, 1.0])
    arm.add_collision_box("box3", SIZE_B3, "box", [T3_X, T3_Y, T3_Z], [0.0, 0.0, 0.0, 1.0])

    #arm.add_collision_box(CUBE_NAME, [0.06, 0.06, 0.06], "box", [TARGET_X, TARGET_Y, TARGET_Z], [0.0, 0.0, 0.0, 1.0])
    time.sleep(2.0)

    # --- ÉTAPE 0 : Home ---
    print("--- ÉTAPE 0 : Mouvement vers Home ---")
    arm.move_to_configuration([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    arm.wait_until_executed()

    # Tolérances (Crucial pour éviter les erreurs de planification)
    arm.goal_position_tolerance = 0.05
    arm.goal_orientation_tolerance = 3.14

    # ==========================================
    # ÉTAPE A1 : Pré-approche (Se placer en hauteur)
    # ==========================================
    print(f"\n--- ÉTAPE A1 : Pré-approche vers X:{TARGET_X:.2f}, Y:{TARGET_Y:.2f} (En hauteur) ---")
    
    pose_target_high = Pose()
    pose_target_high.position.x = TARGET_X
    pose_target_high.position.y = TARGET_Y
    pose_target_high.position.z = TARGET_Z + 0.35  # 35 cm (Haut pour survoler)
    
    pose_target_high.orientation.x = 1.0 
    pose_target_high.orientation.y = 0.0
    pose_target_high.orientation.z = 0.0
    pose_target_high.orientation.w = 0.0

    # --- ÉTAPE A1 ---
    print(">>> Envoi de la commande A1 à MoveIt...")
    arm.move_to_pose(pose_target_high) 
    
    # ATTENTION : On attend que MoveIt confirme la fin du mouvement
    arm.wait_until_executed() 

    print(">>> Vérification de l'arrivée physique (TF2)...")
    # On réduit la tolérance TF2 pour être sûr qu'on est bien placé avant de descendre
    is_there_high = wait_for_target_safe(tf_buffer, pose_target_high, position_tolerance=0.05, timeout=10.0)

    time.sleep(1.0) # Petite pause pour stabiliser le robot en l'air

    # ==========================================
    # ÉTAPE A2 : Descente verticale
    # ==========================================
    print(f"\n--- ÉTAPE A2 : Descente verticale pour attraper le cube ---")
    
    pose_target_low = Pose()
    pose_target_low.position.x = TARGET_X
    pose_target_low.position.y = TARGET_Y
    pose_target_low.position.z = TARGET_Z + GRIPPER_OFFSET  # 18 cm (On descend !)
    
    pose_target_low.orientation.x = 1.0 
    pose_target_low.orientation.y = 0.0
    pose_target_low.orientation.z = 0.0
    pose_target_low.orientation.w = 0.0

    print(">>> Envoi de la commande A2 à MoveIt...")
    arm.move_to_pose(pose_target_low) # On ignore le retour capricieux de pymoveit2
    
    print(">>> Vérification de l'arrivée physique (TF2)...")
    is_there_low = wait_for_target_safe(tf_buffer, pose_target_low, position_tolerance=0.04, timeout=10.0)

    if not is_there_low:
        print(">>> ERREUR : Le robot n'a pas pu descendre (Collision Gazebo ?).")
    else:
        print(">>> MISSION ACCOMPLIE : Le robot est prêt à saisir le cube !")

    # ==========================================================
    # FERMETURE PROPRE
    # ==========================================================
    print("\nFermeture propre du script...")
    executor.shutdown()
    rclpy.shutdown()
    spin_thread.join(timeout=1.0)
    
if __name__ == "__main__":
    main()
