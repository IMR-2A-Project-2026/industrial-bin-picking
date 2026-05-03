#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import time
import threading
import math
from pymoveit2 import MoveIt2
from tf2_ros import Buffer, TransformListener

def main():
    # 1. Initialisation de ROS 2
    rclpy.init()
    
    # On crée le nœud avec use_sim_time=True pour Gazebo
    node = Node(
        "ur3e_validation_joints",
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)]
    )

    # 2. Lancement du "Spin" en arrière-plan (indispensable pour TF2)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 3. Initialisation de TF2 (pour lire la position du robot)
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    # 4. Initialisation de MoveIt2
    arm = MoveIt2(
        node=node,
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        base_link_name="base_link",
        end_effector_name="tool0",
        group_name="ur3e",
    )

    print("--- SYSTÈME PRÊT : VALIDATION MGI/MGD ---")
    time.sleep(2.0) # On laisse le temps à TF2 de se synchroniser

    # ==========================================================
    # 5. TES ANGLES OCTAVE (À MODIFIER ICI)
    # ==========================================================
    # Remplace les 0.0 par tes valeurs calculées. 
    # Si tes calculs sont en DEGRÉS, utilise math.radians().
    
    theta1 = math.radians(0.0)    # Base
    theta2 = math.radians(-90.0)  # Shoulder
    theta3 = math.radians(45.0)   # Elbow
    theta4 = math.radians(-90.0)  # Wrist 1
    theta5 = math.radians(0.0)  # Wrist 2
    theta6 = math.radians(0.0)    # Wrist 3

    target_joints = [theta1, theta2, theta3, theta4, theta5, theta6]

    # ==========================================================
    # 6. EXÉCUTION DU MOUVEMENT
    # ==========================================================
    print(f"\n>>> Envoi de la configuration articulaire vers Gazebo...")
    arm.move_to_configuration(target_joints)
    arm.wait_until_executed()
    
    print(">>> Mouvement terminé. Calcul de la position réelle...")
    time.sleep(1.0) # Petite pause pour stabiliser la lecture

    # ==========================================================
    # 7. LECTURE DE LA POSITION (Vérification du End-Effector)
    # ==========================================================
    try:
        # On cherche la position de tool0 par rapport à base_link
        now = rclpy.time.Time()
        trans = tf_buffer.lookup_transform('base_link', 'tool0', now, timeout=rclpy.duration.Duration(seconds=1.0))
        
        pos_x = trans.transform.translation.x
        pos_y = trans.transform.translation.y
        pos_z = trans.transform.translation.z

        print("\n" + "="*40)
        print("RÉSULTATS DE LA VALIDATION")
        print("="*40)
        print(f"Position lue dans Gazebo (Base -> Tool0) :")
        print(f"  X : {pos_x:.4f} m")
        print(f"  Y : {pos_y:.4f} m")
        print(f"  Z : {pos_z:.4f} m")
        print("-" * 40)
        print("Compare ces valeurs avec tes calculs Octave !")
        print("="*40 + "\n")

    except Exception as e:
        print(f"\n[ERREUR] Impossible de lire la position TF2 : {e}")

    # 8. Fermeture propre
    print("Fermeture du script...")
    rclpy.shutdown()
    spin_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
