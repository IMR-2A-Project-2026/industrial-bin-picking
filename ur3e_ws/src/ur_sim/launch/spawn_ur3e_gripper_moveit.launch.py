from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory
import os
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    ld = LaunchDescription()

    # --- 1. CHEMINS ET FICHIERS ---
    ur_sim_pkg = get_package_share_directory('ur_sim')
    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')
    
    # ICI : On pointe vers ton fichier ur3e_on_table.sdf
    world_file = os.path.join(ur_sim_pkg, 'worlds', 'ur3e_on_table.sdf')
    
    joint_controllers_file = os.path.join(ur_sim_pkg, 'config', 'ur3e_gripper_controllers.yaml')

    # --- 2. CONFIGURATION MOVEIT ---
    moveit_config = (
        MoveItConfigsBuilder("ur", package_name="ur_gripper_moveit")
        .robot_description(
            file_path="config/ur.urdf.xacro",
            mappings={
                "sim_ignition": "true",
                "simulation_controllers": joint_controllers_file,
            },
        )
        .robot_description_semantic(file_path="config/ur.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, 
            publish_robot_description_semantic=True, 
            publish_planning_scene=True
        )
        .planning_pipelines(pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    # --- 3. ARGUMENTS DE POSITION (C'est ici qu'on place le robot SUR la table) ---
    x_arg = DeclareLaunchArgument('x', default_value='0.4', description='X position')
    y_arg = DeclareLaunchArgument('y', default_value='-0.68', description='Y position')
    z_arg = DeclareLaunchArgument('z', default_value='0.89', description='Z position (Hauteur Table)')

    # --- 4. LANCEMENT DE GAZEBO AVEC LE MONDE ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')),
        launch_arguments={"gz_args": f" -r -v 4 {world_file}"}.items(),
    )

    # --- 5. FAIRE APPARAÎTRE LE ROBOT (SPAWN) ---
    spawn_the_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "robot_description",
            "-name", "ur",
            "-allow_renaming", "true",
            "-x", LaunchConfiguration('x'),
            "-y", LaunchConfiguration('y'),
            "-z", LaunchConfiguration('z'),
        ],
        output='screen',
    )

    # --- 6. CONTROLLEURS ROS2 ---
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[moveit_config.robot_description, joint_controllers_file],
        output='screen',
        remappings=[("~/robot_description", "/robot_description")],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[moveit_config.robot_description],
        output='screen'
    )

    # Spawners (Joints, Trajectoire, Pince)
    js_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    traj_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
    )
    grip_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["robotiq_gripper_controller", "-c", "/controller_manager"],
    )

    # --- 7. MOVEIT (MOVE GROUP) ---
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": True}],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # --- 8. RVIZ ---
    rviz_node = Node(
        package="rviz2", executable="rviz2", output="screen",
        arguments=["-d", os.path.join(get_package_share_directory("ur_gripper_moveit"), "config", "moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": True}
        ],
    )

    gz_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"],
        output="screen",
    )

    # --- 9. GESTION DE L'ORDRE DE LANCEMENT (DÉLAIS) ---
    delay_cm = RegisterEventHandler(OnProcessExit(target_action=spawn_the_robot, on_exit=[controller_manager]))
    delay_js = RegisterEventHandler(OnProcessExit(target_action=spawn_the_robot, on_exit=[js_spawner]))
    delay_traj = RegisterEventHandler(OnProcessExit(target_action=js_spawner, on_exit=[traj_spawner]))
    delay_grip = RegisterEventHandler(OnProcessExit(target_action=traj_spawner, on_exit=[grip_spawner]))
    delay_mg = RegisterEventHandler(OnProcessExit(target_action=grip_spawner, on_exit=[move_group]))
    delay_rviz = RegisterEventHandler(OnProcessExit(target_action=spawn_the_robot, on_exit=[rviz_node]))

    # Ajout de tout à la description
    for action in [x_arg, y_arg, z_arg, gazebo, spawn_the_robot, robot_state_publisher, gz_bridge,
                   delay_cm, delay_js, delay_traj, delay_grip, delay_mg, delay_rviz]:
        ld.add_action(action)

    return ld
