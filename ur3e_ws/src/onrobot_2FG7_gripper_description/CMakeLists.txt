cmake_minimum_required(VERSION 3.8)
project(onrobot_2fg7_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Trouver les outils de build ROS 2
find_package(ament_cmake REQUIRED)

# Installer les dossiers nécessaires (pour que Gazebo/Rviz les trouvent)
install(DIRECTORY
  config
  launch
  meshes
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
