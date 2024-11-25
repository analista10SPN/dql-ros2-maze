cmake_minimum_required(VERSION 3.5)
project(turt_q_learn_ros2)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)  # You are using Python, so this is necessary
find_package(rclcpp REQUIRED) # For C++ if you have any nodes or logic in C++
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)  # Adding this for completeness in case it's used
find_package(nav_msgs REQUIRED)       # Adding this for completeness in case it's used
find_package(gazebo_msgs REQUIRED)    # Adding this if Gazebo-related services are used
find_package(gazebo_ros REQUIRED)     # If using Gazebo with ROS integration
find_package(std_msgs REQUIRED)       # For standard message types if needed

# Install Python nodes
ament_python_install_package(${PROJECT_NAME})

# Install executables if you have C++ nodes (not needed for Python scripts only)
# add_executable(${PROJECT_NAME}_node src/node.cpp)
# ament_target_dependencies(${PROJECT_NAME}_node rclcpp std_msgs ...)
# install(TARGETS ${PROJECT_NAME}_node
#   DESTINATION lib/${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/turt_q_learn_hypers.py  # Replace or add other Python scripts here if applicable
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install models and worlds for Gazebo
install(DIRECTORY models worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
