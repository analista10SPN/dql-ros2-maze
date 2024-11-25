#!/bin/bash

# Update system
sudo apt update && sudo apt install -y curl gnupg lsb-release

# Add ROS 2 apt repository
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list

# Update package index
sudo apt update

# Install ROS 2 dependencies
sudo apt install -y ros-humble-ros-base ros-humble-gazebo-ros-pkgs ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-nav-msgs

# Install pip3 if not installed
sudo apt install -y python3-pip

sudo apt install python3-colcon-common-extensions

# Install Python dependencies
pip3 install -r requirements.txt
