cmake_minimum_required(VERSION 3.8)
project(kuka_gazebo)

find_package(ament_cmake REQUIRED)

# Install directories containing configurations, launch files, and worlds
install(DIRECTORY
  config
  launch
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
