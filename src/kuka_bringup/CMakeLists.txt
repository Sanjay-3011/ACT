cmake_minimum_required(VERSION 3.8)
project(kuka_bringup)

find_package(ament_cmake REQUIRED)

# Install directories containing launch files and scripts
install(DIRECTORY
  launch
  scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
