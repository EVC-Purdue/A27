cmake_minimum_required(VERSION 3.8)
project(kart_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GpsFix.msg"
  "msg/KartTelemetry.msg"
  "msg/DriveCommand.msg"
  "msg/ObjectMeasurement.msg"
  "msg/ObjectMeasurementArray.msg"
  "msg/TrackedObject.msg"
  "msg/TrackedObjectArray.msg"
  "msg/LinePoint.msg"
  "msg/Line.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
