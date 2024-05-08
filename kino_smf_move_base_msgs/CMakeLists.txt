cmake_minimum_required(VERSION 3.0.2)
project(kino_smf_move_base_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pedsim_msgs
  message_generation
  roslib
  actionlib_msgs
  geometry_msgs
)

find_package(Boost REQUIRED COMPONENTS system)

add_message_files(
  FILES
  RelevantAgentState.msg
  RelevantAgentStates.msg
  Path2D.msg
)

add_action_files(
  FILES
  Goto2D.action
  GotoRegion2D.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  pedsim_msgs
  kino_smf_move_base_msgs
  geometry_msgs
  actionlib_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES kino_smf_move_base_msgs
  CATKIN_DEPENDS roscpp rospy std_msgs actionlib_msgs
#  DEPENDS system_lib
)



include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


add_definitions(${EIGEN_DEFINITIONS})


