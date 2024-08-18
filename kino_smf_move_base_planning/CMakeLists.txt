cmake_minimum_required(VERSION 3.8)
project(kino_smf_move_base_planning)

# Set compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set CUDA paths
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
set(CUDACXX /usr/local/cuda-11.8/bin/nvcc)
link_directories(/usr/local/cuda-11.8/lib64)

# Ensure CUDA is found and configured
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set Torch path
set(Torch_DIR "/home/sasm/libtorch-cxx11-abi-shared-with-deps-2.4.0+cu118/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)


find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBFCL REQUIRED "fcl>=0.7.0")
# replace LIBFCL_LIBRARIES with full paths to the libraries
set(LIBFCL_LIBRARIES_FULL "")
foreach(LIBFCL_LIBRARY ${LIBFCL_LIBRARIES})
  find_library(${LIBFCL_LIBRARY}_LIB ${LIBFCL_LIBRARY} ${LIBFCL_LIBRARY_DIRS})
  list(APPEND LIBFCL_LIBRARIES_FULL ${${LIBFCL_LIBRARY}_LIB})
endforeach()
set(LIBFCL_LIBRARIES "${LIBFCL_LIBRARIES_FULL}")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(kino_smf_move_base_msgs REQUIRED)
find_package(pedsim_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_cv REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(grid_map_filters REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_sdf REQUIRED)
find_package(grid_map_visualization REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ompl REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(OCTOMAP REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(octomap_server REQUIRED)

# Include directories
include_directories(
  include
  SYSTEM
)

# Handle testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Set dependencies for ament_target_dependencies
set(dependencies
    rclcpp
    pedsim_msgs
    nav_msgs
    kino_smf_move_base_msgs
    grid_map_core
    grid_map_ros
    grid_map_cv
    grid_map_msgs
    tf2_geometry_msgs
    geometry_msgs
    OMPL
    cv_bridge
    image_transport
    octomap_msgs
    octomap_server
    OCTOMAP
    LIBFCL
)

# Add executable
add_executable(kino_smf_move_base_planner
  src/planning_framework_main.cpp
  src/planner/RRTstarMod.cpp
  src/planner/SSTMod.cpp
  src/sampler/PathLengthDirectInfSamplerMod.cpp
  src/state_validity_checker_grid_map_R2.cpp
  src/local_state_validity_checker_grid_map_SE2.cpp
  src/new_state_sampler.cpp
  src/informed_new_state_sampler.cpp
  src/state_cost_objective.cpp
  src/kinematic_diff_model.cpp
  src/goal_region_angle_tolerance.cpp
)

# Link libraries
ament_target_dependencies(kino_smf_move_base_planner ${dependencies} ${fcl_LIBRARIES})
target_link_libraries(kino_smf_move_base_planner ${OMPL_LIBRARIES} ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${LIBFCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

# Set runtime path for CUDA libraries
set_target_properties(kino_smf_move_base_planner PROPERTIES
    BUILD_RPATH_USE_ORIGIN TRUE
    BUILD_RPATH "/usr/local/cuda-11.8/lib64"
)

# Install targets
install(TARGETS
  kino_smf_move_base_planner
  DESTINATION lib/${PROJECT_NAME})

# ROS 2 package setup
ament_package()
