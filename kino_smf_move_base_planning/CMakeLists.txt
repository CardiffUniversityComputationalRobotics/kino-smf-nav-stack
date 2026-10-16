cmake_minimum_required(VERSION 3.8)
project(kino_smf_move_base_planning)

# Set compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set CUDA paths
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDACXX /usr/local/cuda/bin/nvcc)
set(CUDA_LIBRARY_DIR /usr/local/cuda/lib64)
link_directories("${CUDA_LIBRARY_DIR}")

# Ensure CUDA is found and configured
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set Torch path (~/libtorch: the libtorch zip on x86_64, the venv's pip torch on Jetson)
set(LIBTORCH_ROOT "/home/$ENV{USER}/libtorch")
set(LIBTORCH_LIBRARY_DIR "${LIBTORCH_ROOT}/lib")
set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch")
find_package(Torch REQUIRED)
# Use the C++ standard Torch declares on its target (C++20 for torch 2.14), which does not propagate
get_target_property(TORCH_CXX_STANDARD torch CXX_STANDARD)
if(TORCH_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD ${TORCH_CXX_STANDARD})
endif()

# With pip torch (Jetson), the TensorRT and Torch-TensorRT wheels sit next to it in site-packages
get_filename_component(LIBTORCH_REAL_ROOT "${LIBTORCH_ROOT}" REALPATH)
get_filename_component(LIBTORCH_PARENT_DIR "${LIBTORCH_REAL_ROOT}" DIRECTORY)

# TENSOR RT

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(TENSORRT_LIBRARY_HINTS "${LIBTORCH_PARENT_DIR}/tensorrt_libs")
find_package(TensorRT REQUIRED)

set(TORCH_TRT_PATH "/home/$ENV{USER}/libtorchtrt-2.10.0-tensorrt10.14.1-cuda130-libtorch2.10.0-x86_64-linux/torch_tensorrt")
if(EXISTS "${LIBTORCH_PARENT_DIR}/torch_tensorrt/lib")
  set(TORCH_TRT_PATH "${LIBTORCH_PARENT_DIR}/torch_tensorrt")
endif()
set(TORCH_TRT_INCLUDE_DIR "${TORCH_TRT_PATH}/include")
set(TORCH_TRT_LIBRARY_DIR "${TORCH_TRT_PATH}/lib")
set(TORCH_TRT_LIBSTDCXX "/opt/nvidia/nsight-compute/2025.3.1/host/linux-desktop-glibc_2_11_3-x64/libstdc++.so.6")
get_filename_component(TORCH_TRT_LIBSTDCXX_REAL "${TORCH_TRT_LIBSTDCXX}" REALPATH)
set(TORCH_TRT_RUNTIME_DIR "${CMAKE_CURRENT_BINARY_DIR}/runtime_libs")
set(TORCH_TRT_INSTALL_RUNTIME_DIR "lib/${PROJECT_NAME}/runtime_libs")
file(MAKE_DIRECTORY "${TORCH_TRT_RUNTIME_DIR}")
# nsight-compute's libstdc++ only exists on the x86_64 machine
if(EXISTS "${TORCH_TRT_LIBSTDCXX}")
  file(CREATE_LINK
    "${TORCH_TRT_LIBSTDCXX_REAL}"
    "${TORCH_TRT_RUNTIME_DIR}/libstdc++.so.6"
    SYMBOLIC COPY_ON_ERROR)
else()
  set(TORCH_TRT_LIBSTDCXX "")
endif()

include_directories("${TORCH_TRT_INCLUDE_DIR}")
link_directories("${TORCH_TRT_LIBRARY_DIR}")

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
  src/iterators/AnnulusIterator.cpp
  src/iterators/ReverseCircleIterator.cpp
)

add_library(torchtrt_runtime SHARED IMPORTED)
set_target_properties(torchtrt_runtime PROPERTIES
    IMPORTED_LOCATION "${TORCH_TRT_LIBRARY_DIR}/libtorchtrt_runtime.so"
)
add_library(torchtrt_plugins SHARED IMPORTED)
set_target_properties(torchtrt_plugins PROPERTIES
    IMPORTED_LOCATION "${TORCH_TRT_LIBRARY_DIR}/libtorchtrt_plugins.so"
)
add_library(torch_global_deps SHARED IMPORTED)
set_target_properties(torch_global_deps PROPERTIES
    IMPORTED_LOCATION "${LIBTORCH_LIBRARY_DIR}/libtorch_global_deps.so"
)
# Torch-TensorRT 2.14 also needs libtorch_nvshmem.so, which its own RUNPATH cannot find
set(TORCH_NVSHMEM_LIB "")
if(EXISTS "${LIBTORCH_LIBRARY_DIR}/libtorch_nvshmem.so")
  set(TORCH_NVSHMEM_LIB "${LIBTORCH_LIBRARY_DIR}/libtorch_nvshmem.so")
endif()

# Note: no OpenMP here on purpose. libtorch ships its own libgomp and the dependency
# tree drags in the system one; two OpenMP runtimes in one process corrupt the heap.

# Link libraries. The TensorRT model is a TorchScript module loaded with torch::jit::load:
# nothing references the Torch-TensorRT runtime directly, it only registers the TRT engine
# ops when loaded, so --no-as-needed keeps the linker from dropping it.
ament_target_dependencies(kino_smf_move_base_planner ${dependencies})
target_link_libraries(kino_smf_move_base_planner
  ${OMPL_LIBRARIES}
  ${TORCH_LIBRARIES}
  torch_global_deps
  ${TORCH_NVSHMEM_LIB}
  torchtrt_runtime
  torchtrt_plugins
  ${TORCH_TRT_LIBSTDCXX}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARIES}
)

# Set runtime path for the Torch, Torch-TensorRT, CUDA and TensorRT libraries
set_target_properties(kino_smf_move_base_planner PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/runtime_libs;${LIBTORCH_LIBRARY_DIR};${TORCH_TRT_LIBRARY_DIR};${CUDA_LIBRARY_DIR};${TENSORRT_LIBRARY_DIR}"
    BUILD_RPATH "$ORIGIN/runtime_libs;${LIBTORCH_LIBRARY_DIR};${TORCH_TRT_LIBRARY_DIR};${CUDA_LIBRARY_DIR};${TENSORRT_LIBRARY_DIR}"
    SKIP_BUILD_RPATH FALSE
    LINK_FLAGS "-Wl,--no-as-needed"
)

# Install targets
install(TARGETS
  kino_smf_move_base_planner
  DESTINATION lib/${PROJECT_NAME})
if(TORCH_TRT_LIBSTDCXX)
  install(FILES
    "${TORCH_TRT_LIBSTDCXX_REAL}"
    DESTINATION "${TORCH_TRT_INSTALL_RUNTIME_DIR}"
    RENAME libstdc++.so.6)
endif()

# ROS 2 package setup
ament_package()
