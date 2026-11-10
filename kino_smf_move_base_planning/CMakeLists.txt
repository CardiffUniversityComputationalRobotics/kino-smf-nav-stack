cmake_minimum_required(VERSION 3.8)
project(kino_smf_move_base_planning)

# Set compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ==============================================================================
# Platform detection
# ==============================================================================
# The package builds on x86_64 (amd64) workstations and on aarch64 (arm64) NVIDIA Jetsons.
# Only the locations of CUDA, libtorch, Torch-TensorRT and TensorRT differ, so each one is
# searched where it is usually installed on the detected platform. Any of them can be
# forced, e.g. colcon build --cmake-args -DLIBTORCH_ROOT=/path/to/libtorch
# (see "Installation" in the README).
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" KINO_SMF_ARCH)
if(KINO_SMF_ARCH MATCHES "^(x86_64|amd64)$")
  set(KINO_SMF_ARCH "x86_64")
elseif(KINO_SMF_ARCH MATCHES "^(aarch64|arm64)$")
  set(KINO_SMF_ARCH "aarch64")
else()
  message(WARNING "kino_smf: untested processor '${CMAKE_SYSTEM_PROCESSOR}', only x86_64 and aarch64 are supported")
endif()

set(KINO_SMF_JETSON FALSE)
if(KINO_SMF_ARCH STREQUAL "aarch64" AND NOT CMAKE_CROSSCOMPILING)
  if(EXISTS "/etc/nv_tegra_release")
    set(KINO_SMF_JETSON TRUE)
  elseif(EXISTS "/proc/device-tree/model")
    file(READ "/proc/device-tree/model" _kino_smf_board)
    if(_kino_smf_board MATCHES "Jetson")
      set(KINO_SMF_JETSON TRUE)
    endif()
  endif()
endif()

set(KINO_SMF_HOME "$ENV{HOME}")
if(NOT KINO_SMF_HOME)
  set(KINO_SMF_HOME "/home/$ENV{USER}")
endif()

# ==============================================================================
# CUDA toolkit
# ==============================================================================
# -DCUDA_TOOLKIT_ROOT_DIR, else $CUDA_HOME / $CUDA_PATH, /usr/local/cuda (created by JetPack
# and by the NVIDIA packages), the newest /usr/local/cuda-X.Y or /usr (Ubuntu's nvidia-cuda-toolkit)
if(NOT CUDA_TOOLKIT_ROOT_DIR)
  file(GLOB _kino_smf_cuda_dirs LIST_DIRECTORIES true "/usr/local/cuda-*")
  list(SORT _kino_smf_cuda_dirs COMPARE NATURAL ORDER DESCENDING)
  foreach(_dir "$ENV{CUDA_HOME}" "$ENV{CUDA_PATH}" "/usr/local/cuda" ${_kino_smf_cuda_dirs} "/usr")
    if(_dir AND EXISTS "${_dir}/bin/nvcc")
      set(CUDA_TOOLKIT_ROOT_DIR "${_dir}")
      break()
    endif()
  endforeach()
endif()
if(NOT CUDA_TOOLKIT_ROOT_DIR OR NOT EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
  message(FATAL_ERROR
    "kino_smf: no CUDA toolkit (bin/nvcc) found for ${KINO_SMF_ARCH}. Install it (JetPack "
    "includes it on a Jetson) or pass -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda")
endif()
if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD) # FindCUDA, which the CMake config of libtorch uses too
endif()
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
# lib64 on x86_64, targets/<arch>-linux/lib on a Jetson: use the directory cudart was found in
get_filename_component(CUDA_LIBRARY_DIR "${CUDA_CUDART_LIBRARY}" DIRECTORY)
link_directories("${CUDA_LIBRARY_DIR}")

# ==============================================================================
# libtorch
# ==============================================================================
# -DLIBTORCH_ROOT / $LIBTORCH_ROOT, else ~/libtorch: the libtorch zip extracted there on
# x86_64, a symlink to the torch package of the pip wheel on a Jetson (there is no libtorch
# zip for aarch64). The other usual places are tried after it, the ones of the platform first.
if(NOT LIBTORCH_ROOT AND DEFINED ENV{LIBTORCH_ROOT})
  set(LIBTORCH_ROOT "$ENV{LIBTORCH_ROOT}")
endif()

if(LIBTORCH_ROOT)
  if(NOT EXISTS "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
    message(FATAL_ERROR "kino_smf: LIBTORCH_ROOT=${LIBTORCH_ROOT} has no share/cmake/Torch/TorchConfig.cmake")
  endif()
  set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch")
elseif(NOT Torch_DIR OR NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
  # libtorch zips extracted in versioned directories (newest first) and system-wide copies
  file(GLOB _kino_smf_torch_zips LIST_DIRECTORIES true "${KINO_SMF_HOME}/libtorch*/libtorch")
  list(SORT _kino_smf_torch_zips COMPARE NATURAL ORDER DESCENDING)
  set(_kino_smf_torch_zip_roots ${_kino_smf_torch_zips} /opt/libtorch /usr/local/libtorch)

  # torch package of a pip wheel: the one python3 imports (active venv or system), then the
  # venvs in the home directory and pip --user installs
  set(_kino_smf_torch_pip_roots "")
  find_program(KINO_SMF_PYTHON3 NAMES python3)
  mark_as_advanced(KINO_SMF_PYTHON3)
  if(KINO_SMF_PYTHON3)
    execute_process(
      COMMAND "${KINO_SMF_PYTHON3}" -c
        "import importlib.util as u; s = u.find_spec('torch'); print(s.submodule_search_locations[0] if s else '')"
      OUTPUT_VARIABLE _kino_smf_python_torch
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    list(APPEND _kino_smf_torch_pip_roots ${_kino_smf_python_torch})
  endif()
  file(GLOB _kino_smf_venv_torch LIST_DIRECTORIES true
    "${KINO_SMF_HOME}/*/lib/python3*/site-packages/torch"
    "${KINO_SMF_HOME}/.*/lib/python3*/site-packages/torch")
  list(APPEND _kino_smf_torch_pip_roots ${_kino_smf_venv_torch})

  if(KINO_SMF_ARCH STREQUAL "aarch64")
    set(_kino_smf_torch_roots "${KINO_SMF_HOME}/libtorch" ${_kino_smf_torch_pip_roots} ${_kino_smf_torch_zip_roots})
  else()
    set(_kino_smf_torch_roots "${KINO_SMF_HOME}/libtorch" ${_kino_smf_torch_zip_roots} ${_kino_smf_torch_pip_roots})
  endif()
  list(REMOVE_DUPLICATES _kino_smf_torch_roots)

  foreach(_root IN LISTS _kino_smf_torch_roots)
    if(EXISTS "${_root}/share/cmake/Torch/TorchConfig.cmake")
      set(Torch_DIR "${_root}/share/cmake/Torch")
      break()
    endif()
  endforeach()

  if(NOT Torch_DIR OR NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
    if(KINO_SMF_ARCH STREQUAL "aarch64")
      set(_kino_smf_hint "install the torch wheel built for your JetPack (e.g. in a venv) and link its site-packages/torch to ~/libtorch")
    else()
      set(_kino_smf_hint "extract the libtorch zip from pytorch.org (cxx11 ABI, CUDA build matching the toolkit) to ~/libtorch")
    endif()
    string(REPLACE ";" "\n  " _kino_smf_searched "${_kino_smf_torch_roots}")
    message(FATAL_ERROR "kino_smf: libtorch not found for ${KINO_SMF_ARCH}: ${_kino_smf_hint}, "
      "or pass -DLIBTORCH_ROOT=/path/to/libtorch. Searched:\n  ${_kino_smf_searched}")
  endif()
endif()

find_package(Torch REQUIRED)
# Use the C++ standard Torch declares on its target (C++20 for torch 2.14), which does not propagate
get_target_property(TORCH_CXX_STANDARD torch CXX_STANDARD)
if(TORCH_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD ${TORCH_CXX_STANDARD})
endif()

# libtorch root, and the directory holding it: site-packages for a pip torch, where the
# TensorRT and Torch-TensorRT wheels are installed too
get_filename_component(KINO_SMF_TORCH_ROOT "${Torch_DIR}/../../.." ABSOLUTE)
set(LIBTORCH_LIBRARY_DIR "${KINO_SMF_TORCH_ROOT}/lib")
get_filename_component(_kino_smf_torch_real_root "${KINO_SMF_TORCH_ROOT}" REALPATH)
get_filename_component(KINO_SMF_TORCH_PARENT_DIR "${_kino_smf_torch_real_root}" DIRECTORY)

# ==============================================================================
# Torch-TensorRT and TensorRT
# ==============================================================================
# A TensorRT model (e.g. model_xy_yaw_trt_fp16.ts) is a TorchScript module whose engine ops
# are registered by the Torch-TensorRT runtime:
#   x86_64: the libtorchtrt release tarball (~/libtorchtrt-*-x86_64-linux/torch_tensorrt)
#           with the system TensorRT (/usr/lib/x86_64-linux-gnu)
#   Jetson: the torch_tensorrt and tensorrt wheels installed next to the pip torch
# AUTO links the runtime when it is found, ON requires it, OFF never links it. Without it
# only plain TorchScript models can be loaded.
set(KINO_SMF_TORCH_TENSORRT "AUTO" CACHE STRING "Link the Torch-TensorRT runtime: AUTO, ON or OFF")
set_property(CACHE KINO_SMF_TORCH_TENSORRT PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${KINO_SMF_TORCH_TENSORRT}" _kino_smf_trt_mode)

set(KINO_SMF_WITH_TORCH_TENSORRT FALSE)
if(_kino_smf_trt_mode STREQUAL "AUTO" OR _kino_smf_trt_mode)
  set(_kino_smf_trt_required "")
  if(NOT _kino_smf_trt_mode STREQUAL "AUTO")
    set(_kino_smf_trt_required REQUIRED)
  endif()

  set(TORCH_TENSORRT_HINTS "${KINO_SMF_TORCH_PARENT_DIR}/torch_tensorrt")
  find_package(TorchTensorRT ${_kino_smf_trt_required})
  if(TorchTensorRT_FOUND)
    # the TensorRT the runtime was built against: same soname, pip wheel next to torch first
    set(TENSORRT_LIBRARY_HINTS "${KINO_SMF_TORCH_PARENT_DIR}/tensorrt_libs")
    set(TENSORRT_SONAME_VERSION "${TORCH_TENSORRT_NVINFER_SOVERSION}")
    find_package(TensorRT ${_kino_smf_trt_required})
    if(TensorRT_FOUND)
      set(KINO_SMF_WITH_TORCH_TENSORRT TRUE)
    endif()
  endif()

  if(NOT KINO_SMF_WITH_TORCH_TENSORRT)
    message(WARNING
      "kino_smf: building WITHOUT the Torch-TensorRT runtime, so TensorRT models (*_trt_*.ts) "
      "cannot be loaded, only plain TorchScript ones. If Torch-TensorRT and TensorRT are installed "
      "somewhere else, pass -DTORCH_TENSORRT_ROOT=<dir> and/or -DTENSORRT_ROOT=<dir>, "
      "-DKINO_SMF_TORCH_TENSORRT=OFF silences this warning.")
  endif()
endif()

if(KINO_SMF_JETSON)
  set(_kino_smf_platform "${KINO_SMF_ARCH} (NVIDIA Jetson)")
else()
  set(_kino_smf_platform "${KINO_SMF_ARCH}")
endif()
message(STATUS "kino_smf: platform        ${_kino_smf_platform}")
message(STATUS "kino_smf: CUDA            ${CUDA_TOOLKIT_ROOT_DIR} (${CUDA_VERSION_STRING})")
message(STATUS "kino_smf: libtorch        ${KINO_SMF_TORCH_ROOT} (${Torch_VERSION})")
if(KINO_SMF_WITH_TORCH_TENSORRT)
  message(STATUS "kino_smf: Torch-TensorRT  ${TORCH_TENSORRT_ROOT_DIR}")
  message(STATUS "kino_smf: TensorRT        ${TENSORRT_LIBRARIES}")
  if(TORCH_TENSORRT_LIBSTDCXX)
    message(STATUS "kino_smf: libstdc++       ${TORCH_TENSORRT_LIBSTDCXX} (shipped in runtime_libs)")
  endif()
else()
  message(STATUS "kino_smf: Torch-TensorRT  not used")
endif()

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

# Deep learning libraries and the directories the executable finds them in at runtime.
# libtorch_global_deps loads the CUDA libraries of torch, as importing torch does in python.
set(KINO_SMF_DL_LIBRARIES ${TORCH_LIBRARIES})
if(EXISTS "${LIBTORCH_LIBRARY_DIR}/libtorch_global_deps.so")
  list(APPEND KINO_SMF_DL_LIBRARIES "${LIBTORCH_LIBRARY_DIR}/libtorch_global_deps.so")
endif()
set(KINO_SMF_RPATH "${LIBTORCH_LIBRARY_DIR}" "${CUDA_LIBRARY_DIR}")

if(KINO_SMF_WITH_TORCH_TENSORRT)
  # Torch-TensorRT 2.14 also needs libtorch_nvshmem.so, which its own RUNPATH cannot find
  if(EXISTS "${LIBTORCH_LIBRARY_DIR}/libtorch_nvshmem.so")
    list(APPEND KINO_SMF_DL_LIBRARIES "${LIBTORCH_LIBRARY_DIR}/libtorch_nvshmem.so")
  endif()
  list(APPEND KINO_SMF_DL_LIBRARIES ${TORCH_TENSORRT_LIBRARIES} ${TENSORRT_LIBRARIES})
  list(APPEND KINO_SMF_RPATH "${TORCH_TENSORRT_LIBRARY_DIR}" "${TENSORRT_LIBRARY_DIR}")

  # newer libstdc++ the prebuilt Torch-TensorRT needs: linked, and shipped in runtime_libs
  # next to the executable, first in its RUNPATH so that it wins over the system one
  if(TORCH_TENSORRT_LIBSTDCXX)
    set(_kino_smf_runtime_dir "${CMAKE_CURRENT_BINARY_DIR}/runtime_libs")
    file(MAKE_DIRECTORY "${_kino_smf_runtime_dir}")
    file(CREATE_LINK
      "${TORCH_TENSORRT_LIBSTDCXX}"
      "${_kino_smf_runtime_dir}/libstdc++.so.6"
      SYMBOLIC COPY_ON_ERROR)
    list(APPEND KINO_SMF_DL_LIBRARIES "${TORCH_TENSORRT_LIBSTDCXX}")
    list(INSERT KINO_SMF_RPATH 0 "$ORIGIN/runtime_libs")
  endif()
endif()
list(REMOVE_DUPLICATES KINO_SMF_RPATH)

# Note: no OpenMP here on purpose. libtorch ships its own libgomp and the dependency
# tree drags in the system one; two OpenMP runtimes in one process corrupt the heap.

# Link libraries. The TensorRT model is a TorchScript module loaded with torch::jit::load:
# nothing references the Torch-TensorRT runtime directly, it only registers the TRT engine
# ops when loaded, so --no-as-needed keeps the linker from dropping it.
ament_target_dependencies(kino_smf_move_base_planner ${dependencies})
target_link_libraries(kino_smf_move_base_planner
  ${OMPL_LIBRARIES}
  ${KINO_SMF_DL_LIBRARIES}
  ${CUDA_LIBRARIES}
)
if(KINO_SMF_WITH_TORCH_TENSORRT)
  target_compile_definitions(kino_smf_move_base_planner PRIVATE KINO_SMF_WITH_TORCH_TENSORRT=1)
else()
  target_compile_definitions(kino_smf_move_base_planner PRIVATE KINO_SMF_WITH_TORCH_TENSORRT=0)
endif()

# Runtime path of the Torch, Torch-TensorRT, CUDA and TensorRT libraries, the same in the
# build and install trees
set_target_properties(kino_smf_move_base_planner PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${KINO_SMF_RPATH}"
    LINK_FLAGS "-Wl,--no-as-needed"
)

# Install targets
install(TARGETS
  kino_smf_move_base_planner
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})
if(KINO_SMF_WITH_TORCH_TENSORRT AND TORCH_TENSORRT_LIBSTDCXX)
  install(FILES
    "${TORCH_TENSORRT_LIBSTDCXX}"
    DESTINATION "lib/${PROJECT_NAME}/runtime_libs"
    RENAME libstdc++.so.6)
endif()

# ROS 2 package setup
ament_package()
