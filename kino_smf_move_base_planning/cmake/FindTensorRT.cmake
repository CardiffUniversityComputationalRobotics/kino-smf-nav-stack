# FindTensorRT
# ------------
# Finds the TensorRT libraries (nvinfer, nvinfer_plugin) used by the Torch-TensorRT runtime.
# Headers are not needed.
#
# Searched in this order:
#   TENSORRT_LIBRARY_HINTS  e.g. site-packages/tensorrt_libs next to a pip torch (Jetson)
#   TENSORRT_ROOT           CMake or environment variable, e.g. an extracted TensorRT tarball
#   /usr/lib/<multiarch>    the apt / JetPack packages (x86_64-linux-gnu, aarch64-linux-gnu)
#
# With TENSORRT_SONAME_VERSION (e.g. 10) only libnvinfer.so.10 and libnvinfer_plugin.so.10
# are accepted, so that TensorRT matches the version Torch-TensorRT was built against.
#
# Defines TensorRT_FOUND, TENSORRT_LIBRARIES and TENSORRT_LIBRARY_DIR.

if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
endif()

set(_tensorrt_hints ${TENSORRT_LIBRARY_HINTS})
if(TENSORRT_ROOT)
  list(APPEND _tensorrt_hints "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64")
endif()
set(_tensorrt_paths /usr/local/lib /usr/lib)
if(CMAKE_LIBRARY_ARCHITECTURE)
  list(INSERT _tensorrt_paths 0 "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
endif()

if(TENSORRT_SONAME_VERSION)
  set(_nvinfer_names "libnvinfer.so.${TENSORRT_SONAME_VERSION}")
  set(_nvinfer_plugin_names "libnvinfer_plugin.so.${TENSORRT_SONAME_VERSION}")
  # forget the libraries a previous run found for another version
  foreach(_var NVINFER_LIB NVINFER_PLUGIN_LIB)
    if(${_var} AND NOT ${_var} MATCHES "\\.so\\.${TENSORRT_SONAME_VERSION}$")
      unset(${_var} CACHE)
    endif()
  endforeach()
else()
  # the pip wheels only ship the versioned sonames (TensorRT 10 or 11)
  set(_nvinfer_names nvinfer libnvinfer.so.11 libnvinfer.so.10)
  set(_nvinfer_plugin_names nvinfer_plugin libnvinfer_plugin.so.11 libnvinfer_plugin.so.10)
endif()

find_library(NVINFER_LIB NAMES ${_nvinfer_names} NAMES_PER_DIR
  HINTS ${_tensorrt_hints}
  PATHS ${_tensorrt_paths}
)
find_library(NVINFER_PLUGIN_LIB NAMES ${_nvinfer_plugin_names} NAMES_PER_DIR
  HINTS ${_tensorrt_hints}
  PATHS ${_tensorrt_paths}
)

set(TENSORRT_LIBRARIES "")
set(TENSORRT_LIBRARY_DIR "")
if(NVINFER_LIB)
  set(TENSORRT_LIBRARIES "${NVINFER_LIB}")
  get_filename_component(TENSORRT_LIBRARY_DIR "${NVINFER_LIB}" DIRECTORY)
  if(NVINFER_PLUGIN_LIB)
    list(APPEND TENSORRT_LIBRARIES "${NVINFER_PLUGIN_LIB}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT REQUIRED_VARS NVINFER_LIB)

mark_as_advanced(NVINFER_LIB NVINFER_PLUGIN_LIB)
