find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include /usr/include/x86_64-linux-gnu /usr/local/include
)

# The pip wheel (Jetson) only ships libnvinfer.so.11 in tensorrt_libs, and no headers
find_library(NVINFER_LIB NAMES nvinfer libnvinfer.so.11 NAMES_PER_DIR
  HINTS ${TENSORRT_LIBRARY_HINTS}
  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)

find_library(NVINFER_PLUGIN_LIB NAMES nvinfer_plugin libnvinfer_plugin.so.11 NAMES_PER_DIR
  HINTS ${TENSORRT_LIBRARY_HINTS}
  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)

set(TENSORRT_LIBRARIES
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
)
get_filename_component(TENSORRT_LIBRARY_DIR "${NVINFER_LIB}" DIRECTORY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorRT DEFAULT_MSG
    NVINFER_LIB
)

mark_as_advanced(TENSORRT_INCLUDE_DIR NVINFER_LIB NVINFER_PLUGIN_LIB)
