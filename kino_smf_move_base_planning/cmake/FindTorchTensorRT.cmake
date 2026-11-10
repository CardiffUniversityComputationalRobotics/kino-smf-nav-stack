# FindTorchTensorRT
# -----------------
# Finds the Torch-TensorRT runtime (libtorchtrt_runtime.so, libtorchtrt_plugins.so), which
# registers the TensorRT engine ops of TensorRT-compiled TorchScript modules.
#
# Searched in this order (the first directory with lib/libtorchtrt_runtime.so is used):
#   TORCH_TENSORRT_ROOT   CMake or environment variable, nothing else is tried when it is set
#   TORCH_TENSORRT_HINTS  e.g. site-packages/torch_tensorrt next to a pip torch (Jetson)
#   ~/torch_tensorrt and ~/libtorchtrt-*-<processor>-linux/torch_tensorrt: the release
#                         tarball extracted in the home directory (x86_64), newest first
#
# A prebuilt Torch-TensorRT may need a newer libstdc++ than the system one (the x86_64
# tarball does on Ubuntu 24.04). Then a libstdc++.so.6 recent enough is looked for among the
# ones shipped with Nsight Compute / Nsight Systems, unless TORCH_TENSORRT_LIBSTDCXX gives one.
#
# Defines:
#   TorchTensorRT_FOUND
#   TORCH_TENSORRT_ROOT_DIR, TORCH_TENSORRT_LIBRARY_DIR
#   TORCH_TENSORRT_LIBRARIES          the runtime, and the plugins library when present
#   TORCH_TENSORRT_NVINFER_SOVERSION  major version of the TensorRT it links (10: libnvinfer.so.10)
#   TORCH_TENSORRT_LIBSTDCXX          libstdc++.so.6 to link and ship with the executable,
#                                     empty when the system one is recent enough

# Highest GLIBCXX_3.4.x and CXXABI_1.3.x versions named in the given binaries: the versions
# a library needs, or the versions libstdc++ itself provides
function(_torch_trt_libstdcxx_versions glibcxx_var cxxabi_var)
  set(_glibcxx "")
  set(_cxxabi "")
  foreach(_file IN LISTS ARGN)
    if(EXISTS "${_file}")
      file(STRINGS "${_file}" _tags REGEX "^(GLIBCXX_3\\.4|CXXABI_1\\.3)(\\.[0-9]+)?$")
      foreach(_tag IN LISTS _tags)
        string(REGEX REPLACE "^[A-Z]+_" "" _version "${_tag}")
        if(_tag MATCHES "^GLIBCXX_")
          if(NOT _glibcxx OR _version VERSION_GREATER _glibcxx)
            set(_glibcxx "${_version}")
          endif()
        elseif(NOT _cxxabi OR _version VERSION_GREATER _cxxabi)
          set(_cxxabi "${_version}")
        endif()
      endforeach()
    endif()
  endforeach()
  set(${glibcxx_var} "${_glibcxx}" PARENT_SCOPE)
  set(${cxxabi_var} "${_cxxabi}" PARENT_SCOPE)
endfunction()

# TRUE when the provided versions cover the needed ones (an unknown need is covered)
function(_torch_trt_libstdcxx_covers out_var need_glibcxx need_cxxabi have_glibcxx have_cxxabi)
  set(_covers TRUE)
  if(need_glibcxx AND (NOT have_glibcxx OR need_glibcxx VERSION_GREATER have_glibcxx))
    set(_covers FALSE)
  endif()
  if(need_cxxabi AND (NOT have_cxxabi OR need_cxxabi VERSION_GREATER have_cxxabi))
    set(_covers FALSE)
  endif()
  set(${out_var} ${_covers} PARENT_SCOPE)
endfunction()

set(_torch_trt_forced_libstdcxx "${TORCH_TENSORRT_LIBSTDCXX}")
set(TORCH_TENSORRT_ROOT_DIR "")
set(TORCH_TENSORRT_LIBRARY_DIR "")
set(TORCH_TENSORRT_LIBRARIES "")
set(TORCH_TENSORRT_NVINFER_SOVERSION "")
set(TORCH_TENSORRT_LIBSTDCXX "")
set(TORCH_TENSORRT_LIBSTDCXX_OK TRUE)
set(_torch_trt_reason "")

if(NOT TORCH_TENSORRT_ROOT AND DEFINED ENV{TORCH_TENSORRT_ROOT})
  set(TORCH_TENSORRT_ROOT "$ENV{TORCH_TENSORRT_ROOT}")
endif()
if(TORCH_TENSORRT_ROOT)
  set(_torch_trt_candidates "${TORCH_TENSORRT_ROOT}")
else()
  file(GLOB _torch_trt_tarballs LIST_DIRECTORIES true
    "$ENV{HOME}/libtorchtrt*-${CMAKE_SYSTEM_PROCESSOR}-linux/torch_tensorrt")
  list(SORT _torch_trt_tarballs COMPARE NATURAL ORDER DESCENDING)
  set(_torch_trt_candidates ${TORCH_TENSORRT_HINTS} "$ENV{HOME}/torch_tensorrt" ${_torch_trt_tarballs})
endif()

foreach(_dir IN LISTS _torch_trt_candidates)
  if(_dir AND EXISTS "${_dir}/lib/libtorchtrt_runtime.so")
    set(TORCH_TENSORRT_ROOT_DIR "${_dir}")
    break()
  endif()
endforeach()

if(TORCH_TENSORRT_ROOT_DIR)
  set(TORCH_TENSORRT_LIBRARY_DIR "${TORCH_TENSORRT_ROOT_DIR}/lib")
  set(_torch_trt_runtime "${TORCH_TENSORRT_LIBRARY_DIR}/libtorchtrt_runtime.so")
  set(TORCH_TENSORRT_LIBRARIES "${_torch_trt_runtime}")
  if(EXISTS "${TORCH_TENSORRT_LIBRARY_DIR}/libtorchtrt_plugins.so")
    list(APPEND TORCH_TENSORRT_LIBRARIES "${TORCH_TENSORRT_LIBRARY_DIR}/libtorchtrt_plugins.so")
  endif()

  # the TensorRT it was built against, so that the same libnvinfer.so.<N> gets linked
  file(STRINGS "${_torch_trt_runtime}" _torch_trt_nvinfer REGEX "^libnvinfer\\.so\\.[0-9]+$")
  if(_torch_trt_nvinfer)
    list(GET _torch_trt_nvinfer 0 _torch_trt_nvinfer)
    string(REGEX REPLACE "^libnvinfer\\.so\\." "" TORCH_TENSORRT_NVINFER_SOVERSION "${_torch_trt_nvinfer}")
  endif()

  if(_torch_trt_forced_libstdcxx)
    if(EXISTS "${_torch_trt_forced_libstdcxx}")
      get_filename_component(TORCH_TENSORRT_LIBSTDCXX "${_torch_trt_forced_libstdcxx}" REALPATH)
    else()
      set(TORCH_TENSORRT_LIBSTDCXX_OK FALSE)
      set(_torch_trt_reason "TORCH_TENSORRT_LIBSTDCXX=${_torch_trt_forced_libstdcxx} does not exist")
    endif()
  else()
    # what it needs against what the libstdc++ the compiler links provides
    _torch_trt_libstdcxx_versions(_need_glibcxx _need_cxxabi ${TORCH_TENSORRT_LIBRARIES})
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.so.6
      OUTPUT_VARIABLE _torch_trt_system_libstdcxx
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    _torch_trt_libstdcxx_versions(_have_glibcxx _have_cxxabi "${_torch_trt_system_libstdcxx}")
    set(_torch_trt_system_ok TRUE)
    if(_have_glibcxx) # skipped when the system libstdc++ could not be read
      _torch_trt_libstdcxx_covers(_torch_trt_system_ok
        "${_need_glibcxx}" "${_need_cxxabi}" "${_have_glibcxx}" "${_have_cxxabi}")
    endif()

    if(NOT _torch_trt_system_ok)
      file(GLOB _torch_trt_libstdcxx_candidates
        "/opt/nvidia/nsight-compute/*/host/*/libstdc++.so.6"
        "/opt/nvidia/nsight-systems/*/host-linux-*/libstdc++.so.6"
        "/usr/local/cuda*/nsight-compute*/host/*/libstdc++.so.6"
        "/usr/local/cuda*/nsight-systems*/host-linux-*/libstdc++.so.6")
      set(_torch_trt_best "")
      set(_torch_trt_best_glibcxx "0")
      foreach(_candidate IN LISTS _torch_trt_libstdcxx_candidates)
        _torch_trt_libstdcxx_versions(_candidate_glibcxx _candidate_cxxabi "${_candidate}")
        _torch_trt_libstdcxx_covers(_candidate_ok
          "${_need_glibcxx}" "${_need_cxxabi}" "${_candidate_glibcxx}" "${_candidate_cxxabi}")
        if(_candidate_ok AND _candidate_glibcxx VERSION_GREATER _torch_trt_best_glibcxx)
          set(_torch_trt_best "${_candidate}")
          set(_torch_trt_best_glibcxx "${_candidate_glibcxx}")
        endif()
      endforeach()

      if(_torch_trt_best)
        get_filename_component(TORCH_TENSORRT_LIBSTDCXX "${_torch_trt_best}" REALPATH)
      else()
        set(TORCH_TENSORRT_LIBSTDCXX_OK FALSE)
        set(_torch_trt_needs "GLIBCXX_${_need_glibcxx}")
        if(_need_cxxabi)
          string(APPEND _torch_trt_needs " and CXXABI_${_need_cxxabi}")
        endif()
        set(_torch_trt_has "GLIBCXX_${_have_glibcxx}")
        if(_have_cxxabi)
          string(APPEND _torch_trt_has " and CXXABI_${_have_cxxabi}")
        endif()
        string(CONCAT _torch_trt_reason
          "it needs ${_torch_trt_needs} but the system libstdc++ provides ${_torch_trt_has} "
          "and no newer libstdc++.so.6 was found, pass one with -DTORCH_TENSORRT_LIBSTDCXX=/path/to/libstdc++.so.6")
      endif()
    endif()
  endif()
endif()

set(_torch_trt_fphsa_reason "")
if(_torch_trt_reason)
  set(_torch_trt_fphsa_reason REASON_FAILURE_MESSAGE "${_torch_trt_reason}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TorchTensorRT
  REQUIRED_VARS TORCH_TENSORRT_ROOT_DIR TORCH_TENSORRT_LIBSTDCXX_OK
  ${_torch_trt_fphsa_reason}
)
