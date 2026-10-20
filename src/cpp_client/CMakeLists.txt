cmake_minimum_required(VERSION 3.20)
project(donkey_cpp_client CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- 依赖 ---
# nlohmann/json: brew 安装(/opt/homebrew/include),或 -DJSON_INCLUDE_DIR=...
find_path(JSON_INCLUDE_DIR nlohmann/json.hpp
  PATHS /opt/homebrew/include /usr/local/include)
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "nlohmann/json.hpp not found; brew install nlohmann-json")
endif()

# ONNX Runtime: brew 安装;可用 -DONNXRUNTIME_ROOT=... 覆盖
set(ONNXRUNTIME_ROOT "" CACHE PATH "onnxruntime root (contains include/ and lib/)")
if(NOT ONNXRUNTIME_ROOT)
  foreach(cand /opt/homebrew/opt/onnxruntime /usr/local/opt/onnxruntime /opt/homebrew /usr/local)
    if(EXISTS ${cand}/include/onnxruntime/core/session/onnxruntime_c_api.h)
      set(ONNXRUNTIME_ROOT ${cand})
      break()
    endif()
    if(EXISTS ${cand}/include/onnxruntime/onnxruntime_c_api.h)
      set(ONNXRUNTIME_ROOT ${cand})
      break()
    endif()
    if(EXISTS ${cand}/include/onnxruntime_c_api.h)
      set(ONNXRUNTIME_ROOT ${cand})
      break()
    endif()
  endforeach()
endif()
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "onnxruntime headers not found; brew install onnxruntime")
endif()

find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS ${ONNXRUNTIME_ROOT}/lib /opt/homebrew/lib /usr/local/lib)
if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "libonnxruntime not found")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
  PATHS ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
        ${ONNXRUNTIME_ROOT}/include/onnxruntime
        ${ONNXRUNTIME_ROOT}/include
        /opt/homebrew/include/onnxruntime/core/session
        /opt/homebrew/include/onnxruntime
        /usr/local/include/onnxruntime/core/session
        /usr/local/include/onnxruntime)

add_executable(dkcpp main.cpp)
target_include_directories(dkcpp PRIVATE ${JSON_INCLUDE_DIR} ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(dkcpp PRIVATE ${ONNXRUNTIME_LIB})
if(APPLE)
  # brew onnxruntime 是 keg-only 风格路径,给 dylib 加 rpath 便于直接运行
  get_filename_component(_ort_libdir ${ONNXRUNTIME_LIB} DIRECTORY)
  set_target_properties(dkcpp PROPERTIES BUILD_RPATH "${_ort_libdir}")
endif()
