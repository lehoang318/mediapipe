cmake_minimum_required(VERSION 3.16)
project(face_video CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(HOME_LOCAL "$ENV{HOME}/.local")

# --- OpenCV (installed at ~/.local) ---
set(OpenCV_DIR "${HOME_LOCAL}/lib/cmake/opencv4" CACHE PATH "OpenCV cmake config dir")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

# --- ONNX Runtime GPU (CUDA 13 build, installed at ~/.local).
# The shipped onnxruntimeConfig.cmake assumes a /usr/local layout (lib64 +
# include/onnxruntime), so locate headers and the library manually.
set(ORT_ROOT "${HOME_LOCAL}" CACHE PATH "onnxruntime install prefix")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ORT_ROOT}/include/onnxruntime-gpu_cuda13-1.30.0"
        "${ORT_ROOT}/include/onnxruntime"
        "${ORT_ROOT}/include"
  REQUIRED)
find_library(ORT_LIBRARY onnxruntime
  HINTS "${ORT_ROOT}/lib" "${ORT_ROOT}/lib64"
  REQUIRED)
message(STATUS "ORT headers : ${ORT_INCLUDE_DIR}")
message(STATUS "ORT library : ${ORT_LIBRARY}")
message(STATUS "OpenCV      : ${OpenCV_VERSION} (${OpenCV_DIR})")

add_executable(face_video
  src/main.cpp
  src/face_pipeline.cpp
  src/anchors.cpp
  src/profiler.cpp
  src/draw.cpp)

target_include_directories(face_video PRIVATE src ${ORT_INCLUDE_DIR})
target_link_libraries(face_video PRIVATE ${OpenCV_LIBS} ${ORT_LIBRARY})

# Let the loader find the ~/.local shared objects at run time.
set(RPATHS "${ORT_ROOT}/lib" "${ORT_ROOT}/lib64" "${OpenCV_DIR}/../..")
set_target_properties(face_video PROPERTIES
  BUILD_RPATH "${RPATHS}"
  INSTALL_RPATH "${RPATHS}")
