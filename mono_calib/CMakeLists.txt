cmake_minimum_required(VERSION 3.5)

Project(mono_calib)

# Enable C++11
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

##下面是输出信息
message(STATUS "Opnecv ;ibrary status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}  ")


include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
) 


add_executable(mono_calib mono_calib.cpp)


target_link_libraries(mono_calib
    -ljpeg
    ${OpenCV_LIBS}
)