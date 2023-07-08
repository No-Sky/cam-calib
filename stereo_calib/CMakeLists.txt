cmake_minimum_required(VERSION 3.5)

Project(stereo_calib)

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


add_executable(stereo_calib stereo_calib.cpp)

# add_executable(stereo_calib_example stereo_calib_example.cpp)

add_executable(stereo_calib_ex1 stereo_calib_ex1.cpp)


target_link_libraries(stereo_calib
    -ljpeg
    ${OpenCV_LIBS}
)


# target_link_libraries(stereo_calib_example
#     -ljpeg
#     ${OpenCV_LIBS}
# )

target_link_libraries(stereo_calib_ex1
    -ljpeg
    ${OpenCV_LIBS}
)