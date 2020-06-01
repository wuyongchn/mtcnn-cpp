cmake_minimum_required(VERSION 3.14)
project(face_detection)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(face_detection
        main.cc bounding_box.h bounding_box.cc  utils.h utils.cc mtcnn.cc mtcnn.h
        )
target_link_libraries(face_detection ${OpenCV_LIBS}  ${Caffe_LIBRARIES})
