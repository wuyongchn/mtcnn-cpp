cmake_minimum_required(VERSION 3.5)
project(face_detection)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
add_subdirectory(bbox)
add_subdirectory(mtcnn)
add_subdirectory(umeyama)
add_executable(face_detection main.cc)
target_link_libraries(face_detection ${OpenCV_LIBS} mtcnn umeyama)
