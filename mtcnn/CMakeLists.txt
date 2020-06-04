find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

aux_source_directory(./ MTCNN_SRC)
add_library(mtcnn ${MTCNN_SRC})
target_link_libraries(mtcnn ${Caffe_LIBRARIES} bbox)
