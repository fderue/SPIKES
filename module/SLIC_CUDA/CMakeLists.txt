
set(SOURCE_FILES
    "src/SlicCuda.cpp"
)
set(INCLUDE_FILES
    "include/sliccuda/SlicCuda.hpp"
)

add_library(sliccuda ${SOURCE_FILES} ${INCLUDE_FILES})
target_include_directories(sliccuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(sliccuda
                PUBLIC
                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/cuda>"
                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda>"
                    ${CUDA_INCLUDE_DIRS}
            )


cuda_add_library(sliccudalib "cuda/SlicCuda.cu" "cuda/SlicCuda.cuh")



target_link_libraries(sliccuda
    PUBLIC
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        sliccudalib
)
