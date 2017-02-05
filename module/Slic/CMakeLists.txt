
set(SOURCE_FILES
    "src/Slic.cpp"
)
set(INCLUDE_FILES
    "include/slic/Slic.h"
)


add_library(slic ${SOURCE_FILES} ${INCLUDE_FILES})
target_include_directories(slic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


target_link_libraries(slic
    PUBLIC
        ${OpenCV_LIBS}
)