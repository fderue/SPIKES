
set(SOURCE_FILES
    "src/funUtils.cpp"
)
set(INCLUDE_FILES
    "include/utils/funUtils.h"
)


add_library(utils ${SOURCE_FILES} ${INCLUDE_FILES})
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


target_link_libraries(utils
    PUBLIC
        ${OpenCV_LIBS}
)