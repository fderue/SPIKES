
set(SOURCE_FILES
    "src/KpEngine.cpp"
)
set(INCLUDE_FILES
    "include/featengine/KpEngine.h"
    "include/featengine/MatchEngine.h"
)


add_library(featengine ${SOURCE_FILES} ${INCLUDE_FILES})
target_include_directories(featengine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


target_link_libraries(featengine
    PUBLIC
        ${OpenCV_LIBS}
)