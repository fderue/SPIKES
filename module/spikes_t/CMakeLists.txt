
set(SOURCE_FILES
    "src/displaySpikeS_T.cpp"
    "src/matchingSpikeS.cpp"
    "src/modelUpdate.cpp"
    "src/SpikeS.cpp"
    "src/SpikeS_T.cpp"
    "src/Spikes_T_Sub.cpp"
    "src/SpxEngine.cpp"
    "src/stateEstimation.cpp"
    "src/Superpixel.cpp"

)
set(INCLUDE_FILES
    "include/spikes_t/ParamSpikeS_T.h"
    "include/spikes_t/Segmentor.h"
    "include/spikes_t/SpikeS.h"
    "include/spikes_t/SpikeS_T.h"
    "include/spikes_t/SpxEngine.h"
    "include/spikes_t/Superpixel.h"
)


add_library(spikes_t ${SOURCE_FILES} ${INCLUDE_FILES})
target_include_directories(spikes_t PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


target_link_libraries(spikes_t
    PUBLIC
        ${OpenCV_LIBS}
        featengine
        utils
        sliccuda
        slic
)