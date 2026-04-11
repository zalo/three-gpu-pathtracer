cmake_minimum_required(VERSION 3.20)
project(tinybvh_wasm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tinybvh tinybvh_bindings.cpp)
target_include_directories(tinybvh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tinybvh)

# Emscripten-specific settings
if(EMSCRIPTEN)
    set_target_properties(tinybvh PROPERTIES SUFFIX ".js")
    target_link_options(tinybvh PRIVATE
        -sALLOW_MEMORY_GROWTH=1
        -sMODULARIZE=1
        -sEXPORT_NAME=TinyBVH
        -sEXPORT_ES6=1
        -sENVIRONMENT=web
        -sEXPORTED_FUNCTIONS=['_malloc','_free']
        -sEXPORTED_RUNTIME_METHODS=['HEAPF32','HEAPU32']
        -sASSERTIONS=2
        -sSTACK_SIZE=4194304
        -sINITIAL_MEMORY=67108864
        -sMAXIMUM_MEMORY=536870912
        --bind
        -O2
    )
    target_compile_options(tinybvh PRIVATE -O2)
endif()
