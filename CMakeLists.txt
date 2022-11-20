cmake_minimum_required(VERSION 3.8)
project(HighDynamicRangeRender
        VERSION 1.0.0
        DESCRIPTION "Rendering High Dynamic Range Images"
        )
find_package( OpenCV REQUIRED )
file(GLOB_RECURSE SRC_FILES src/*.cpp)

add_executable(main main.cpp ${SRC_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

target_link_libraries( main ${OpenCV_LIBS})
target_compile_features( main PRIVATE cxx_std_17)
target_include_directories( main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
