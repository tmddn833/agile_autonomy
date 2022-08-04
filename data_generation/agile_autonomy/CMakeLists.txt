cmake_minimum_required(VERSION 2.8.3)
project(agile_autonomy)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
add_compile_options(-O3)

find_package(catkin_simple REQUIRED COMPONENTS)

catkin_package(
        LIBRARIES
        CATKIN_DEPENDS
)
catkin_simple(ALL_DEPS_REQUIRED)


cs_add_executable(agile_autonomy_airsim src/agile_autonomy_airsim.cpp)

target_include_directories(agile_autonomy_airsim PUBLIC
        ../../../rpg_mpc/model/quadrotor_mpc_codegen/
        ../../../rpg_mpc/externals/qpoases
        ../../../rpg_mpc/externals/qpoases/INCLUDE
        ../../../rpg_mpc/externals/qpoases/SRC)

target_link_libraries(
        agile_autonomy_airsim stdc++fs
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
)

cs_install()
cs_export()
