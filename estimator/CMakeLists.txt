cmake_minimum_required(VERSION 2.8.3)
project(estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models)

find_package(OpenCV REQUIRED)
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

#不打算给其他软件包作依赖
catkin_package()


#[[
add_library(vins_lib
    src/estimator/parameters.cpp
)

target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
]]

add_executable(estimator_exc src/rosNodeTest.cpp)
target_link_libraries(estimator_exc ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}) 



