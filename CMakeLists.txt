cmake_minimum_required(VERSION 3.10)
project(SFMProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)
# Find Eigen package
find_package(Eigen3 REQUIRED)

# Include directories for Eigen
include_directories(${EIGEN3_INCLUDE_DIR})

# Find g2o package
set (G2O_ROOT  "/usr/local")
set(G2O_DIR "/usr/local/lib/cmake/g2o") # Adjust the path as necessary
find_package(G2O REQUIRED)



# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories for g2o
include_directories(${G2O_INCLUDE_DIRS})

# Add directory containing header files
include_directories(include)

# Add source files
add_executable(SFMProject
    src/main.cpp
    src/CameraParameters.cpp
    src/ImageLoader.cpp
    src/FeatureMatcher.cpp
    src/ImageRegistration.cpp
    src/SfMInitializer.cpp
)

# Link against OpenCV and g2o libraries
target_link_libraries(SFMProject ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_TYPES_SBA} ${G2O_TYPES_SLAM3D} ${G2O_TYPES_SLAM3D_ADDONS} ${G2O_TYPES_DATA} ${G2O_TYPES_ICP} ${G2O_TYPES_SBA} ${G2O_TYPES_SIM3} ${G2O_TYPES_SLAM2D} ${G2O_TYPES_SLAM2D_ADDONS} ${G2O_TYPES_SLAM3D})