cmake_minimum_required(VERSION 3.10)
project(SFMProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
# set(OpenCV_DIR "/home/shivam/opencv_build/opencv/build")
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
# include_directories(
    
#     /home/shivam/opencv_build/opencv_contrib/modules/sfm/include/opencv2
# )



# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

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

# Link against OpenCV libraries
target_link_libraries(SFMProject ${OpenCV_LIBS})
