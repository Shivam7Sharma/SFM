# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shivam/Computer_Vision/SFM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shivam/Computer_Vision/SFM/build

# Include any dependencies generated for this target.
include CMakeFiles/SFMProject.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SFMProject.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SFMProject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SFMProject.dir/flags.make

CMakeFiles/SFMProject.dir/src/main.cpp.o: CMakeFiles/SFMProject.dir/flags.make
CMakeFiles/SFMProject.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/SFMProject.dir/src/main.cpp.o: CMakeFiles/SFMProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shivam/Computer_Vision/SFM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SFMProject.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SFMProject.dir/src/main.cpp.o -MF CMakeFiles/SFMProject.dir/src/main.cpp.o.d -o CMakeFiles/SFMProject.dir/src/main.cpp.o -c /home/shivam/Computer_Vision/SFM/src/main.cpp

CMakeFiles/SFMProject.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SFMProject.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shivam/Computer_Vision/SFM/src/main.cpp > CMakeFiles/SFMProject.dir/src/main.cpp.i

CMakeFiles/SFMProject.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SFMProject.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shivam/Computer_Vision/SFM/src/main.cpp -o CMakeFiles/SFMProject.dir/src/main.cpp.s

CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.o: CMakeFiles/SFMProject.dir/flags.make
CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.o: ../src/CameraParameters.cpp
CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.o: CMakeFiles/SFMProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shivam/Computer_Vision/SFM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.o -MF CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.o.d -o CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.o -c /home/shivam/Computer_Vision/SFM/src/CameraParameters.cpp

CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shivam/Computer_Vision/SFM/src/CameraParameters.cpp > CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.i

CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shivam/Computer_Vision/SFM/src/CameraParameters.cpp -o CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.s

CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.o: CMakeFiles/SFMProject.dir/flags.make
CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.o: ../src/ImageLoader.cpp
CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.o: CMakeFiles/SFMProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shivam/Computer_Vision/SFM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.o -MF CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.o.d -o CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.o -c /home/shivam/Computer_Vision/SFM/src/ImageLoader.cpp

CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shivam/Computer_Vision/SFM/src/ImageLoader.cpp > CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.i

CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shivam/Computer_Vision/SFM/src/ImageLoader.cpp -o CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.s

CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.o: CMakeFiles/SFMProject.dir/flags.make
CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.o: ../src/FeatureMatcher.cpp
CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.o: CMakeFiles/SFMProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shivam/Computer_Vision/SFM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.o -MF CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.o.d -o CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.o -c /home/shivam/Computer_Vision/SFM/src/FeatureMatcher.cpp

CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shivam/Computer_Vision/SFM/src/FeatureMatcher.cpp > CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.i

CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shivam/Computer_Vision/SFM/src/FeatureMatcher.cpp -o CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.s

CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.o: CMakeFiles/SFMProject.dir/flags.make
CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.o: ../src/ImageRegistration.cpp
CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.o: CMakeFiles/SFMProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shivam/Computer_Vision/SFM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.o -MF CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.o.d -o CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.o -c /home/shivam/Computer_Vision/SFM/src/ImageRegistration.cpp

CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shivam/Computer_Vision/SFM/src/ImageRegistration.cpp > CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.i

CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shivam/Computer_Vision/SFM/src/ImageRegistration.cpp -o CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.s

CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.o: CMakeFiles/SFMProject.dir/flags.make
CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.o: ../src/SfMInitializer.cpp
CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.o: CMakeFiles/SFMProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shivam/Computer_Vision/SFM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.o -MF CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.o.d -o CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.o -c /home/shivam/Computer_Vision/SFM/src/SfMInitializer.cpp

CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shivam/Computer_Vision/SFM/src/SfMInitializer.cpp > CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.i

CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shivam/Computer_Vision/SFM/src/SfMInitializer.cpp -o CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.s

# Object files for target SFMProject
SFMProject_OBJECTS = \
"CMakeFiles/SFMProject.dir/src/main.cpp.o" \
"CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.o" \
"CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.o" \
"CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.o" \
"CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.o" \
"CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.o"

# External object files for target SFMProject
SFMProject_EXTERNAL_OBJECTS =

SFMProject: CMakeFiles/SFMProject.dir/src/main.cpp.o
SFMProject: CMakeFiles/SFMProject.dir/src/CameraParameters.cpp.o
SFMProject: CMakeFiles/SFMProject.dir/src/ImageLoader.cpp.o
SFMProject: CMakeFiles/SFMProject.dir/src/FeatureMatcher.cpp.o
SFMProject: CMakeFiles/SFMProject.dir/src/ImageRegistration.cpp.o
SFMProject: CMakeFiles/SFMProject.dir/src/SfMInitializer.cpp.o
SFMProject: CMakeFiles/SFMProject.dir/build.make
SFMProject: /usr/local/lib/libopencv_gapi.so.4.9.0
SFMProject: /usr/local/lib/libopencv_stitching.so.4.9.0
SFMProject: /usr/local/lib/libopencv_alphamat.so.4.9.0
SFMProject: /usr/local/lib/libopencv_aruco.so.4.9.0
SFMProject: /usr/local/lib/libopencv_bgsegm.so.4.9.0
SFMProject: /usr/local/lib/libopencv_bioinspired.so.4.9.0
SFMProject: /usr/local/lib/libopencv_ccalib.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudabgsegm.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudafeatures2d.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudaobjdetect.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudastereo.so.4.9.0
SFMProject: /usr/local/lib/libopencv_dnn_objdetect.so.4.9.0
SFMProject: /usr/local/lib/libopencv_dnn_superres.so.4.9.0
SFMProject: /usr/local/lib/libopencv_dpm.so.4.9.0
SFMProject: /usr/local/lib/libopencv_face.so.4.9.0
SFMProject: /usr/local/lib/libopencv_freetype.so.4.9.0
SFMProject: /usr/local/lib/libopencv_fuzzy.so.4.9.0
SFMProject: /usr/local/lib/libopencv_hdf.so.4.9.0
SFMProject: /usr/local/lib/libopencv_hfs.so.4.9.0
SFMProject: /usr/local/lib/libopencv_img_hash.so.4.9.0
SFMProject: /usr/local/lib/libopencv_intensity_transform.so.4.9.0
SFMProject: /usr/local/lib/libopencv_line_descriptor.so.4.9.0
SFMProject: /usr/local/lib/libopencv_mcc.so.4.9.0
SFMProject: /usr/local/lib/libopencv_quality.so.4.9.0
SFMProject: /usr/local/lib/libopencv_rapid.so.4.9.0
SFMProject: /usr/local/lib/libopencv_reg.so.4.9.0
SFMProject: /usr/local/lib/libopencv_rgbd.so.4.9.0
SFMProject: /usr/local/lib/libopencv_saliency.so.4.9.0
SFMProject: /usr/local/lib/libopencv_sfm.so.4.9.0
SFMProject: /usr/local/lib/libopencv_signal.so.4.9.0
SFMProject: /usr/local/lib/libopencv_stereo.so.4.9.0
SFMProject: /usr/local/lib/libopencv_structured_light.so.4.9.0
SFMProject: /usr/local/lib/libopencv_superres.so.4.9.0
SFMProject: /usr/local/lib/libopencv_surface_matching.so.4.9.0
SFMProject: /usr/local/lib/libopencv_tracking.so.4.9.0
SFMProject: /usr/local/lib/libopencv_videostab.so.4.9.0
SFMProject: /usr/local/lib/libopencv_viz.so.4.9.0
SFMProject: /usr/local/lib/libopencv_wechat_qrcode.so.4.9.0
SFMProject: /usr/local/lib/libopencv_xfeatures2d.so.4.9.0
SFMProject: /usr/local/lib/libopencv_xobjdetect.so.4.9.0
SFMProject: /usr/local/lib/libopencv_xphoto.so.4.9.0
SFMProject: /usr/local/lib/libg2o_core.so
SFMProject: /usr/local/lib/libg2o_stuff.so
SFMProject: /usr/local/lib/libg2o_solver_csparse.so
SFMProject: /usr/local/lib/libg2o_csparse_extension.so
SFMProject: /usr/local/lib/libg2o_types_sba.so
SFMProject: /usr/local/lib/libg2o_types_slam3d.so
SFMProject: /usr/local/lib/libg2o_types_data.so
SFMProject: /usr/local/lib/libg2o_types_icp.so
SFMProject: /usr/local/lib/libg2o_types_sba.so
SFMProject: /usr/local/lib/libg2o_types_sim3.so
SFMProject: /usr/local/lib/libg2o_types_slam2d.so
SFMProject: /usr/local/lib/libg2o_types_slam3d.so
SFMProject: /usr/local/lib/libopencv_shape.so.4.9.0
SFMProject: /usr/local/lib/libopencv_highgui.so.4.9.0
SFMProject: /usr/local/lib/libopencv_datasets.so.4.9.0
SFMProject: /usr/local/lib/libopencv_plot.so.4.9.0
SFMProject: /usr/local/lib/libopencv_text.so.4.9.0
SFMProject: /usr/local/lib/libopencv_ml.so.4.9.0
SFMProject: /usr/local/lib/libopencv_phase_unwrapping.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudacodec.so.4.9.0
SFMProject: /usr/local/lib/libopencv_videoio.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudaoptflow.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudalegacy.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudawarping.so.4.9.0
SFMProject: /usr/local/lib/libopencv_optflow.so.4.9.0
SFMProject: /usr/local/lib/libopencv_ximgproc.so.4.9.0
SFMProject: /usr/local/lib/libopencv_video.so.4.9.0
SFMProject: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
SFMProject: /usr/local/lib/libopencv_objdetect.so.4.9.0
SFMProject: /usr/local/lib/libopencv_calib3d.so.4.9.0
SFMProject: /usr/local/lib/libopencv_dnn.so.4.9.0
SFMProject: /usr/local/lib/libopencv_features2d.so.4.9.0
SFMProject: /usr/local/lib/libopencv_flann.so.4.9.0
SFMProject: /usr/local/lib/libopencv_photo.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudaimgproc.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudafilters.so.4.9.0
SFMProject: /usr/local/lib/libopencv_imgproc.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudaarithm.so.4.9.0
SFMProject: /usr/local/lib/libopencv_core.so.4.9.0
SFMProject: /usr/local/lib/libopencv_cudev.so.4.9.0
SFMProject: /usr/local/lib/libg2o_types_sba.so
SFMProject: /usr/local/lib/libg2o_types_data.so
SFMProject: /usr/local/lib/libg2o_types_icp.so
SFMProject: /usr/local/lib/libg2o_types_sim3.so
SFMProject: /usr/local/lib/libg2o_types_slam2d.so
SFMProject: CMakeFiles/SFMProject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shivam/Computer_Vision/SFM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable SFMProject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SFMProject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SFMProject.dir/build: SFMProject
.PHONY : CMakeFiles/SFMProject.dir/build

CMakeFiles/SFMProject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SFMProject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SFMProject.dir/clean

CMakeFiles/SFMProject.dir/depend:
	cd /home/shivam/Computer_Vision/SFM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shivam/Computer_Vision/SFM /home/shivam/Computer_Vision/SFM /home/shivam/Computer_Vision/SFM/build /home/shivam/Computer_Vision/SFM/build /home/shivam/Computer_Vision/SFM/build/CMakeFiles/SFMProject.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SFMProject.dir/depend

