# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/lbl_wrapper.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lbl_wrapper.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lbl_wrapper.dir/flags.make

CMakeFiles/lbl_wrapper.dir/lblcube.cpp.o: CMakeFiles/lbl_wrapper.dir/flags.make
CMakeFiles/lbl_wrapper.dir/lblcube.cpp.o: ../lblcube.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lbl_wrapper.dir/lblcube.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lbl_wrapper.dir/lblcube.cpp.o -c /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/lblcube.cpp

CMakeFiles/lbl_wrapper.dir/lblcube.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lbl_wrapper.dir/lblcube.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/lblcube.cpp > CMakeFiles/lbl_wrapper.dir/lblcube.cpp.i

CMakeFiles/lbl_wrapper.dir/lblcube.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lbl_wrapper.dir/lblcube.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/lblcube.cpp -o CMakeFiles/lbl_wrapper.dir/lblcube.cpp.s

# Object files for target lbl_wrapper
lbl_wrapper_OBJECTS = \
"CMakeFiles/lbl_wrapper.dir/lblcube.cpp.o"

# External object files for target lbl_wrapper
lbl_wrapper_EXTERNAL_OBJECTS =

lbl_wrapper: CMakeFiles/lbl_wrapper.dir/lblcube.cpp.o
lbl_wrapper: CMakeFiles/lbl_wrapper.dir/build.make
lbl_wrapper: CMakeFiles/lbl_wrapper.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lbl_wrapper"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lbl_wrapper.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lbl_wrapper.dir/build: lbl_wrapper

.PHONY : CMakeFiles/lbl_wrapper.dir/build

CMakeFiles/lbl_wrapper.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lbl_wrapper.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lbl_wrapper.dir/clean

CMakeFiles/lbl_wrapper.dir/depend:
	cd /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/cmake-build-debug /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/cmake-build-debug /Users/apple/Desktop/profile/easy_cube/rest_server/cube_solver/lbl_wrapper/cmake-build-debug/CMakeFiles/lbl_wrapper.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lbl_wrapper.dir/depend

