if(apple)
# cmake 3.12 is required to get openmp working with clang
cmake_minimum_required(VERSION 3.12)
else(apple)
# Version 3.6 required for list(FILTER)
cmake_minimum_required(VERSION 3.6)
endif(apple)


set(CMAKE_INSTALL_MESSAGE LAZY)

project(dragonfly LANGUAGES C)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX $ENV{HOME}/.local CACHE PATH "Default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")
if(CMAKE_COMPILER_IS_GNUCC)
	if(CMAKE_C_COMPILER_VERSION VERSION_GREATER 4.4)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result")
		#endif(CMAKE_CXX_COMPILER_VERSION)
	endif()
endif(CMAKE_COMPILER_IS_GNUCC)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

set(USE_HDF5 ON CACHE BOOL "If ON, build and support HDF5 input files")
if(USE_HDF5)
	#set(HDF5_USE_STATIC_LIBRARIES TRUE)
	set(HDF5_PREFER_PARALLEL TRUE)
	find_package(HDF5)
	if(HDF5_FOUND)
		add_definitions(-DWITH_HDF5)
		include_directories(${HDF5_INCLUDE_DIR})
		message(STATUS "libhdf5 found. Supporting HDF5 input files")
	else(HDF5_FOUND)
		message(STATUS "libhdf5 not found. Compiling without HDF5 support")
	endif(HDF5_FOUND)
endif(USE_HDF5)

# Compile object files
file(GLOB OBJSOURCE "src/*.c")
list(FILTER OBJSOURCE EXCLUDE REGEX ".*emc.c")
foreach(src ${OBJSOURCE})
	string(REGEX REPLACE ".*/src/(.*).c" "\\1.o" obj ${src})
	add_library(${obj} OBJECT ${src})
	list(APPEND OBJECTS $<TARGET_OBJECTS:${obj}>)
endforeach(src)

# Compile emc program
file(GLOB EMCSOURCE "src/*emc.c")
add_executable(emc ${EMCSOURCE} ${OBJECTS})
target_link_libraries(emc GSL::gsl GSL::gslcblas)
target_link_libraries(emc m)
if(HDF5_FOUND)
	target_link_libraries(emc ${HDF5_C_LIBRARIES})
endif()
target_link_libraries(emc ${MPI_LIBRARIES} ${OpenMP_C_LIBRARIES})
if (MPI_COMPILE_FLAGS)
	set_target_properties(emc PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if (MPI_LINK_FLAGS)
	set_target_properties(emc PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
set_target_properties(emc PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Install emc, aux and dragonfly_init
install(TARGETS emc DESTINATION dragonfly/bin)
install(PROGRAMS dragonfly_init DESTINATION dragonfly/bin)
install(DIRECTORY aux DESTINATION dragonfly)
install(FILES config.ini DESTINATION dragonfly)

# Compile and install utilities
add_subdirectory(utils)

