cmake_minimum_required(VERSION 3.15...4.0)

set(DRIVER_NAME "HandTrackCamVR")

project(HandCameraDriver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(OPENVR_DIR "${REPO_ROOT}/third_party/openvr" CACHE PATH "Path to the OpenVR SDK")

if(NOT EXISTS "${OPENVR_DIR}/headers/openvr_driver.h")
	message(FATAL_ERROR
		"OpenVR SDK headers were not found at:\n"
		"  ${OPENVR_DIR}\n\n"
		"The SDK is vendored as a git submodule. Fetch it with:\n"
		"  git submodule update --init --recursive\n\n"
		"Or point CMake at an existing checkout:\n"
		"  cmake -DOPENVR_DIR=/path/to/openvr ..")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(PLATFORM 64)
else()
	set(PLATFORM 32)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
	add_compile_definitions(LINUX POSIX)
	set(ARCH_TARGET "linux${PLATFORM}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
	add_compile_definitions(OSX POSIX)
	set(ARCH_TARGET "osx32")
elseif(WIN32)
	add_compile_definitions(_WIN32)
	set(ARCH_TARGET "win${PLATFORM}")
	link_libraries(ws2_32)
else()
	message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

find_library(OPENVR_LIBRARIES
	NAMES openvr_api
	PATHS "${OPENVR_DIR}/lib"
	PATH_SUFFIXES win${PLATFORM} Win${PLATFORM} linux${PLATFORM} osx${PLATFORM}
	NO_DEFAULT_PATH
)

if(NOT OPENVR_LIBRARIES)
	message(FATAL_ERROR
		"Could not find the openvr_api import library under:\n"
		"  ${OPENVR_DIR}/lib")
endif()

message(STATUS "OpenVR SDK: ${OPENVR_DIR}")
message(STATUS "Target arch: ${ARCH_TARGET}")

message(STATUS "Excluding DriverMain.cpp: legacy prototype, duplicates HmdDriverFactory and binds port 65432 a second time")

add_library(driver_${DRIVER_NAME} SHARED
	hmd_driver_factory.cpp
	device_provider.cpp
	controller_device_driver.cpp
	hand_tracking_listener.cpp
	"${OPENVR_DIR}/samples/drivers/utils/driverlog/driverlog.cpp"
)

target_include_directories(driver_${DRIVER_NAME} PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${OPENVR_DIR}/headers"
	"${OPENVR_DIR}/samples/drivers/utils/driverlog"
	"${OPENVR_DIR}/samples/drivers/utils/vrmath"
)

target_link_libraries(driver_${DRIVER_NAME} PRIVATE "${OPENVR_LIBRARIES}")

set(DRIVER_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../bin/${ARCH_TARGET}")

set_target_properties(driver_${DRIVER_NAME} PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${DRIVER_OUTPUT_DIR}"
	PREFIX ""
)

foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
	set_target_properties(driver_${DRIVER_NAME} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY_${_config} "${DRIVER_OUTPUT_DIR}"
	)
endforeach()