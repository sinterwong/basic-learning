CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
SET(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Project setting
INCLUDE(cmake/messagecolor.cmake)
INCLUDE(cmake/utils.cmake)
BASIC_APP_EXTRACT_VERSION()

PROJECT(BasicLearning VERSION ${BASIC_APP_VERSION} LANGUAGES CXX)

OPTION(BUILD_TESTS "Build with tests" ON)
OPTION(BUILD_TOOLS "Build with tools" ON)
OPTION(BUILD_WITH_CUDA "Build with tests" OFF)

MESSAGE(INFO "--------------------------------")
MESSAGE(STATUS "Build BasicLearning: ${BASIC_APP_VERSION}")
MESSAGE(STATUS "Build with tests: ${BUILD_TESTS}")
MESSAGE(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

# Include conan
INCLUDE(cmake/conan.cmake)
conan_config_install()

INCLUDE(${CMAKE_BINARY_DIR}/generators/conan_toolchain.cmake)

# Set default build to debug
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose Release, Debug, RelWithDebInfo or MinSizeRel" FORCE)
ENDIF()

IF(NOT TARGET_OS)
    SET(TARGET_OS ${CMAKE_SYSTEM_NAME})
ENDIF()

IF(NOT TARGET_ARCH)
    SET(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
ENDIF()

IF(NOT TARGET_HARDWARE)
    SET(TARGET_HARDWARE generic)
ENDIF()

MESSAGE(STATUS "Your target os : ${TARGET_OS}")
MESSAGE(STATUS "Your target arch : ${TARGET_ARCH}")
MESSAGE(STATUS "Your target hardware : ${TARGET_HARDWARE}")

MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Set compilation options
MESSAGE(INFO "--------------------------------")
INCLUDE(cmake/load_3rdparty.cmake)
MESSAGE(INFO "Loading 3rdparty libraries")

SET(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${TARGET_ARCH})
SET(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

MESSAGE("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
MESSAGE("-- output path:  ${PROJECT_OUTPUT_DIR}")

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

# Build 3rdparty
MESSAGE(INFO "3rdparty libraries has loaded.")
MESSAGE(INFO "--------------------------------")
ADD_SUBDIRECTORY(3rdparty)

# Build application
MESSAGE(INFO "--------------------------------")
ADD_SUBDIRECTORY(src)

IF(BUILD_TESTS)
    MESSAGE(INFO "--------------------------------")
    ADD_SUBDIRECTORY(tests)
ENDIF()

IF(BUILD_TOOLS)
    MESSAGE(INFO "--------------------------------")
    ADD_SUBDIRECTORY(tools)
ENDIF()
