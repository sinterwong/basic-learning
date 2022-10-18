CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
SET(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#--------------------------------------------
# Project setting
#--------------------------------------------
# compiler configuration
INCLUDE(cmake/messagecolor.cmake)
INCLUDE(cmake/utils.cmake)
BASIC_APP_EXTRACT_VERSION()

PROJECT(BasicLearning VERSION ${BASIC_APP_VERSION} LANGUAGES CXX)

OPTION(BUILD_TESTS "Build with tests" ON)

MESSAGE(INFO "--------------------------------")
MESSAGE(STATUS "Build BasicLearning: ${BASIC_APP_VERSION}")
MESSAGE(STATUS "Build with tests: ${BUILD_TESTS}")
MESSAGE(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
IF (NOT CMAKE_TOOLCHAIN_FILE)
    MESSAGE(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
ENDIF()


#--------------------------------------------
# Set default build to debug
#--------------------------------------------
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose Release or Debug" FORCE)
ENDIF()

IF(NOT TARGET_OS)
    SET(TARGET_OS linux)
ENDIF()

IF(NOT TARGET_ARCH)
    SET(TARGET_ARCH x86_64)
ENDIF()

IF(NOT TARGET_HARDWARE)
    SET(TARGET_HARDWARE none)  # [none/X3/Jetson]
ENDIF()

MESSAGE(STATUS "Your target os : ${TARGET_OS}")
MESSAGE(STATUS "Your target arch : ${TARGET_ARCH}")
MESSAGE(STATUS "Your target hardware : ${TARGET_HARDWARE}")

MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
#--------------------------------------------
# Set compilation options
#--------------------------------------------
MESSAGE(INFO "--------------------------------")
INCLUDE(cmake/toolchain_options.cmake)
INCLUDE(cmake/load_3rdparty.cmake)
MESSAGE(INFO "Loading 3rdparty libraries")

IF(TARGET_HARDWARE STREQUAL "X3")
    MESSAGE(STATUS "Loading X3 dependent libraries")
    LOAD_X3()
ELSEIF(TARGET_HARDWARE EQUAL "Jetson")
    MESSAGE(STATUS "Loading Jetson dependent libraries")
    LOAD_Jetson()
ELSE()
    MESSAGE(STATUS "No TARGET_HARDWARE specified!")
ENDIF()
MESSAGE(INFO "--------------------------------")

SET(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${TARGET_ARCH})
SET(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

MESSAGE("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
MESSAGE("-- output path:  ${PROJECT_OUTPUT_DIR}")

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

MESSAGE(INFO "3rdparty libraries has loaded.")
#--------------------------------------------
# Build 3rdparty
#--------------------------------------------
MESSAGE(INFO "--------------------------------")
#--------------------------------------------
ADD_SUBDIRECTORY(3rdparty)

# Build application
#--------------------------------------------
MESSAGE(INFO "--------------------------------")
ADD_SUBDIRECTORY(src)

IF(BUILD_TESTS)
    MESSAGE(INFO "--------------------------------")
    ADD_SUBDIRECTORY(tests)
ENDIF()
