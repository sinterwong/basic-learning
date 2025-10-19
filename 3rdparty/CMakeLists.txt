CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(AICore3rdparty)

SET(BUILD_TESTING OFF CACHE BOOL "Disable testing for 3rdparty libs")
SET(WITH_GTEST OFF CACHE BOOL "Disable gtest for glog")

IF(NOT TARGET logger)
    MESSAGE(STATUS "logger target not found. Adding subdirectory...")
    ADD_SUBDIRECTORY(logger EXCLUDE_FROM_ALL)
ELSE()
    MESSAGE(STATUS "logger target already exists. Skipping.")
ENDIF()

FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
