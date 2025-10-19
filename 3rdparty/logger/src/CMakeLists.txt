CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(logger VERSION 1.0.0)

INCLUDE(GNUInstallDirs)
INCLUDE(CMakePackageConfigHelpers)

LOAD_GFLAGS()
LOAD_GLOG()

SET(CMAKE_POSITION_INDEPENDENT_CODE ON) 

SET(DEPENDENCY_LIBS
        glog::glog
)

FILE(GLOB CURRENT_DIR_PUBLIC_HEAD RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} logger.hpp)
FILE(GLOB CURRENT_DIR_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} logger.cpp)

# No longer need to manually copy headers

ADD_LIBRARY(logger SHARED
    ${CURRENT_DIR_SRCS}
    ${CURRENT_DIR_PUBLIC_HEAD}
)
ADD_LIBRARY(logger::logger ALIAS logger) # Alias for modern CMake usage

TARGET_INCLUDE_DIRECTORIES(logger
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> # For build tree
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/logger> # For install tree, headers will be in 'logger' subdir
    PRIVATE
        ${PROJECT_INCLUDE_DIR} # If there are any truly private internal includes from root
)
TARGET_LINK_LIBRARIES(logger PRIVATE ${DEPENDENCY_LIBS})

# Install the target and its public headers
INSTALL(TARGETS logger
    EXPORT logger_export_set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT bin
)

# Install the public header files explicitly for the package
INSTALL(FILES ${CURRENT_DIR_PUBLIC_HEAD}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/logger # Keep them in an 'logger' subdirectory for namespacing
    COMPONENT dev
)

# Generate the logger-config.cmake and logger-config-version.cmake files
WRITE_BASIC_PACKAGE_VERSION_FILE(
    "${CMAKE_CURRENT_BINARY_DIR}/logger-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

EXPORT(EXPORT logger_export_set
    FILE "${CMAKE_CURRENT_BINARY_DIR}/logger-targets.cmake"
    NAMESPACE logger::
)

CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/logger-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/logger-config.cmake"
    @ONLY
)

INSTALL(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/logger-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/logger-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/logger
    COMPONENT dev
)

INSTALL(EXPORT logger_export_set
    FILE logger-targets.cmake
    NAMESPACE logger::
    DESTINATION ${CMAKE_INSTALL_DATADIR}/logger
    COMPONENT dev
)