CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

LOAD_SPDLOG()
LOAD_GFLAGS()

AUX_SOURCE_DIRECTORY(. CURRENT_DIR_SRCS)
ADD_LIBRARY(module_logger ${CURRENT_DIR_SRCS})
FILE(GLOB_RECURSE CURRENT_DIR_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

TARGET_INCLUDE_DIRECTORIES(module_logger PUBLIC ${3RDPARTY_DIR}/spdlog/include)
TARGET_LINK_LIBRARIES(module_logger spdlog::spdlog)

FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/logger)

FOREACH(include ${CURRENT_DIR_HEAD})
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/logger COPYONLY)
ENDFOREACH()

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/logger
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "CMakeLists.txt" EXCLUDE
)

INSTALL(TARGETS module_logger
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

LINK_LIBRARIES(
    gflags
)

FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/logger)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/logger)
FILE(GLOB APP_SOURCES *.cc)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cc" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    TARGET_LINK_LIBRARIES(${testname} module_logger)
ENDFOREACH(sourcefile ${APP_SOURCES})