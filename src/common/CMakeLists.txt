CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

FILE(GLOB_RECURSE CURRENT_DIR_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/common)
FOREACH(include ${CURRENT_DIR_HEAD})
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/common COPYONLY)
ENDFOREACH()

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "CMakeLists.txt" EXCLUDE
    )
