CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

LOAD_PCL()

AUX_SOURCE_DIRECTORY(. CURRENT_DIR_SRCS)
ADD_LIBRARY(point_cloud ${CURRENT_DIR_SRCS})
TARGET_INCLUDE_DIRECTORIES(point_cloud PRIVATE ${PROJECT_INCLUDE_DIR} ${CMAKE_INCLUDE_PATH})

FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/point_cloud)
FILE(GLOB_RECURSE CURRENT_DIR_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
FOREACH(include ${CURRENT_DIR_HEAD})
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/point_cloud COPYONLY)
ENDFOREACH()

LINK_DIRECTORIES(${CMAKE_LIBRARY_PATH})

FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/point_cloud)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/point_cloud)
FILE(GLOB_RECURSE APP_SOURCES *.cc)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cc" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    TARGET_INCLUDE_DIRECTORIES(${testname} PRIVATE ${PROJECT_INCLUDE_DIR} ${CMAKE_INCLUDE_PATH})
    TARGET_LINK_LIBRARIES(${testname} PRIVATE point_cloud pcl_filters pcl_io pcl_io_ply pcl_common boost_filesystem)
ENDFOREACH(sourcefile ${APP_SOURCES})
