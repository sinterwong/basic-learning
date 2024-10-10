CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

LOAD_TASKFLOW()
LOAD_GFLAGS()
LOAD_CURL()
LOAD_ONNXRUNTIME()
LOAD_OPENCV()

FILE(GLOB_RECURSE CURRENT_DIR_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/use_taskflow)
FOREACH(include ${CURRENT_DIR_HEAD})
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/use_taskflow COPYONLY)
ENDFOREACH()

FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/use_taskflow)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/use_taskflow)
FILE(GLOB_RECURSE APP_SOURCES *.cc)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")

FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cc" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    TARGET_INCLUDE_DIRECTORIES(${testname} PRIVATE ${Taskflow_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${onnxruntime_INCLUDE_DIR} ${CURL_INCLUDE_DIR} ${gflags_INCLUDE_DIR})
    TARGET_LINK_LIBRARIES(${testname} PRIVATE onnxruntime::onnxruntime opencv::opencv Taskflow::Taskflow CURL::libcurl gflags::gflags)

    INSTALL(TARGETS ${testname} DESTINATION bin/use_taskflow)
ENDFOREACH(sourcefile ${APP_SOURCES})
