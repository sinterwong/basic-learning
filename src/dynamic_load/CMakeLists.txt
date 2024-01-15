CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/dynamic_load)

LINK_LIBRARIES(dl)

FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/dynamic_load)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/dynamic_load)
FILE(GLOB APP_SOURCES *.cc)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cc" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
ENDFOREACH(sourcefile ${APP_SOURCES})