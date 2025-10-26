CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

LOAD_GTEST()

FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/leetcode)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/leetcode)
FILE(GLOB_RECURSE APP_SOURCES *.cc)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")

FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cc" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    TARGET_INCLUDE_DIRECTORIES(${testname} PRIVATE ${CMAKE_INCLUDE_PATH})
    TARGET_LINK_LIBRARIES(${testname} PRIVATE gtest::gtest)

    INSTALL(TARGETS ${testname} DESTINATION bin/leetcode)
ENDFOREACH(sourcefile ${APP_SOURCES})
