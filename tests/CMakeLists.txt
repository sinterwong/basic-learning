CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Tests)

INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
)

# LINK_DIRECTORIES(
#     ${COMMON_LIBRARY_DIR}
# )

LINK_LIBRARIES(
    gflags::gflags
    pthread
    module_logger
)

FILE(GLOB APP_SOURCES *.cpp)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cpp" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/tests)

    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(sourcefile ${APP_SOURCES})
