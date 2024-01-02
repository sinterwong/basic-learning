CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

LOAD_CUDA()

INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

LINK_DIRECTORIES(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

LINK_LIBRARIES(
    pthread
    ${CUDA_cublas_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
)

SET(CUDA_ARCH "-arch=sm_87")

FILE(GLOB_RECURSE CURRENT_DIR_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/cuda_demo)
FOREACH(include ${CURRENT_DIR_HEAD})
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/cuda_demo COPYONLY)
ENDFOREACH()

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/cuda_demo
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "CMakeLists.txt" EXCLUDE
    )

FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/cuda_demo)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/cuda_demo)
FILE(GLOB_RECURSE APP_SOURCES *.cu)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cu" "" testname ${sourcefilewithoutpath})
    CUDA_ADD_EXECUTABLE( ${testname} ${sourcefile})
    TARGET_COMPILE_OPTIONS(${testname} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:
            ${CUDA_ARCH};
            -std=c++17;
            -rdc=true;
            -Xcompiler;
            -pthread
            -lnvshmem 
            -lmpi_cxx 
            -lmpi
        >
    )
ENDFOREACH(sourcefile ${APP_SOURCES})
