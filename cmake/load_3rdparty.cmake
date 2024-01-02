# Once done, this will define
#  ......

SET(3RDPARTY_ROOT ${PROJECT_SOURCE_DIR}/3rdparty)
SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${TARGET_OS}_${TARGET_ARCH})
MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")

MACRO(LOAD_SPDLOG)
    SET(SPDLOG_HOME ${3RDPARTY_DIR}/spdlog)
    SET(SPDLOG_LIBRARY_DIR ${SPDLOG_HOME}/lib)
    LIST(APPEND CMAKE_PREFIX_PATH ${SPDLOG_LIBRARY_DIR}/cmake)
    FIND_PACKAGE(spdlog CONFIG REQUIRED)
ENDMACRO()

MACRO(LOAD_GFLAGS)
    SET(GFLAGS_HOME ${3RDPARTY_DIR}/gflags)
    SET(GFLAGS_LIBRARY_DIR ${GFLAGS_HOME}/lib)
    LIST(APPEND CMAKE_PREFIX_PATH ${GFLAGS_LIBRARY_DIR}/cmake)
    FIND_PACKAGE(gflags CONFIG REQUIRED)
ENDMACRO()

MACRO(LOAD_X3)
   # define dnn lib path
    SET(DNN_PATH ${3RDPARTY_DIR}/x3_sdk/dnn)
    SET(APPSDK_PATH ${3RDPARTY_DIR}/x3_sdk/appuser)

    SET(DNN_LIB_PATH ${DNN_PATH}/lib)
    SET(APPSDK_LIB_PATH ${APPSDK_PATH}/lib/hbbpu)
    SET(BPU_libs dnn cnn_intf hbrt_bernoulli_aarch64)

    INCLUDE_DIRECTORIES(${DNN_PATH}/include
                        ${APPSDK_PATH}/include)
    LINK_DIRECTORIES(${DNN_LIB_PATH}
                    ${APPSDK_PATH}/lib/hbbpu
                    ${APPSDK_PATH}/lib)
ENDMACRO()

MACRO(LOAD_OPENCV)
    SET(OPENCV_HOME ${3RDPARTY_DIR}/opencv)
    LIST(APPEND CMAKE_PREFIX_PATH ${OPENCV_HOME}/lib/cmake)
    FIND_PACKAGE(OpenCV CONFIG REQUIRED opencv_core opencv_highgui opencv_video opencv_imgcodecs opencv_imgproc)
    IF(OpenCV_INCLUDE_DIRS)
        MESSAGE(STATUS "Opencv library status:")
        MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
        MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "OpenCV not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_CUDA)
    FIND_PACKAGE(CUDA REQUIRED)
    MESSAGE("-- CUDA version: ${CUDA_VERSION} ${CUDA_NVCC_FLAGS}")
ENDMACRO()
