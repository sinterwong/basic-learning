# Specialized libraries can be compiled separately, soft-linked to the 3RDPARTY_DIR, and then handled independently.
SET(3RDPARTY_ROOT ${PROJECT_SOURCE_DIR}/3rdparty)
SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${TARGET_OS}_${TARGET_ARCH})
MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")

MACRO(LOAD_SPDLOG)
    ADD_DEFINITIONS(-DSPDLOG_USE_STD_FORMAT)
    FIND_PACKAGE(spdlog REQUIRED)
ENDMACRO()

MACRO(LOAD_GFLAGS)
    # SET(GFLAGS_HOME ${3RDPARTY_DIR}/gflags)
    # LIST(APPEND CMAKE_PREFIX_PATH ${GFLAGS_HOME}/lib/cmake)
    # SET(GFLAGS_LIB_DIR ${GFLAGS_HOME}/lib)
    FIND_PACKAGE(gflags REQUIRED)
    SET(GFLAGS_LIB_DIR ${gflags_LIB_DIRS_RELEASE})
    SET(GFLAGS_LIBS ${gflags_LIBS_RELEASE})
ENDMACRO()

MACRO(LOAD_TASKFLOW)
    FIND_PACKAGE(Taskflow REQUIRED)
ENDMACRO()

MACRO(LOAD_EIGEN)
    FIND_PACKAGE(Eigen3 REQUIRED)
ENDMACRO()

MACRO(LOAD_CURL)
    FIND_PACKAGE(CURL REQUIRED)
ENDMACRO()

MACRO(LOAD_OPENCV)
    SET(OPENCV_HOME ${3RDPARTY_DIR}/opencv)
    SET(OPENCV_LIB_DIR ${OPENCV_HOME}/lib)
    LIST(APPEND CMAKE_PREFIX_PATH ${OPENCV_HOME}/lib/cmake)
    FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui video videoio)

    IF(OpenCV_INCLUDE_DIRS)
        MESSAGE(STATUS "Opencv library status:")
        MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
        MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "OpenCV not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_ONNXRUNTIME)
    FIND_FILE(ONNXRUNTIME_INCLUDE_DIR include ${3RDPARTY_DIR}/onnxruntime NO_DEFAULT_PATH)
    FIND_FILE(ONNXRUNTIME_LIBRARY_DIR lib ${3RDPARTY_DIR}/onnxruntime NO_DEFAULT_PATH)
    SET(ONNXRUNTIME_LIBS
        onnxruntime
    )

    IF(ONNXRUNTIME_INCLUDE_DIR)
        MESSAGE(STATUS "ONNXRUNTIME_INCLUDE_DIR : ${ONNXRUNTIME_INCLUDE_DIR}")
        MESSAGE(STATUS "ONNXRUNTIME_LIBRARY_DIR : ${ONNXRUNTIME_LIBRARY_DIR}")
        MESSAGE(STATUS "ONNXRUNTIME_LIBS : ${ONNXRUNTIME_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "ONNXRUNTIME_LIBS not found!")
    ENDIF()
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

MACRO(LOAD_CUDA)
    FIND_PACKAGE(CUDA REQUIRED)
    MESSAGE("-- CUDA version: ${CUDA_VERSION} ${CUDA_NVCC_FLAGS}")
ENDMACRO()
