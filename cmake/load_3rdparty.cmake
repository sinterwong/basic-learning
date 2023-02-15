# Once done, this will define
#  ......

SET(3RDPARTY_ROOT ${PROJECT_SOURCE_DIR}/3rdparty)
SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${TARGET_OS}_${TARGET_ARCH}_${TARGET_HARDWARE})
MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")

MACRO(LOAD_X3)
   # define dnn lib path
    SET(DNN_PATH "~/.horizon/ddk/xj3_aarch64/dnn/")
    SET(APPSDK_PATH "~/.horizon/ddk/xj3_aarch64/appsdk/appuser/")

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
    SET(OPENCV_HOME /home/wangxt/workspace/softwares/opencv-4.5)
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