# Once done, this will define
#  ......

SET(3RDPARTY_ROOT ${PROJECT_SOURCE_DIR}/3rdparty)
SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${TARGET_OS}_${TARGET_ARCH})
MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")

MACRO(LOAD_GLOG)
    FIND_FILE(GLOG_INCLUDE_DIR include ${3RDPARTY_DIR}/glog NO_DEFAULT_PATH)
    FIND_FILE(GLOG_LIBRARY_DIR lib ${3RDPARTY_DIR}/glog NO_DEFAULT_PATH)

    SET(GLOG_LIBS
        glog
    )
    IF(GLOG_INCLUDE_DIR)
        MESSAGE(STATUS "GLOG_INCLUDE_DIR : ${GLOG_INCLUDE_DIR}")
        MESSAGE(STATUS "GLOG_LIBRARY_DIR : ${GLOG_LIBRARY_DIR}")
        MESSAGE(STATUS "GLOG_LIBS : ${GLOG_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "GLOG_LIBS not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_BOOST)
    SET(BOOST_HOME ${3RDPARTY_DIR}/boost)
    LIST(APPEND CMAKE_PREFIX_PATH ${BOOST_HOME}/lib/cmake)
    FIND_PACKAGE(Boost REQUIRED COMPONENTS wserialization wave unit_test_framework type_erasure timer thread system serialization regex random program_options prg_exec_monitor nowide math_tr1l math_tr1f math_tr1 math_c99l log_setup log locale json graph filesystem date_time contract context container chrono atomic)
    IF(Boost_INCLUDE_DIRS)
        MESSAGE(STATUS "Boost library status:")
        MESSAGE(STATUS "    include path: ${Boost_INCLUDE_DIRS}")
        MESSAGE(STATUS "    libraries path: ${Boost_LIBRARY_DIRS}")
        MESSAGE(STATUS "    libraries: ${Boost_LIBRARIES}")
        MESSAGE( STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")
    ELSE()
        MESSAGE(FATAL_ERROR "Boost not found!")
        ENDIF()
ENDMACRO()

MACRO(LOAD_OPENCV)
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
    SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/src/module/jetson_utils/cuda" )
    FIND_PACKAGE(CUDA)
    MESSAGE("-- CUDA version: ${CUDA_VERSION}")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

    IF(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        MESSAGE("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
        SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

        IF(CUDA_VERSION_MAJOR GREATER 9)
            MESSAGE("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
            SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
        ENDIF()

        IF(CUDA_VERSION_MAJOR GREATER 10)
            MESSAGE("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
            SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
        ENDIF()
	    LINK_DIRECTORIES(/usr/lib/aarch64-linux-gnu/tegra)
    ENDIF()
ENDMACRO()

MACRO(LOAD_FMT)
    FIND_FILE(FMT_INCLUDE_DIR include ${3RDPARTY_DIR}/fmt NO_DEFAULT_PATH)
    FIND_FILE(FMT_LIBRARY_DIR lib ${3RDPARTY_DIR}/fmt NO_DEFAULT_PATH)
    SET(FMT_LIBS 
        fmt
    )
    IF(FMT_INCLUDE_DIR)
        MESSAGE(STATUS "fmt libraries path: ${FMT_INCLUDE_DIR}")
        MESSAGE(STATUS "fmt libraries path: ${FMT_LIBRARY_DIR}")
        MESSAGE(STATUS "fmt libraries : ${FMT_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "FMT_INCLUDE_DIR not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_FFMPEG)
    FIND_FILE(FFMPEG_INCLUDE_DIR include ${3RDPARTY_DIR}/ffmpeg NO_DEFAULT_PATH)
    FIND_FILE(FFMPEG_LIBRARY_DIR lib ${3RDPARTY_DIR}/ffmpeg NO_DEFAULT_PATH)
    SET(FFMPEG_LIBS 
        avcodec
        avdevice
        avfilter
        avformat
        avutil
        swresample
        swscale
    )
    IF(FFMPEG_INCLUDE_DIR)
        MESSAGE(STATUS "ffmpeg libraries path: ${FFMPEG_INCLUDE_DIR}")
        MESSAGE(STATUS "ffmpeg libraries path: ${FFMPEG_LIBRARY_DIR}")
        MESSAGE(STATUS "ffmpeg libraries : ${FFMPEG_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "FFMPEG_INCLUDE_DIR not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_PROTOBUF)
    SET(PROTOBUF_LIB_ROOT ${3RDPARTY_DIR}/protobuf)
    SET(Protobuf_PREFIX_PATH
        ${PROTOBUF_LIB_ROOT}/include
        ${PROTOBUF_LIB_ROOT}/lib
        ${PROTOBUF_LIB_ROOT}/bin
    )
    LIST(APPEND CMAKE_PREFIX_PATH ${Protobuf_PREFIX_PATH})
    FIND_PACKAGE(Protobuf REQUIRED)
    IF(PROTOBUF_INCLUDE_DIR)
        MESSAGE(STATUS "Protobuf_INCLUDE_DIR : ${Protobuf_INCLUDE_DIR}")
        MESSAGE(STATUS "Protobuf_LIBRARY : ${Protobuf_LIBRARY}")
    ELSE()
        MESSAGE(FATAL_ERROR "Protobuf_LIBRARY not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_FREETYPE)
    FIND_FILE(FREETYPE_INCLUDE_DIR include ${3RDPARTY_DIR}/freetype NO_DEFAULT_PATH)
    FIND_FILE(FREETYPE_LIBRARY_DIR lib ${3RDPARTY_DIR}/freetype NO_DEFAULT_PATH)
    SET(FREETYPE_LIBS
        freetype
    )
    IF(FREETYPE_INCLUDE_DIR)
        MESSAGE(STATUS "FREETYPE_INCLUDE_DIR : ${FREETYPE_INCLUDE_DIR}")
        MESSAGE(STATUS "FREETYPE_LIBRARY_DIR : ${FREETYPE_LIBRARY_DIR}")
        MESSAGE(STATUS "FREETYPE_LIBS : ${FREETYPE_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "FREETYPE_LIBS not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_OPENSSL)
    FIND_FILE(OPENSSL_INCLUDE_DIR include ${3RDPARTY_DIR}/openssl NO_DEFAULT_PATH)
    FIND_FILE(OPENSSL_LIBRARY_DIR lib ${3RDPARTY_DIR}/openssl NO_DEFAULT_PATH)
    SET(OPENSSL_LIBS
        ssl
        crypto
    )
    IF(OPENSSL_INCLUDE_DIR)
        MESSAGE(STATUS "OPENSSL_INCLUDE_DIR : ${OPENSSL_INCLUDE_DIR}")
        MESSAGE(STATUS "OPENSSL_LIBRARY_DIR : ${OPENSSL_LIBRARY_DIR}")
        MESSAGE(STATUS "OPENSSL_LIBS : ${OPENSSL_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "OPENSSL_LIBS not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_CURL)
    # FIND_FILE(CURL_INCLUDE_DIR include ${3RDPARTY_DIR}/curl NO_DEFAULT_PATH)
    # FIND_FILE(CURL_LIBRARY_DIR lib ${3RDPARTY_DIR}/curl NO_DEFAULT_PATH)
    # SET(CURL_LIBS
    #     curl
    # )
    # IF(CURL_INCLUDE_DIR)
    #     MESSAGE(STATUS "CURL_INCLUDE_DIR : ${CURL_INCLUDE_DIR}")
    #     MESSAGE(STATUS "CURL_LIBRARY_DIR : ${CURL_LIBRARY_DIR}")
    #     MESSAGE(STATUS "CURL_LIBS : ${CURL_LIBS}")
    # ELSE()
    #     MESSAGE(FATAL_ERROR "CURL_LIBS not found!")
    # ENDIF()
    SET(CURL_LIB_ROOT ${3RDPARTY_DIR}/curl)
    SET(CURL_PREFIX_PATH
        ${CURL_LIB_ROOT}/include
        ${CURL_LIB_ROOT}/lib
        ${CURL_LIB_ROOT}/bin
    )
    LIST(APPEND CMAKE_PREFIX_PATH ${CURL_PREFIX_PATH})
    FIND_PACKAGE(CURL REQUIRED)
    IF(CURL_FOUND)
        SET(requiredlibs ${requiredlibs} ${CURL_LIBRARIES} )
        MESSAGE(STATUS "CURL_INCLUDE_DIRS : ${CURL_INCLUDE_DIRS}")
        MESSAGE(STATUS "CURL_LIBRARIES : ${CURL_LIBRARIES}")
    ELSE(CURL_FOUND)
        MESSAGE(FATAL_ERROR "Could not find the CURL library and development files.")
    ENDIF(CURL_FOUND)
ENDMACRO()

MACRO(LOAD_MNN)
    FIND_FILE(MNN_INCLUDE_DIR include ${3RDPARTY_DIR}/mnn NO_DEFAULT_PATH)
    FIND_FILE(MNN_LIBRARY_DIR lib ${3RDPARTY_DIR}/mnn NO_DEFAULT_PATH)
    SET(MNN_LIBS
        MNN
    )
    IF(MNN_INCLUDE_DIR)
        MESSAGE(STATUS "MNN_INCLUDE_DIR : ${MNN_INCLUDE_DIR}")
        MESSAGE(STATUS "MNN_LIBRARY_DIR : ${MNN_LIBRARY_DIR}")
        MESSAGE(STATUS "MNN_LIBS : ${MNN_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "MNN_LIBS not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_TENGINE)
    FIND_FILE(TENGINE_INCLUDE_DIR include ${3RDPARTY_DIR}/tengine NO_DEFAULT_PATH)
    FIND_FILE(TENGINE_LIBRARY_DIR lib ${3RDPARTY_DIR}/tengine NO_DEFAULT_PATH)
    SET(TENGINE_LIBS
        tengine
    )
    IF(TENGINE_INCLUDE_DIR)
        MESSAGE(STATUS "TENGINE_INCLUDE_DIR : ${TENGINE_INCLUDE_DIR}")
        MESSAGE(STATUS "TENGINE_LIBRARY_DIR : ${TENGINE_LIBRARY_DIR}")
        MESSAGE(STATUS "TENGINE_LIBS : ${TENGINE_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "TENGINE_LIBS not found!")
    ENDIF()
ENDMACRO()
