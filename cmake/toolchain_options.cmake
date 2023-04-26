IF(CMAKE_VERSION VERSION_LESS "3.10")
    IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    ENDIF()
ELSE()
    SET(CMAKE_CXX_STANDARD 17)
ENDIF()
