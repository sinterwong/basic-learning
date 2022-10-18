MESSAGE(STATUS "Configure Cross Compiler")

SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)
SET(TARGET_OS linux)
SET(TARGET_ARCH aarch64)
SET(TARGET_HARDWARE X3)

# set ${CMAKE_C_FLAGS} and ${CMAKE_CXX_FLAGS}flag for cross-compiled process
SET(CROSS_COMPILATION_ARM x3)
SET(CROSS_COMPILATION_ARCHITECTURE aarch64)

SET(CMAKE_C_COMPILER       ${TOOLCHAIN_ROOTDIR}/bin/aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER     ${TOOLCHAIN_ROOTDIR}/bin/aarch64-linux-gnu-g++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_C_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE " -Wall -Werror -O3 ")
set(CMAKE_C_FLAGS_RELEASE " -Wall -Werror -O3 ")

# set searching rules for cross-compiler
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

SET(CMAKE_SKIP_BUILD_RPATH TRUE)
SET(CMAKE_SKIP_RPATH TRUE)

# set g++ param
# -fopenmp link libgomp
# SET(CMAKE_CXX_FLAGS "-std=c++17 -march=armv8-a -mfloat-abi=softfp -mfpu=neon-vfpv4 \
#     -ffunction-sections \
#     -fdata-sections -O2 -fstack-protector-strong -lm -ldl -lstdc++\
#     ${CMAKE_CXX_FLAGS}")
