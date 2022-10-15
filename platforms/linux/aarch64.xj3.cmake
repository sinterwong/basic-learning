MESSAGE(STATUS "Configure Cross Compiler")

SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_C_COMPILER       ${TOOLCHAIN_ROOTDIR}/bin/aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER     ${TOOLCHAIN_ROOTDIR}/bin/aarch64-linux-gnu-g++)

# set searching rules for cross-compiler
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

SET(TARGET_OS linux)
SET(TARGET_ARCH aarch64)

SET(CMAKE_SKIP_BUILD_RPATH TRUE)
SET(CMAKE_SKIP_RPATH TRUE)

# set ${CMAKE_C_FLAGS} and ${CMAKE_CXX_FLAGS}flag for cross-compiled process
SET(CROSS_COMPILATION_ARM xj3)
SET(CROSS_COMPILATION_ARCHITECTURE aarch64)

# set g++ param
# -fopenmp link libgomp
# SET(CMAKE_CXX_FLAGS "-std=c++17 -march=aarch64 -mfloat-abi=softfp -mfpu=neon-vfpv4 \
#     -ffunction-sections \
#     -fdata-sections -O2 -fstack-protector-strong -lm -ldl -lstdc++\
#     ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=c++11 -march=armv8-a -O2 -lstdc++ ${CMAKE_CXX_FLAGS}")
