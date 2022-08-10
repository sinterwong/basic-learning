# rm -rf build
# mkdir build
cd build

cmake -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64.x3.cmake \
      -DTOOLCHAIN_ROOTDIR=/opt/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu .. 
