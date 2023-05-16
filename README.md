# basic-learning

## 项目介绍
主要为了便捷地进行cpp的学习，项目中将会包含语法、数据结构、设计模式、并发等代码的练习与实现。

## 主要项目结构
- src: 代码实现（包含单元测试，只要文件以.cc结尾就可以被编译成可执行程序并存储到build/${arch}/同名目录）
- cmake: 环境构建相关文件，包含第三方库加载等cmake文件。
- scripts: 脚本文件
- platform: 交叉编译环境构建相关cmake文件
- tests: 测试代码

## 项目构建
- 本地构建
```shell
mkdir build && cd build
cmake ..
make
```
- 交叉编译构建（参考scripts中的build_x3.sh）
```shell
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../platform/toolchain/arm-linux-gnueabihf.cmake ..
make
```

## 项目运行
```shell
./build/${arch}/${module}/executable
```

## TODO
- [ ] gtest单元测试
- [ ] gperftools性能分析


## 参考
- [CMakeCppProjectTemplate](https://github.com/yicm/CMakeCppProjectTemplate)
- [C++并发编程实战](https://book.douban.com/subject/35653912/)
- [Effective Modern C++](https://book.douban.com/subject/25923597/)
- [慕课网部分课程](https://www.imooc.com/)