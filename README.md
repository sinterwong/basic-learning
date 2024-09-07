# basic-learning

## 项目介绍

本项目旨在提供一个便捷的 C++ 学习环境，涵盖语法、数据结构、设计模式、并发等方面的代码练习与实现。

## 特性

- 模块化设计，方便学习和使用不同主题的代码示例。
- 使用 Conan 进行依赖管理，简化构建过程。
- 支持交叉编译，方便在不同平台上进行开发和测试。
- 集成单元测试和性能分析工具，帮助提高代码质量和效率。

## 开发环境

- **操作系统:** Linux, Windows
- **编译器:** GCC 11+（即将适配MSVC v143）
- **构建工具:** CMake 3.15+
- **包管理器:** Conan 2.3.0+
- **IDE:** VSCode

## 项目结构

```
basic-learning/
├── src/
│   ├── module1/
│   │   └── main.cc
│   └── module2/
│       └── main.cc
├── cmake/ 
├── scripts/
├── platform/
├── tests/ 
├── build/
└── README.md
```

## 构建与运行

### 使用 Conan 构建

1.  安装 Conan: 可以使用 Miniconda 安装 Conan，并参考配置 `settings.json` 文件。
2.  使用 CMake 工具构建项目: 项目配置好后，可直接使用 VSCode 的 CMake Tools 插件进行构建。

### 手动管理依赖（Optional）

1.  将手动编译安装的依赖库软链接到以下目录:  
    `/repo/3rdparty/target/${TARGET_OS}_${TARGET_ARCH}`  
    例如: `/repo/3rdparty/target/Linux_x86_64/opencv`
2.  在 `/repo/load_3rdparty.cmake` 文件中管理依赖库的加载。

### 运行程序

构建完成后，可执行文件位于 `build/${arch}/${module}` 目录下，例如:

```bash
./build/x86_64/module1/module1
```

## 未来计划

- [ ]  gtest 单元测试框架


