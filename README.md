# basic-learning

<p align="center">
  <img src="assets/logo.png" alt="basic-learning Logo" width="300"> <br/>
</p>

<p align="center">
  一个便捷的 C++ 学习环境，涵盖语法、数据结构、设计模式、并发等方面的内容。
</p>

---
[English](README_EN.md) | [简体中文](README.md)
---

## 功能 ✨

- **模块化设计:** 方便学习和使用不同主题的代码示例。
- **依赖管理:** 使用 Conan 简化构建过程。
- **交叉编译支持:** 方便在不同平台上进行开发和测试。
- **质量保证:** 集成单元测试和性能分析工具。


## 项目构建 🚀

### 使用 Conan 构建

1. **安装 Conan:** 推荐使用 Miniconda 并配置 `settings.json` 文件。
2. **使用 CMake 构建:** 使用 VSCode 的 CMake Tools 插件进行无缝构建。


### 手动管理依赖（Optional）

1. **链接库:** 将手动编译的库文件软链接到 `/repo/3rdparty/target/${TARGET_OS}_${TARGET_ARCH}`  (例如 `/repo/3rdparty/target/Linux_x86_64/opencv`)。
2. **管理依赖:** 使用 `/repo/load_3rdparty.cmake` 文件管理依赖库的加载。


### 运行程序 🏃

构建完成后，可执行文件位于 `build/${arch}/${module}` 目录下 (例如 `./build/x86_64/module1/module1`)。


## 项目结构 🏗️

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


## Env 🛠️

- **操作系统:** Linux, Windows
- **编译器:** GCC 11+ (即将支持 MSVC v143)
- **构建系统:** CMake 3.15+
- **包管理器:** Conan 2.3.0+
- **IDE:** VSCode


## TODO 🗺️

- [x] 支持 MSVC
- [ ] 构建 CI 流水线

---

<p align="center">
  Happy coding! 😊
</p>
