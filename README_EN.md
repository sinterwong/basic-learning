# basic-learning

<p align="center">
  <img src="assets/logo.png" alt="basic-learning Logo" width="300"> <br/>
</p>

<p align="center">
  A convenient C++ learning environment covering syntax, data structures, design patterns, concurrency, and more.
</p>

---
[English](README_EN.md) | [简体中文](README.md)
---

## Features ✨

- **Modular Design:** Easily learn and use code examples from different topics.
- **Dependency Management:** Uses Conan for simplified building.
- **Cross-Compilation Support:** Develop and test on various platforms.
- **Quality Assurance:** Integrated unit testing and performance analysis tools.


## Build 🚀

### Build with Conan

1. **Install Conan:** Use Miniconda (recommended) and configure the `settings.json` file.
2. **Build with CMake:**  Utilize VSCode's CMake Tools extension for a seamless build process.


### Manual Dependency Management (Optional)

1. **Link Libraries:** Symbolically link manually compiled libraries to `/repo/3rdparty/target/${TARGET_OS}_${TARGET_ARCH}` (e.g., `/repo/3rdparty/target/Linux_x86_64/opencv`).
2. **Manage Dependencies:** Use `/repo/load_3rdparty.cmake` to manage the loading of your libraries.


### Running the Program 🏃

After building, executables are located in `build/${arch}/${module}` (e.g., `./build/x86_64/module1/module1`).


## Project Structure 🏗️

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



## Environment 🛠️

- **OS:** Linux, Windows
- **Compiler:** GCC 11+ (MSVC v143 support coming soon)
- **Build System:** CMake 3.15+
- **Package Manager:** Conan 2.3.0+
- **IDE:** VSCode


## Roadmap 🗺️

- [x] Support MSVC
- [ ] Build CI pipeline

---

<p align="center">
  Happy coding! 😊
</p>