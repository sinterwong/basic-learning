/**
 * @file version.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-04-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

// for cmake
#define FLOWENGINE_VER_MAJOR 0
#define FLOWENGINE_VER_MINOR 1
#define FLOWENGINE_VER_PATCH 0

#define FLOWENGINE_VERSION (FLOWENGINE_VER_MAJOR * 10000 + FLOWENGINE_VER_MINOR * 100 + FLOWENGINE_VER_PATCH)

// for source code
#define _FLOWENGINE_STR(s) #s
#define FLOWENGINE_PROJECT_VERSION(major, minor, patch) "v" _FLOWENGINE_STR(major.minor.patch)
