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
#define BASIC_VER_MAJOR 0
#define BASIC_VER_MINOR 1
#define BASIC_VER_PATCH 0

#define BASIC_VERSION (BASIC_VER_MAJOR * 10000 + BASIC_VER_MINOR * 100 + BASIC_VER_PATCH)

// for source code
#define _BASIC_STR(s) #s
#define BASIC_PROJECT_VERSION(major, minor, patch) "v" _BASIC_STR(major.minor.patch)
