#pragma once

// for cmake
#define BASIC_VER_MAJOR 1
#define BASIC_VER_MINOR 2
#define BASIC_VER_PATCH 0

#define BASIC_APP_VERSION                                                      \
  (BASIC_VER_MAJOR * 10000 + BASIC_VER_MINOR * 100 + BASIC_VER_PATCH)

// for source code
#define _BASIC_APP_STR(s) #s
#define BASIC_APP_PROJECT_VERSION(major, minor, patch)                         \
  "v" _BASIC_APP_STR(major.minor.patch)

#define BASIC_APP_VERSION_STR                                                  \
  BASIC_APP_PROJECT_VERSION(BASIC_APP_VER_MAJOR, BASIC_APP_VER_MINOR,          \
                            BASIC_APP_VER_PATCH)
