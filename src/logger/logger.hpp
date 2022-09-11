#ifndef _BASIC_CORE_LOGGER_LOGGER_HPP_
#define _BASIC_CORE_LOGGER_LOGGER_HPP_


#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

// You can define SPDLOG_ACTIVE_LEVEL to the desired log level before including "spdlog.h".
// This will turn on/off logging statements at compile time
#include "spdlog/spdlog.h"


// logger setting
#define BASIC_LOGGER_NAME "basic"
#define BASIC_LOGGER_LOGGER_ERROR_FILENAME "logs/basic_error.log"
#define BASIC_LOGGER_LOGGER_TRACE_FILENAME "logs/basic_error.log"
#define BASIC_LOGGER_PATTERN "[%Y-%m-%d %H:%M:%S.%e][%^%l%$][%t][%s:%#] %v"
#define BASIC_LOGGER_ROTATING_MAX_FILE_SIZE (1024*1024)
#define BASIC_LOGGER_ROTATING_MAX_FILE_NUM 5


#define _TRACE 0
#define _DEBUG 1
#define _INFO 2
#define _WARN 3
#define _ERROR 4
#define _CRITI 5
#define _OFF 6

#define BASIC_LOGGER_TRACE(...) BasicLearningLoggerOut(_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define BASIC_LOGGER_DEBUG(...) BasicLearningLoggerOut(_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define BASIC_LOGGER_INFO(...) BasicLearningLoggerOut(_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define BASIC_LOGGER_WARN(...) BasicLearningLoggerOut(_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define BASIC_LOGGER_ERROR(...) BasicLearningLoggerOut(_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define BASIC_LOGGER_CRITICAL(...) BasicLearningLoggerOut(_CRITI, __FILE__, __LINE__, __VA_ARGS__)


#ifdef __cplusplus
    extern "C" {
#endif


void BasicLearningLoggerInit(
        const bool with_color_console,
        const bool with_console,
        const bool with_error,
        const bool with_trace);

void BasicLearningLoggerSetLevel(const int level);

void BasicLearningLoggerSetPattern(const char* format);

void BasicLearningLoggerSetFlushEvery(const int interval);

void BasicLearningLoggerDrop();


#ifdef __cplusplus
    }
#endif

template<typename... T>
void BasicLearningLoggerOut(const int level,
        const char* filename,
        const int line,
        const T &...msg) {
    // Note: sdplog::get is a thread safe function
    std::shared_ptr<spdlog::logger> logger_ptr = spdlog::get(BASIC_LOGGER_NAME);
    if (!logger_ptr) {
        fprintf(stderr, "Failed to get logger, Please init logger firstly.\n");
    }
    logger_ptr.get()->log(spdlog::source_loc{filename, line, SPDLOG_FUNCTION}, static_cast<spdlog::level::level_enum>(level), msg...);
}

#endif
