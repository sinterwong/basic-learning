#ifndef _FLOWENGINE_CORE_LOGGER_LOGGER_HPP_
#define _FLOWENGINE_CORE_LOGGER_LOGGER_HPP_


#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

// You can define SPDLOG_ACTIVE_LEVEL to the desired log level before including "spdlog.h".
// This will turn on/off logging statements at compile time
#include "spdlog/spdlog.h"


// logger setting
#define FLOWENGINE_LOGGER_NAME "flowengine"
#define FLOWENGINE_LOGGER_LOGGER_ERROR_FILENAME "logs/flowengine_error.log"
#define FLOWENGINE_LOGGER_LOGGER_TRACE_FILENAME "logs/flowengine_error.log"
#define FLOWENGINE_LOGGER_PATTERN "[%Y-%m-%d %H:%M:%S.%e][%^%l%$][%t][%s:%#] %v"
#define FLOWENGINE_LOGGER_ROTATING_MAX_FILE_SIZE (1024*1024)
#define FLOWENGINE_LOGGER_ROTATING_MAX_FILE_NUM 5


#define _TRACE 0
#define _DEBUG 1
#define _INFO 2
#define _WARN 3
#define _ERROR 4
#define _CRITI 5
#define _OFF 6

#define FLOWENGINE_LOGGER_TRACE(...) FlowEngineLoggerOut(_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define FLOWENGINE_LOGGER_DEBUG(...) FlowEngineLoggerOut(_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define FLOWENGINE_LOGGER_INFO(...) FlowEngineLoggerOut(_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define FLOWENGINE_LOGGER_WARN(...) FlowEngineLoggerOut(_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define FLOWENGINE_LOGGER_ERROR(...) FlowEngineLoggerOut(_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define FLOWENGINE_LOGGER_CRITICAL(...) FlowEngineLoggerOut(_CRITI, __FILE__, __LINE__, __VA_ARGS__)


#ifdef __cplusplus
    extern "C" {
#endif


void FlowEngineLoggerInit(
        const bool with_color_console,
        const bool with_console,
        const bool with_error,
        const bool with_trace);

void FlowEngineLoggerSetLevel(const int level);

void FlowEngineLoggerSetPattern(const char* format);

void FlowEngineLoggerSetFlushEvery(const int interval);

void FlowEngineLoggerDrop();


#ifdef __cplusplus
    }
#endif

template<typename... T>
void FlowEngineLoggerOut(const int level,
        const char* filename,
        const int line,
        const T &...msg) {
    // Note: sdplog::get is a thread safe function
    std::shared_ptr<spdlog::logger> logger_ptr = spdlog::get(FLOWENGINE_LOGGER_NAME);
    if (!logger_ptr) {
        fprintf(stderr, "Failed to get logger, Please init logger firstly.\n");
    }
    logger_ptr.get()->log(spdlog::source_loc{filename, line,
            SPDLOG_FUNCTION},
            static_cast<spdlog::level::level_enum>(level),
            msg...);
}


#endif
