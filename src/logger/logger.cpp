#include "logger.hpp"

#include <iostream>
#include <memory>
#include <vector>

#include "spdlog/sinks/basic_file_sink.h"       // support for basic file logging
#include "spdlog/sinks/rotating_file_sink.h"    // support for rotating file logging
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/stdout_sinks.h"

// creating loggers with multiple sinks
void FlowEngineLoggerInit(
        const bool with_color_console,
        const bool with_console,
        const bool with_error,
        const bool with_trace) {
    std::vector<spdlog::sink_ptr> sinks;

    if (with_color_console) {
        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sink->set_level(spdlog::level::trace);
        //sink->set_pattern(FLOWENGINE_LOGGER_PATTERN);
        sinks.push_back(sink);
    } else if (with_console) {
        auto sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
        sink->set_level(spdlog::level::trace);
        //sink->set_pattern(FLOWENGINE_LOGGER_PATTERN);
        sinks.push_back(sink);
    }

    if (with_error) {
        auto with_error_logger_rotating = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                    FLOWENGINE_LOGGER_LOGGER_ERROR_FILENAME,
                    FLOWENGINE_LOGGER_ROTATING_MAX_FILE_SIZE,
                    FLOWENGINE_LOGGER_ROTATING_MAX_FILE_NUM);
        with_error_logger_rotating->set_level(spdlog::level::err);
        //with_error_logger_rotating->set_pattern(FLOWENGINE_LOGGER_PATTERN);
        sinks.push_back(with_error_logger_rotating);
    }

    if (with_trace) {
        auto with_trace_logger_rotating = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                    FLOWENGINE_LOGGER_LOGGER_TRACE_FILENAME,
                    FLOWENGINE_LOGGER_ROTATING_MAX_FILE_SIZE,
                    FLOWENGINE_LOGGER_ROTATING_MAX_FILE_NUM);
        with_trace_logger_rotating->set_level(spdlog::level::trace);
        //with_trace_logger_rotating->set_pattern(FLOWENGINE_LOGGER_PATTERN);
        sinks.push_back(with_trace_logger_rotating);
    }

    auto combined_logger = std::make_shared<spdlog::logger>(FLOWENGINE_LOGGER_NAME,
            begin(sinks), end(sinks));
    // register it if you need to access it globally
    // set_level will limit all sinks
    combined_logger->set_level(spdlog::level::trace);
    combined_logger->set_pattern(FLOWENGINE_LOGGER_PATTERN);
    spdlog::register_logger(combined_logger);

    // set flush every 2 seconds
    FlowEngineLoggerSetFlushEvery(2);
}

void FlowEngineLoggerSetLevel(const int level) {
    // Note: sdplog::get is a thread safe function
    std::shared_ptr<spdlog::logger> logger_ptr = spdlog::get(FLOWENGINE_LOGGER_NAME);
    if (!logger_ptr) {
        fprintf(stderr, "Failed to get logger, Please init logger firstly.\n");
    }

    logger_ptr->set_level(static_cast<spdlog::level::level_enum>(level));
}

void FlowEngineLoggerSetPattern(const char* format) {
    // Note: sdplog::get is a thread safe function
    std::shared_ptr<spdlog::logger> logger_ptr = spdlog::get(FLOWENGINE_LOGGER_NAME);
    if (!logger_ptr) {
        fprintf(stderr, "Failed to get logger, Please init logger firstly.\n");
    }

    logger_ptr->set_pattern(format);
}

void FlowEngineLoggerSetFlushEvery(const int interval) {
    spdlog::flush_every(std::chrono::seconds(interval));
}

// drop all loggers reference
void FlowEngineLoggerDrop() {
    spdlog::drop_all();
}
