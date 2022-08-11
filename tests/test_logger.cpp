#include "logger/logger.hpp"

int main()
{
    FlowEngineLoggerInit(true, true, true, true);

    FLOWENGINE_LOGGER_TRACE("hello logger, {}", 2020);
    FLOWENGINE_LOGGER_DEBUG("hello logger, {}", 2020);
    FLOWENGINE_LOGGER_INFO("hello logger, {}", 2020);
    FLOWENGINE_LOGGER_WARN("hello logger, {}", 2020);
    FLOWENGINE_LOGGER_ERROR("hello logger, {}", 2020);
    FLOWENGINE_LOGGER_CRITICAL("hello logger, {}", 2020);

    FlowEngineLoggerDrop();

    return 0;
}
