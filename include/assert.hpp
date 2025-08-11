#pragma once
#include <iostream>
#include <spdlog/spdlog.h>

#define ASSERT(condition, message)                              \
    do                                                          \
    {                                                           \
        if (!(condition))                                       \
        {                                                       \
            spdlog::error("{}\n\tfile: {}, line:{}\n", message, \
                          __FILE__, __LINE__);                  \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (false)

#define DEBUG(message)                \
    do                                \
    {                                 \
        spdlog::debug("{}", message); \
    } while (false)

#define INFO(message)                \
    do                               \
    {                                \
        spdlog::info("{}", message); \
    } while (false)

#define WARN(message)                \
    do                               \
    {                                \
        spdlog::warn("{}", message); \
    } while (false)
