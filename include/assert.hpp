#pragma once
#include <iostream>

#define ASSERT(condition, message)                                                       \
    do                                                                                   \
    {                                                                                    \
        if (!(condition))                                                                \
        {                                                                                \
            std::cerr << "\n\n[ERROR] - " << message << ", file " <<                     \
            __FILE__ << ", line " << __LINE__ <<                                         \
            "\n\t\t  " << #condition  << std::endl;                                      \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (false)