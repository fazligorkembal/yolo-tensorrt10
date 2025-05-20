#pragma once
#include <iostream>

#define ASSERT(condition, message)                                                       \
    do                                                                                   \
    {                                                                                    \
        if (!(condition))                                                                \
        {                                                                                \
            std::cerr << "\n\nAssertion failed: " << message << "\n\t\t  " << #condition \
                      << ", file " << __FILE__ << ", line " << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (false)