//
// Created by hacht on 2/28/17.
//

#pragma once

#include <cassert>
#include <cstdarg>
#include <string>

namespace simpleCNN {

    enum class Color { RED, GREEN, BLUE, YELLOW };

    inline const char *getColorEscape(Color c) {
        switch (c) {
        case Color::RED: return "\033[31m";
        case Color::GREEN: return "\033[32m";
        case Color::BLUE: return "\033[34m";
        case Color::YELLOW: return "\033[33m";
        default: assert(0); return "";
        }
    }

    // Ex: "Called with %d arguments: %s %s %s", 2, string, string, string.
    inline void coloredPrint(Color c, const char *fmt, ...) {
        va_list args;
        va_start(args, fmt);

        printf("%s", getColorEscape(c));
        vprintf(fmt, args);
        printf("\033[m");

        va_end(args);
    }

    inline void coloredPrint(Color c, const std::string &msg) {
        coloredPrint(c, msg.c_str());
    }

}  // namespace simpleCNN
