//
// Created by hacht on 2/28/17.
//

#pragma once

#include <exception>
#include <iostream>
#include <string>
#include "colored_print.h"

namespace simpleCNN {

  /**
   * error exception class for simpleCNN
   **/
  class simple_error : public std::exception {
   public:
    explicit simple_error(const std::string& msg) : msg_(msg) {}

    const char* what() const throw() override { return msg_.c_str(); }

   private:
    std::string msg_;
  };

  /**
   * warning class for simpleCNN (for debug)
   **/
  class simple_warn {
   public:
    explicit simple_warn(const std::string& msg) : msg_(msg) {
      coloredPrint(Color::RED, msg_h_);
      std::cout << msg_ << std::endl;
    }

   private:
    std::string msg_;
    std::string msg_h_ = std::string("[WARNING] ");
  };

  /**
   * info class for simpleCNN (for debug)
   **/
  class simple_info {
   public:
    explicit simple_info(const std::string& msg) : msg_(msg) {
      coloredPrint(Color::GREEN, msg_h);
      std::cout << msg_ << std::endl;
    }

   private:
    std::string msg_;
    std::string msg_h = std::string("[INFO] ");
  };

  class simple_not_implemented_error : public simple_error {
   public:
    explicit simple_not_implemented_error(const std::string& msg = "not implemented") : simple_error(msg) {}
  };
}  // namespace simpleCNN
