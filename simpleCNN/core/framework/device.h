//
// Created by hacht on 3/4/17.
//
#pragma once

#include <iostream>
#include "../../util/simple_error.h"

namespace simpleCNN {

  enum class device_t { NONE, CPU, GPU };

  inline std::ostream& operator<<(std::ostream& os, device_t type) {
    switch (type) {
      case device_t::NONE: os << "NONE"; break;
      case device_t::CPU: os << "CPU"; break;
      case device_t::GPU: os << "GPU"; break;
      default: throw simple_error("Not supported ostream enum: " + std::to_string(static_cast<int>(type))); break;
    }
    return os;
  }

  class Device {
   public:
    inline explicit Device(device_t type);

    /* CPU/GPU OpenCL constructor.
    * Device context is initialized in constructor.
    *
    * @param type The device type. Can be both CPU and GPU.
    * @param platform_id The platform identification number.
    * @param device_id The device identification number.
    */
    inline explicit Device(device_t type, const int platform_id, const int device_id);

    // Returns the device type
    device_t type() const { return type_; }

    // Returns the platform id
    int platformId() const { return platform_id_; }

    // Returns the device id
    int deviceId() const { return device_id_; }

    bool operator==(const Device& d) const {
      if (d.type() == this->type() && d.platformId() == this->platformId() && d.deviceId() == this->deviceId()) {
        return true;
      }
      return false;
    }

   private:
    /* The device type */
    device_t type_;

    /* The platform identification number */
    int platform_id_;
    /* The device identification number */
    int device_id_;
  };
}  // namespace simpleCNN
