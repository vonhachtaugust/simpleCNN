//
// Created by hacht on 5/16/17.
//

#pragma once

#include "../util/simple_error.h"

namespace simpleCNN {

  template <typename T, typename Container = std::vector<T>>
  static void to_archive(std::ofstream& ofs,
                         Container& args,
                         const size_t precision = std::numeric_limits<T>::digits10 + 2) {
    ofs << std::setprecision(precision);

    for (auto iter = std::begin(args); iter != std::end(args); ++iter) {
      ofs << *iter << " ";
    }
  }

  template <typename T, typename Container = std::vector<T>>
  static void from_archive(std::ifstream& ifs,
                           Container& data,
                           const size_t precision = std::numeric_limits<T>::digits10 + 2) {
    ifs >> std::setprecision(precision);

    for (auto iter = std::begin(data); iter != std::end(data); ++iter) {
      ifs >> *iter;
    }
  }

  template <typename T, typename Container = std::vector<T>>
  static void save_data_to_file(const std::string& filename, Container& args) {
    std::ofstream ofs(filename.c_str(), std::ios::binary);

    if (ofs.bad() || ofs.fail()) {
      throw simple_error("Failed to open: " + filename);
    }

    to_archive<T>(ofs, args);
  }

  template <typename T, typename Container = std::vector<T>>
  static void load_data_from_file(const std::string& filename, Container& data) {
    std::ifstream ifs(filename.c_str(), std::ios::binary);

    if (ifs.fail() || ifs.bad()) {
      throw simple_error("Failed to open " + filename);
    }

    from_archive<T>(ifs, data);
  }
}