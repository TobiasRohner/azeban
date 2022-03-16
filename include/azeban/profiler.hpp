/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef AZEBAN_PROFILER_HPP_
#define AZEBAN_PROFILER_HPP_

#include <azeban/config.hpp>
#include <chrono>
#include <forward_list>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <zisa/config.hpp>
#if ZISA_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace azeban {

class Profiler {
public:
  using clock = std::chrono::steady_clock;
  using time_point_t = typename clock::time_point;
  using duration_t = std::chrono::duration<float, std::milli>;

  struct RecordHost {
    std::string name;
    time_point_t start_time;
    time_point_t end_time;
  };

#if ZISA_HAS_CUDA
  struct RecordDevice {
    std::string name;
    cudaStream_t stream;
    time_point_t start_time;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
  };
#endif

  struct Timespan {
    Timespan() = default;
    Timespan(const RecordHost &record);
#if ZISA_HAS_CUDA
    Timespan(const RecordDevice &record);
#endif
    std::string name;
    time_point_t start_time;
    duration_t duration;
  };

  struct StageSummary {
    duration_t total_time = duration_t(0);
    size_t num_calls = 0;
  };

  static void start();
  static void stop();
  static RecordHost *start(const std::string &name);
  static void stop(RecordHost *record);
#if ZISA_HAS_CUDA
  static RecordDevice *start(const std::string &name, cudaStream_t stream);
  static void stop(RecordDevice *record);
#endif

  static void serialize(std::ostream &os);
  static void summarize(std::ostream &os);

private:
  static std::forward_list<RecordHost> host_records_;
#if ZISA_HAS_CUDA
  static std::forward_list<RecordDevice> device_records_;
#endif
  static time_point_t start_time;
  static time_point_t end_time;

  template <typename RecordType>
  static std::vector<Timespan>
  to_timeline(const std::forward_list<RecordType> &records);

  static std::map<std::string, StageSummary>
  build_summary(const std::vector<Timespan> &timeline);
};

#if AZEBAN_DO_PROFILE
class ProfileHost {
public:
  ProfileHost(const std::string &name);
  ~ProfileHost();
  void stop();

private:
  Profiler::RecordHost *record_;
};
#if ZISA_HAS_CUDA
class ProfileDevice {
public:
  ProfileDevice(const std::string &name, cudaStream_t stream);
  ~ProfileDevice();
  void stop();

private:
  Profiler::RecordDevice *record_;
};
#endif
#else
class ProfileHost {
public:
  ProfileHost(const std::string &){};
  void stop(){};
};
#if ZISA_HAS_CUDA
class ProfileDevice {
public:
  ProfileDevice(const std::string &, cudaStream_t){};
  void stop(){};
};
#endif
#endif

}

#endif
