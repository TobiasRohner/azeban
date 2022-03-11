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
#include <azeban/profiler.hpp>
#if ZISA_HAS_CUDA
#include <cuda_runtime.h>
#endif
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif
#include <algorithm>
#include <azeban/cuda/cuda_check_error.hpp>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <iomanip>
#include <type_traits>

namespace azeban {

std::forward_list<Profiler::RecordHost> Profiler::host_records_
    = std::forward_list<Profiler::RecordHost>();
#if ZISA_HAS_CUDA
std::forward_list<Profiler::RecordDevice> Profiler::device_records_
    = std::forward_list<Profiler::RecordDevice>();
#endif
Profiler::time_point_t Profiler::start_time = Profiler::time_point_t();
Profiler::time_point_t Profiler::end_time = Profiler::time_point_t();

Profiler::Timespan::Timespan(const RecordHost &record)
    : name(record.name),
      start_time(record.start_time),
      duration(std::chrono::duration_cast<duration_t>(record.end_time
                                                      - record.start_time)) {}

#if ZISA_HAS_CUDA
Profiler::Timespan::Timespan(const RecordDevice &record)
    : name(record.name), start_time(record.start_time) {
  cudaEventSynchronize(record.end_event);
  float dt;
  const auto err
      = cudaEventElapsedTime(&dt, record.start_event, record.end_event);
  cudaCheckError(err);
  duration = duration_t(dt);
}
#endif

void Profiler::start() { start_time = clock::now(); }

void Profiler::stop() { end_time = clock::now(); }

Profiler::RecordHost *Profiler::start(const std::string &name) {
  host_records_.emplace_front();
  RecordHost *record = &host_records_.front();
  record->name = name;
  record->start_time = clock::now();
  return record;
}

void Profiler::stop(RecordHost *record) { record->end_time = clock::now(); }

#if ZISA_HAS_CUDA
Profiler::RecordDevice *Profiler::start(const std::string &name,
                                        cudaStream_t stream) {
  device_records_.emplace_front();
  RecordDevice *record = &device_records_.front();
  record->name = name;
  record->stream = stream;
  record->start_time = clock::now();
  cudaEventCreate(&(record->start_event));
  cudaEventRecord(record->start_event, stream);
  return record;
}

void Profiler::stop(RecordDevice *record) {
  cudaEventCreate(&(record->end_event));
  cudaEventRecord(record->end_event, record->stream);
}
#endif

void Profiler::serialize(std::ostream &os) {
  std::vector<Timespan> timeline = to_timeline(host_records_);
  for (const Timespan &ts : timeline) {
    duration_t start
        = std::chrono::duration_cast<duration_t>(ts.start_time - start_time);
    os << std::setprecision(
        std::numeric_limits<typename duration_t::rep>::digits10 + 1)
       << ts.name << ' ' << start.count() << ' ' << ts.duration.count() << ' ';
  }
  os << '\n';
  timeline = to_timeline(device_records_);
  for (const Timespan &ts : timeline) {
    duration_t start
        = std::chrono::duration_cast<duration_t>(ts.start_time - start_time);
    os << std::setprecision(
        std::numeric_limits<typename duration_t::rep>::digits10 + 1)
       << ts.name << ' ' << start.count() << ' ' << ts.duration.count() << ' ';
  }
}

template <typename RecordType>
std::vector<Profiler::Timespan>
Profiler::to_timeline(const std::forward_list<RecordType> &records) {
  std::vector<Timespan> timeline;
  for (const RecordType &record : records) {
    timeline.emplace_back(record);
  }
  return timeline;
}

template std::vector<Profiler::Timespan>
Profiler::to_timeline(const std::forward_list<Profiler::RecordHost> &);
template std::vector<Profiler::Timespan>
Profiler::to_timeline(const std::forward_list<Profiler::RecordDevice> &);

#if AZEBAN_DO_PROFILE
ProfileHost::ProfileHost(const std::string &name)
    : record_(Profiler::start(name)) {}

ProfileHost::~ProfileHost() { stop(); }

void ProfileHost::stop() {
  if (record_) {
    Profiler::stop(record_);
    record_ = nullptr;
  }
}

#if ZISA_HAS_CUDA
ProfileDevice::ProfileDevice(const std::string &name, cudaStream_t stream)
    : record_(Profiler::start(name, stream)) {}

ProfileDevice::~ProfileDevice() { stop(); }

void ProfileDevice::stop() {
  if (record_) {
    Profiler::stop(record_);
    record_ = nullptr;
  }
}
#endif
#endif

}
