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
#include <azeban/utils/format_time.hpp>
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

void Profiler::summarize(std::ostream &os) {
  const std::map<std::string, StageSummary> summary
      = build_summary(to_timeline(host_records_));
  struct TableEntry {
    TableEntry(const std::string &name_, const StageSummary &stage)
        : name(name_),
          percentage(100 * stage.total_time / (end_time - start_time)),
          total_time(stage.total_time),
          num_calls(stage.num_calls),
          time_per_call(stage.total_time / stage.num_calls) {}
    std::string name;
    double percentage;
    duration_t total_time;
    size_t num_calls;
    duration_t time_per_call;
  };
  std::vector<TableEntry> table_entries;
  for (const auto &[name, stage] : summary) {
    table_entries.emplace_back(name, stage);
  }
  std::sort(table_entries.begin(),
            table_entries.end(),
            [](const TableEntry &lhs, const TableEntry &rhs) {
              return lhs.percentage > rhs.percentage;
            });
  std::vector<std::string> percentage_col;
  std::vector<std::string> total_time_col;
  std::vector<std::string> num_calls_col;
  std::vector<std::string> time_per_call_col;
  std::vector<std::string> name_col;
  int percentage_col_width = 6;
  int total_time_col_width = 10;
  int num_calls_col_width = 9;
  int time_per_call_col_width = 13;
  int name_col_width = 4;
  for (const TableEntry &entry : table_entries) {
    percentage_col.push_back(fmt::format("{:.3f}%", entry.percentage));
    percentage_col_width = std::max(
        percentage_col_width, static_cast<int>(percentage_col.back().length()));
    total_time_col.push_back(format_time(entry.total_time));
    total_time_col_width = std::max(
        total_time_col_width, static_cast<int>(total_time_col.back().length()));
    num_calls_col.push_back(fmt::format("{}", entry.num_calls));
    num_calls_col_width = std::max(
        num_calls_col_width, static_cast<int>(num_calls_col.back().length()));
    time_per_call_col.push_back(format_time(entry.time_per_call));
    time_per_call_col_width
        = std::max(time_per_call_col_width,
                   static_cast<int>(time_per_call_col.back().size()));
    name_col.push_back(entry.name);
    name_col_width
        = std::max(name_col_width, static_cast<int>(name_col.back().length()));
  }
  std::string table
      = "Total Simulation Time: " + format_time(end_time - start_time) + '\n';
  table += fmt::format(" {:<{}} | {:<{}} | {:<{}} | {:<{}} | {:<{}} \n",
                       "Time %",
                       percentage_col_width,
                       "Total Time",
                       total_time_col_width,
                       "Nr. Calls",
                       num_calls_col_width,
                       "Time per Call",
                       time_per_call_col_width,
                       "Name",
                       name_col_width);
  const int total_table_width = 1 + percentage_col_width + 3
                                + total_time_col_width + 3 + num_calls_col_width
                                + 3 + time_per_call_col_width + 3
                                + name_col_width + 1;
  table += std::string(total_table_width, '-') + '\n';
  for (int row = 0; row < table_entries.size(); ++row) {
    table += fmt::format(" {:>{}} | {:>{}} | {:>{}} | {:>{}} | {:<{}} \n",
                         percentage_col[row],
                         percentage_col_width,
                         total_time_col[row],
                         total_time_col_width,
                         num_calls_col[row],
                         num_calls_col_width,
                         time_per_call_col[row],
                         time_per_call_col_width,
                         name_col[row],
                         name_col_width);
  }
  os << table;
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

std::map<std::string, Profiler::StageSummary>
Profiler::build_summary(const std::vector<Profiler::Timespan> &timeline) {
  std::map<std::string, StageSummary> summary;
  for (const auto &span : timeline) {
    auto &stage = summary[span.name];
    ++stage.num_calls;
    stage.total_time += span.duration;
  }
  return summary;
}

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
