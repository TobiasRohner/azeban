#ifndef PROFILER_H_
#define PROFILER_H_

#include <azeban/config.hpp>
#include <chrono>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <zisa/config.hpp>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

namespace azeban {

class Profiler {
public:
  using clock = std::chrono::steady_clock;
  using duration = typename clock::duration;
  using time_point = typename clock::time_point;

  struct Stage {
    Stage(const std::string &_name)
        : name(_name), num_calls(0), start_time(), elapsed(0) {}
    std::string name;
    zisa::int_t num_calls;
    time_point start_time;
    duration elapsed;
  };

  static void start();
  static void stop();
  static void sync();
  static void start(const std::string &name);
  static void stop(const std::string &name);
#if AZEBAN_HAS_MPI
  static void sync(MPI_Comm comm);
  static void start(const std::string &name, MPI_Comm comm);
  static void stop(const std::string &name, MPI_Comm comm);
#endif

  static std::string summary();
  static nlohmann::json json();

private:
  static std::map<std::string, Stage> stages_;
  static time_point start_time;
  static duration elapsed;
};

#if AZEBAN_DO_PROFILE
#define AZEBAN_PROFILE_START(...) Profiler::start(__VA_ARGS__)
#define AZEBAN_PROFILE_STOP(...) Profiler::stop(__VA_ARGS__)
#else
#define AZEBAN_PROFILE_START(...)
#define AZEBAN_PROFILE_STOP(...)
#endif

}

#endif
