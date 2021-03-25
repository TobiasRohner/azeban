#include <azeban/profiler.hpp>
#if ZISA_HAS_CUDA
#include <cuda_runtime.h>
#endif
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif
#include <algorithm>
#include <fmt/chrono.h>
#include <fmt/core.h>

namespace azeban {

std::map<std::string, Profiler::Stage> Profiler::stages_
    = std::map<std::string, Profiler::Stage>();
Profiler::time_point Profiler::start_time = Profiler::time_point();
Profiler::duration Profiler::elapsed = Profiler::duration(0);

void Profiler::start() {
  sync();
  start_time = clock::now();
}

void Profiler::stop() {
  sync();
  time_point end_time = clock::now();
  elapsed = end_time - start_time;
}

void Profiler::sync() {
#if ZISA_HAS_CUDA
  cudaDeviceSynchronize();
#endif
#if AZEBAN_HAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void Profiler::start(const std::string &name) {
  auto [it, is_new] = stages_.emplace(name, name);
  Stage &stage = it->second;
  ++stage.num_calls;
  sync();
  stage.start_time = clock::now();
}

void Profiler::stop(const std::string &name) {
  sync();
  time_point end_time = clock::now();
  Stage &stage = stages_.find(name)->second;
  stage.elapsed += end_time - stage.start_time;
}

std::string Profiler::summary() {
  struct StageSummary {
    real_t percentage;
    std::string percentage_str;
    std::string name;
    std::string num_calls;
    std::string elapsed_total;
    std::string elapsed_per_call;
  };

  std::vector<StageSummary> stage_summ;
  for (auto [name, stage] : stages_) {
    stage_summ.emplace_back();
    StageSummary &summ = stage_summ.back();
    summ.percentage
        = static_cast<real_t>(stage.elapsed.count()) / elapsed.count();
    summ.percentage_str = fmt::format("{}%", 100 * summ.percentage);
    summ.name = stage.name;
    summ.num_calls = fmt::format("{}", stage.num_calls);
    summ.elapsed_total = fmt::format("{}", stage.elapsed);
    summ.elapsed_per_call = fmt::format("{}", stage.elapsed / stage.num_calls);
  }
  const auto ordering = [](const StageSummary &s1, const StageSummary &s2) {
    return s1.percentage > s2.percentage;
  };
  std::sort(stage_summ.begin(), stage_summ.end(), ordering);

  const std::string percentage_header = "Time %";
  const std::string name_header = "Name";
  const std::string num_calls_header = "Nr. Calls";
  const std::string elapsed_total_header = "Total Time";
  const std::string elapsed_per_call_header = "Time Per Call";
  zisa::int_t percentage_size = percentage_header.size();
  zisa::int_t name_size = name_header.size();
  zisa::int_t num_calls_size = num_calls_header.size();
  zisa::int_t elapsed_total_size = elapsed_total_header.size();
  zisa::int_t elapsed_per_call_size = elapsed_per_call_header.size();
  for (auto &&summ : stage_summ) {
    percentage_size = std::max(percentage_size, summ.percentage_str.size());
    name_size = std::max(name_size, summ.name.size());
    num_calls_size = std::max(num_calls_size, summ.num_calls.size());
    elapsed_total_size
        = std::max(elapsed_total_size, summ.elapsed_total.size());
    elapsed_per_call_size
        = std::max(elapsed_per_call_size, summ.elapsed_per_call.size());
  }

  const auto pad_left = [](std::string str, zisa::int_t n) {
    str.insert(str.end(), n - str.size(), ' ');
    return str;
  };
  const auto line = [&](const StageSummary &summ) {
    std::string str = " ";
    str += pad_left(summ.percentage_str, percentage_size);
    str += " | ";
    str += pad_left(summ.elapsed_total, elapsed_total_size);
    str += " | ";
    str += pad_left(summ.num_calls, num_calls_size);
    str += " | ";
    str += pad_left(summ.elapsed_per_call, elapsed_per_call_size);
    str += " | ";
    str += pad_left(summ.name, name_size);
    str += "\n";
    return str;
  };

  StageSummary header;
  header.percentage_str = percentage_header;
  header.name = name_header;
  header.num_calls = num_calls_header;
  header.elapsed_total = elapsed_total_header;
  header.elapsed_per_call = elapsed_per_call_header;

  std::string summary_str = line(header);
  summary_str += std::string(summary_str.size() - 1, '-') + '\n';
  for (auto &&summ : stage_summ) {
    summary_str += line(summ);
  }

  return fmt::format("Total Time: {}\n{}", elapsed, summary_str);
}

nlohmann::json Profiler::json() {
  nlohmann::json result;
  result["timeunit"] = "ns";
  result["elapsed"]
      = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  std::vector<nlohmann::json> stages;
  for (auto [name, stage] : stages_) {
    stages.emplace_back();
    auto &sj = stages.back();
    sj["name"] = stage.name;
    sj["num_calls"] = stage.num_calls;
    sj["elapsed"]
        = std::chrono::duration_cast<std::chrono::nanoseconds>(stage.elapsed)
              .count();
  }
  result["stages"] = stages;
  return result;
}

}
