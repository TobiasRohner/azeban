#ifndef AZEBAN_UTILS_FORMAT_TIME_HPP_
#define AZEBAN_UTILS_FORMAT_TIME_HPP_

#include <chrono>
#include <fmt/core.h>

namespace azeban {

template <typename Rep, typename Period>
std::string format_time(std::chrono::duration<Rep, Period> duration) {
  using namespace std::chrono_literals;
  std::string formatted;
  const std::chrono::hours h = std::chrono::floor<std::chrono::hours>(duration);
  if (h.count() > 0) {
    formatted += std::to_string(h.count()) + "h";
    duration -= h;
  }
  const std::chrono::minutes min
      = std::chrono::floor<std::chrono::minutes>(duration);
  if (min.count() > 0) {
    formatted += std::to_string(min.count()) + "min";
    duration -= min;
  }
  if (formatted.size() > 0) {
    using s_t = std::chrono::duration<double>;
    const s_t s = std::chrono::duration_cast<s_t>(duration);
    formatted += fmt::format("{:.3f}s", s.count());
    return formatted;
  } else {
    if (duration < 1us) {
      using ns_t = std::chrono::duration<double, std::nano>;
      const ns_t ns = std::chrono::duration_cast<ns_t>(duration);
      return fmt::format("{:.3f}ns", ns.count());
    } else if (duration < 1ms) {
      using us_t = std::chrono::duration<double, std::micro>;
      const us_t us = std::chrono::duration_cast<us_t>(duration);
      return fmt::format("{:.3f}us", us.count());
    } else if (duration < 1s) {
      using ms_t = std::chrono::duration<double, std::milli>;
      const ms_t ms = std::chrono::duration_cast<ms_t>(duration);
      return fmt::format("{:.3f}ms", ms.count());
    } else {
      using s_t = std::chrono::duration<double>;
      const s_t s = std::chrono::duration_cast<s_t>(duration);
      return fmt::format("{:.3f}s", s.count());
    }
  }
}

}

#endif
