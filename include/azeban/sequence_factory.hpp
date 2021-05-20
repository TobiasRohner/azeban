#ifndef SEQUENCE_FACTORY_HPP_
#define SEQUENCE_FACTORY_HPP_

#include <nlohmann/json.hpp>
#include <vector>


namespace azeban {

template<typename T>
std::vector<T> make_sequence(const nlohmann::json &config) {
  if (config.is_number()) {
    return std::vector<T>(1, config.get<T>());
  }
  else if (config.is_array()) {
    return config.get<std::vector<T>>();
  }
  else if (config.is_object()) {
    std::vector<T> res;
    T start = 0;
    if (config.contains("start")) {
      start = config["start"];
    }
    if (!config.contains("stop")) {
      LOG_ERR("Expected key \"stop\" in sequence specifier");
    }
    T stop = config["stop"];
    if (config.contains("n")) {
      const size_t n = config["n"];
      for (size_t i = 0 ; i <= n ; ++i) {
	res.push_back(start + i * (stop - start) / n);
      }
      return res;
    }
    else if (config.contains("step")) {
      const T step = config["step"];
      res.push_back(start);
      while (res.back() < stop) {
	res.push_back(res.back() + step);
      }
      res.back() = stop;
      return res;
    }
    else {
      LOG_ERR("Expected either \"n\" or \"step\" in sequence specification");
    }
  }
  else {
    LOG_ERR("Unsupported type for a sequence");
  }
}

}


#endif
