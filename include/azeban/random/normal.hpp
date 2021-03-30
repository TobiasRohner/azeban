#ifndef NORMAL_H_
#define NORMAL_H_

#include "random_variable.hpp"
#include <random>

namespace azeban {

template <typename Result, typename RNG>
class Normal final : public RandomVariableImpl<Result> {
  using super = RandomVariableImpl<Result>;

public:
  using result_t = Result;

  Normal(result_t mu, result_t sigma, RNG &rng)
      : distr_(mu, sigma), rng_(rng) {}
  Normal(const Normal &) = default;
  Normal(Normal &&) = default;

  virtual ~Normal() override = default;

  Normal &operator=(const Normal &) = default;
  Normal &operator=(Normal &&) = default;

  virtual result_t get() override { return distr_(rng_); }

private:
  std::normal_distribution<result_t> distr_;
  RNG &rng_;
};

}

#endif
