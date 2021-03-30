#ifndef UNIFORM_H_
#define UNIFORM_H_

#include "random_variable.hpp"
#include <random>

namespace azeban {

template <typename Result, typename RNG>
class Uniform final : public RandomVariableImpl<Result> {
  using super = RandomVariableImpl<Result>;

public:
  using result_t = Result;

  Uniform(result_t min, result_t max, RNG &rng) : distr_(min, max), rng_(rng) {}
  Uniform(const Uniform &) = default;
  Uniform(Uniform &&) = default;

  virtual ~Uniform() override = default;

  Uniform &operator=(const Uniform &) = default;
  Uniform &operator=(Uniform &&) = default;

  virtual result_t get() override { return distr_(rng_); }

private:
  std::uniform_real_distribution<result_t> distr_;
  RNG &rng_;
};

}

#endif
