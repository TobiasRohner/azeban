#ifndef DELTA_H_
#define DELTA_H_

#include "random_variable.hpp"

namespace azeban {

template <typename Result>
class Delta final : public RandomVariableImpl<Result> {
  using super = RandomVariableImpl<Result>;

public:
  using result_t = Result;

  Delta(result_t result) : result_(result) {}
  Delta(const Delta &) = default;
  Delta(Delta &&) = default;

  virtual ~Delta() override = default;

  Delta &operator=(const Delta &) = default;
  Delta &operator=(Delta &&) = default;

  virtual result_t get() override { return result_; }

private:
  result_t result_;
};

}

#endif
