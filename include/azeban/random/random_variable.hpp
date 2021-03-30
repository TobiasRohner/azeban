#ifndef RANDOM_VARIABLE_H_
#define RANDOM_VARIABLE_H_

namespace azeban {

template <typename Result>
class RandomVariableImpl {
public:
  using result_t = Result;

  RandomVariableImpl() = default;
  RandomVariableImpl(const RandomVariableImpl &) = default;
  RandomVariableImpl(RandomVariableImpl &&) = default;

  virtual ~RandomVariableImpl() = default;

  RandomVariableImpl &operator=(const RandomVariableImpl &) = default;
  RandomVariableImpl &operator=(RandomVariableImpl &&) = default;

  virtual Result get() = 0;
};

template <typename Result>
class RandomVariable final {
public:
  using result_t = Result;

  RandomVariable(const std::shared_ptr<RandomVariableImpl<Result>> &rv)
      : rv_(rv) {}
  RandomVariable(const RandomVariable &) = default;
  RandomVariable(RandomVariable &&) = default;

  ~RandomVariable() = default;

  RandomVariable &operator=(const RandomVariable &) = default;
  RandomVariable &operator=(RandomVariable &&) = default;

  operator result_t() { return get(); }

  Result get() { return rv_->get(); }

private:
  std::shared_ptr<RandomVariableImpl<Result>> rv_;
};

}

#endif
