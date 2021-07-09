#ifndef LOGGING_
#define LOGGING_

#include <cstdlib>
#include <fmt/core.h>

#define AZEBAN_ERR_IF(cond, msg)                                               \
  if ((cond)) {                                                                \
    fmt::print(stderr, msg);                                                   \
    std::exit(1);                                                              \
  }

#define AZEBAN_ERR(msg)                                                        \
  fmt::print(stderr, msg);                                                     \
  std::exit(1);

#endif
