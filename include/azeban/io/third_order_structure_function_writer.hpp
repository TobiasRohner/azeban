#ifndef AZEBAN_IO_THIRD_ORDER_STRUCTURE_FUNCTION_WRITER_HPP_
#define AZEBAN_IO_THIRD_ORDER_STRUCTURE_FUNCTION_WRITER_HPP_

#include <azeban/io/structure_function_writer.hpp>
#include <azeban/operations/structure_function_functionals.hpp>

namespace azeban {

template <int Dim>
class ThirdOrderStructureFunctionWriter
    : public StructureFunctionWriter<Dim, SFThirdOrderFunctional> {
  using super = StructureFunctionWriter<Dim, SFThirdOrderFunctional>;

public:
  ANY_DEVICE
  ThirdOrderStructureFunctionWriter(const std::string &path,
                                    const Grid<Dim> &grid,
                                    const std::vector<real_t> &snapshot_times,
                                    zisa::int_t sample_idx_start,
                                    ssize_t max_h);
  ANY_DEVICE_INLINE
  ThirdOrderStructureFunctionWriter(const ThirdOrderStructureFunctionWriter &)
      = default;
  ANY_DEVICE_INLINE
  ThirdOrderStructureFunctionWriter(ThirdOrderStructureFunctionWriter &&)
      = default;

  virtual ~ThirdOrderStructureFunctionWriter() override = default;

  ANY_DEVICE_INLINE ThirdOrderStructureFunctionWriter &
  operator=(const ThirdOrderStructureFunctionWriter &)
      = default;
  ANY_DEVICE_INLINE ThirdOrderStructureFunctionWriter &
  operator=(ThirdOrderStructureFunctionWriter &&)
      = default;

  using super::next_timestep;
  using super::reset;
  using super::write;
};

}

#endif
