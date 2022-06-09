#ifndef AZEBAN_IO_PARAVIEW_CATALYST_WRITER_FACTORY_HPP_
#define AZEBAN_IO_PARAVIEW_CATALYST_WRITER_FACTORY_HPP_

#include <azeban/io/paraview_catalyst_writer.hpp>
#include <memory>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<ParaviewCatalystWriter<Dim>>
make_paraview_catalyst_writer(const nlohmann::json &config,
                              const Grid<Dim> &grid,
                              zisa::int_t sample_idx_start);

}

#endif
