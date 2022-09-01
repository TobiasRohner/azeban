#if AZEBAN_HAS_CATALYST

#include <azeban/io/paraview_catalyst_writer.hpp>
#include <azeban/profiler.hpp>
#include <catalyst.hpp>

namespace azeban {

template <int Dim>
ParaviewCatalystWriter<Dim>::ParaviewCatalystWriter(
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    const std::vector<std::string> &scripts,
    zisa::int_t sample_idx_start)
    : super(grid, snapshot_times, sample_idx_start) {
  conduit_cpp::Node node;
  for (size_t i = 0; i < scripts.size(); ++i) {
    const std::string &script = scripts[i];
    node["catalyst/scripts/script" + std::to_string(i)].set_string(script);
  }
  node["catalyst_load/implementation"] = "paraview";
  catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok) {
    fmt::print(stderr, "Failed to initialize catalyst: {}\n", int(err));
    exit(1);
  }
}

template <int Dim>
void ParaviewCatalystWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, real_t t) {
  ProfileHost pofile("ParaviewCatalystWriter::write");

  conduit_cpp::Node exec_params;

  auto state = exec_params["catalyst/state"];
  state["timestep"].set(snapshot_idx_);
  state["time"].set(t);

  auto channel = exec_params["catalyst/channels/grid"];
  channel["type"].set("mesh");

  auto mesh = channel["data"];
  mesh["coordsets/coords/type"].set("uniform");
  mesh["coordsets/coords/dims/i"].set(grid_.N_phys);
  mesh["coordsets/coords/dims/j"].set(Dim > 1 ? grid_.N_phys : 1);
  mesh["coordsets/coords/dims/k"].set(Dim > 2 ? grid_.N_phys : 1);
  mesh["coordsets/coords/origin/x"].set(0);
  mesh["coordsets/coords/origin/y"].set(0);
  mesh["coordsets/coords/origin/z"].set(0);
  mesh["coordsets/coords/spacing/dx"].set(1. / grid_.N_phys);
  mesh["coordsets/coords/spacing/dy"].set(1. / grid_.N_phys);
  mesh["coordsets/coords/spacing/dz"].set(1. / grid_.N_phys);
  mesh["topologies/mesh/type"].set("uniform");
  mesh["topologies/mesh/coordset"].set("coords");

  auto fields = mesh["fields"];
  fields["velocity/association"].set("vertex");
  fields["velocity/topology"].set("mesh");
  fields["velocity/volume_dependent"].set("false");
  fields["velocity/values/x"].set_external(u.raw(),
                                           zisa::pow<Dim>(grid_.N_phys),
                                           0 * zisa::pow<Dim>(grid_.N_phys)
                                               * sizeof(real_t));
  if (Dim > 1) {
    fields["velocity/values/y"].set_external(u.raw(),
                                             zisa::pow<Dim>(grid_.N_phys),
                                             1 * zisa::pow<Dim>(grid_.N_phys)
                                                 * sizeof(real_t));
  }
  if (Dim > 2) {
    fields["velocity/values/z"].set_external(u.raw(),
                                             zisa::pow<Dim>(grid_.N_phys),
                                             2 * zisa::pow<Dim>(grid_.N_phys)
                                                 * sizeof(real_t));
  }

  catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
  if (err != catalyst_status_ok) {
    fmt::print(stderr, "Failed to execute catalyst: {}\n", int(err));
    exit(1);
  }

  ++snapshot_idx_;
}

template <int Dim>
void ParaviewCatalystWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, real_t t) {
  ZISA_UNUSED(u_hat);
  ZISA_UNUSED(t);
}

template <int Dim>
void ParaviewCatalystWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    real_t t,
    const Communicator *comm) {
  ProfileHost pofile("ParaviewCatalystWriter::write");

  const int size = comm->size();
  const int rank = comm->rank();

  conduit_cpp::Node exec_params;

  auto state = exec_params["catalyst/state"];
  state["timestep"].set(snapshot_idx_);
  state["time"].set(t);

  auto channel = exec_params["catalyst/channels/grid"];
  channel["type"].set("mesh");

  auto mesh = channel["data"];
  mesh["coordsets/coords/type"].set("uniform");
  mesh["coordsets/coords/dims/i"].set(u.shape(1));
  mesh["coordsets/coords/dims/j"].set(Dim > 1 ? grid_.N_phys : 1);
  mesh["coordsets/coords/dims/k"].set(Dim > 2 ? grid_.N_phys : 1);
  mesh["coordsets/coords/origin/x"].set(static_cast<real_t>(rank) / size);
  mesh["coordsets/coords/origin/y"].set(0);
  mesh["coordsets/coords/origin/z"].set(0);
  mesh["coordsets/coords/spacing/dx"].set(1. / grid_.N_phys);
  mesh["coordsets/coords/spacing/dy"].set(1. / grid_.N_phys);
  mesh["coordsets/coords/spacing/dz"].set(1. / grid_.N_phys);
  mesh["topologies/mesh/type"].set("uniform");
  mesh["topologies/mesh/coordset"].set("coords");

  auto fields = mesh["fields"];
  fields["velocity/association"].set("vertex");
  fields["velocity/topology"].set("mesh");
  fields["velocity/volume_dependent"].set("false");
  const size_t elms_per_comp = zisa::product(u.shape()) / u.shape(0);
  fields["velocity/values/x"].set_external(
      u.raw(), elms_per_comp, 0 * elms_per_comp * sizeof(real_t));
  if (Dim > 1) {
    fields["velocity/values/y"].set_external(
        u.raw(), elms_per_comp, 1 * elms_per_comp * sizeof(real_t));
  }
  if (Dim > 2) {
    fields["velocity/values/z"].set_external(
        u.raw(), elms_per_comp, 2 * elms_per_comp * sizeof(real_t));
  }

  catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
  if (err != catalyst_status_ok) {
    fmt::print(stderr, "Failed to execute catalyst: {}\n", int(err));
    exit(1);
  }

  ++snapshot_idx_;
}

template <int Dim>
void ParaviewCatalystWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    real_t t,
    const Communicator *comm) {
  ZISA_UNUSED(u_hat);
  ZISA_UNUSED(t);
  ZISA_UNUSED(comm);
}

template <int Dim>
ParaviewCatalystWriter<Dim>::~ParaviewCatalystWriter() {
  conduit_cpp::Node node;
  catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok) {
    fmt::print(stderr, "Failed to finalize catalyst: {}\n", int(err));
    exit(1);
  }
}

template class ParaviewCatalystWriter<1>;
template class ParaviewCatalystWriter<2>;
template class ParaviewCatalystWriter<3>;

}

#endif
