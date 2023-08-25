#include <azeban/io/writer_factory.hpp>
#include <azeban/logging.hpp>
#include <azeban/operations/fft_factory.hpp>
#include <azeban/profiler.hpp>
#include <boost/program_options.hpp>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <zisa/io/netcdf_serial_writer.hpp>

using namespace azeban;
namespace po = boost::program_options;

struct SampleFile {
public:
  int ncid = 0;

  static std::unique_ptr<SampleFile> open(std::string_view path) {
    return std::unique_ptr<SampleFile>(new SampleFile(path));
  }

  ~SampleFile() {
    int status = nc_close(ncid);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to close sample");
  }

  size_t N() const {
    size_t len;
    int status = nc_inq_dimlen(ncid, get_dimid("N"), &len);
    AZEBAN_ERR_IF(status != NC_NOERR,
                  "Failed to get length of dimension \"N\"");
    return len;
  }

  int dim() const {
    int ndims;
    int status = nc_inq_varndims(ncid, get_varid("u"), &ndims);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to get ndims");
    return ndims;
  }

  template <int Dim>
  void read(const zisa::array_view<real_t, Dim> &data) const {
    ProfileHost profile("SampleFile::read");
    AZEBAN_ERR_IF(Dim != dim() + 1, "Data has wrong dimension");
    const auto read_cpu = [&](const zisa::array_view<real_t, Dim> &data_cpu) {
      const size_t comp_size = data.size() / data.shape(0);
      if (Dim > 1) {
        read_impl("u", data_cpu.raw() + 0 * comp_size);
      }
      if (Dim > 2) {
        read_impl("v", data_cpu.raw() + 1 * comp_size);
      }
      if (Dim > 3) {
        read_impl("w", data_cpu.raw() + 2 * comp_size);
      }
    };
    if (data.memory_location() == zisa::device_type::cpu) {
      read_cpu(data);
    }
#if ZISA_HAS_CUDA
    else if (data.memory_location() == zisa::device_type::cuda) {
      zisa::array<real_t, Dim> data_cpu(data.shape(), zisa::device_type::cpu);
      read_cpu(data_cpu);
      zisa::copy(data, data_cpu);
    }
#endif
    else {
      AZEBAN_ERR("Unknown memory location");
    }
  }

private:
  SampleFile(std::string_view path) {
    int status = nc_open(path.data(), 0, &ncid);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to open sample");
  }

  int get_dimid(std::string_view name) const {
    int dimid;
    int status = nc_inq_dimid(ncid, name.data(), &dimid);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to get netCDF dimension");
    return dimid;
  }

  int get_varid(std::string_view name) const {
    int varid;
    int status = nc_inq_varid(ncid, name.data(), &varid);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to open variable");
    return varid;
  }

  void read_impl(std::string_view name, float *data) const {
    int status = nc_get_var_float(ncid, get_varid(name), data);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to read variable");
  }

  void read_impl(std::string_view name, double *data) const {
    int status = nc_get_var_double(ncid, get_varid(name), data);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to read variable");
  }
};

nlohmann::json read_config(std::string_view config_filename) {
  std::ifstream config_file(config_filename.data());
  nlohmann::json config;
  config_file >> config;
  config["snapshots"] = 0;
  return config;
}

template <int Dim>
void run_for_sample(const nlohmann::json &config,
                    const std::unique_ptr<SampleFile> &sample_file,
                    zisa::int_t sample_idx,
                    zisa::int_t time_idx) {
  const size_t N = sample_file->N();
  const Grid<Dim> grid(N);
  zisa::device_type device = zisa::device_type::cpu;
  if (config.contains("device")) {
    if (config["device"] == "cpu") {
      device = zisa::device_type::cpu;
    }
#if ZISA_HAS_CUDA
    else if (config["device"] == "cuda") {
      device = zisa::device_type::cuda;
    }
#endif
    else {
      AZEBAN_ERR("Unsupported memory location");
    }
  }
  zisa::array<real_t, Dim + 1> sample(grid.shape_phys(Dim), device);
  zisa::array<complex_t, Dim + 1> sample_hat(grid.shape_fourier(Dim), device);
  auto fft = make_fft<Dim>(sample_hat.view(), sample.view(), FFT_FORWARD);
  sample_file->read(sample.view());
  std::unique_ptr<Writer<Dim>> writer
      = make_writer<Dim>(config["writer"], grid, sample_idx);
  writer->set_snapshot_idx(time_idx);
  writer->write(sample, 0);
  fft->forward();
  writer->write(sample_hat, 0);
}

int main(int argc, char *argv[]) {
  Profiler::start();

  std::string sample_path;
  std::string config_path;
  zisa::int_t sample_idx;
  zisa::int_t time;
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")(
      "sample",
      po::value<std::string>(&sample_path)->required(),
      "Path to the sample")("config",
                            po::value<std::string>(&config_path)->required(),
                            "Path to the config file")(
      "sample_idx",
      po::value<zisa::int_t>(&sample_idx)->default_value(0),
      "Index under which to store the output")(
      "time",
      po::value<zisa::int_t>(&time)->default_value(0),
      "Index of the time step");
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    exit(1);
  }
  po::notify(vm);

  auto config = read_config(config_path);
  const std::unique_ptr<SampleFile> sample_file = SampleFile::open(sample_path);
  const int dim = sample_file->dim();
  switch (dim) {
  case 1:
    run_for_sample<1>(config, sample_file, sample_idx, time);
    break;
  case 2:
    run_for_sample<2>(config, sample_file, sample_idx, time);
    break;
  case 3:
    run_for_sample<3>(config, sample_file, sample_idx, time);
    break;
  default:
    std::cout << "Invalid dimension: " << dim << std::endl;
    exit(1);
  }

  Profiler::stop();
  Profiler::summarize(std::cout);
  std::ofstream pstream("profiling.out");
  Profiler::serialize(pstream);

  return EXIT_SUCCESS;
}
