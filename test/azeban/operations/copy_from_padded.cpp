#include <azeban/catch.hpp>
#include <azeban/operations/copy_from_padded.hpp>
#include <azeban/operations/copy_to_padded.hpp>
#include <random>
#include <zisa/memory/array.hpp>

template <int compact_dim, bool... pad>
static void test_unpadding(zisa::int_t N_unpadded,
                           zisa::int_t N_padded,
                           zisa::device_type device) {
  static constexpr int dim_v = sizeof...(pad);
  static_assert(dim_v >= 1 && dim_v <= 3, "");
  static_assert(compact_dim == -1 || (compact_dim >= 0 && compact_dim < dim_v),
                "");

  static constexpr bool pad_arr[dim_v] = {pad...};

  zisa::shape_t<dim_v> shape_unpadded;
  zisa::shape_t<dim_v> shape_padded;
  for (int i = 0; i < dim_v; ++i) {
    shape_unpadded[i] = (i == compact_dim) ? N_unpadded / 2 + 1 : N_unpadded;
    if (pad_arr[i]) {
      shape_padded[i] = (i == compact_dim) ? N_padded / 2 + 1 : N_padded;
    } else {
      shape_padded[i] = N_unpadded;
    }
  }

  zisa::array<azeban::complex_t, dim_v> h_unpadded_arr(shape_unpadded);
  zisa::array<azeban::complex_t, dim_v> h_padded_arr(shape_padded);
  zisa::array<azeban::complex_t, dim_v> unpadded_arr(shape_unpadded, device);
  zisa::array<azeban::complex_t, dim_v> padded_arr(shape_padded, device);
  zisa::array<azeban::complex_t, dim_v> result_arr(shape_unpadded);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (azeban::complex_t &c : h_unpadded_arr) {
    c.x = dist(rng);
    c.y = dist(rng);
  }

  zisa::copy(unpadded_arr, h_unpadded_arr);
  azeban::copy_to_padded(pad..., compact_dim, padded_arr, unpadded_arr);
  azeban::copy_from_padded(pad..., compact_dim, unpadded_arr, padded_arr);
  zisa::copy(result_arr, unpadded_arr);

  for (zisa::int_t i = 0; i < zisa::product(shape_unpadded); ++i) {
    REQUIRE(h_unpadded_arr[i] == result_arr[i]);
  }
}

#define REGISTER_1D_TEST_CASE(CD, PAD_X, DEV_STR, DEV)                         \
  TEST_CASE("Unpadding 1D " #CD "," #PAD_X " " DEV_STR,                        \
            "[operations][copy_from_padded]") {                                \
    std::cout << "TESTING: Unpadding 1D " #CD "," #PAD_X " " DEV_STR           \
                 " [operations][copy_from_padded]"                             \
              << std::endl;                                                    \
    for (zisa::int_t N = 8; N <= 1024; N <<= 1) {                              \
      test_unpadding<CD, PAD_X>(N, N, DEV);                                    \
      test_unpadding<CD, PAD_X>(N, 3 * N / 2, DEV);                            \
      test_unpadding<CD, PAD_X>(N, 2 * N, DEV);                                \
    }                                                                          \
  }

#define REGISTER_2D_TEST_CASE(CD, PAD_X, PAD_Y, DEV_STR, DEV)                  \
  TEST_CASE("Unpadding 2D " #CD "," #PAD_X "," #PAD_Y " " DEV_STR,             \
            "[operations][copy_from_padded]") {                                \
    std::cout << "TESTING: Unpadding 2D " #CD "," #PAD_X "," #PAD_Y            \
                 " " DEV_STR " [operations][copy_from_padded]"                 \
              << std::endl;                                                    \
    for (zisa::int_t N = 8; N <= 1024; N <<= 1) {                              \
      test_unpadding<CD, PAD_X, PAD_Y>(N, N, DEV);                             \
      test_unpadding<CD, PAD_X, PAD_Y>(N, 3 * N / 2, DEV);                     \
      test_unpadding<CD, PAD_X, PAD_Y>(N, 2 * N, DEV);                         \
    }                                                                          \
  }

#define REGISTER_3D_TEST_CASE(CD, PAD_X, PAD_Y, PAD_Z, DEV_STR, DEV)           \
  TEST_CASE("Unpadding 3D " #CD "," #PAD_X "," #PAD_Y "," #PAD_Z " " DEV_STR,  \
            "[operations][copy_from_padded]") {                                \
    std::cout << "TESTING: Unpadding 3D " #CD "," #PAD_X "," #PAD_Y "," #PAD_Z \
                 " " DEV_STR " [operations][copy_from_padded]"                 \
              << std::endl;                                                    \
    for (zisa::int_t N = 8; N <= 64; N <<= 1) {                                \
      test_unpadding<CD, PAD_X, PAD_Y, PAD_Z>(N, N, DEV);                      \
      test_unpadding<CD, PAD_X, PAD_Y, PAD_Z>(N, 3 * N / 2, DEV);              \
      test_unpadding<CD, PAD_X, PAD_Y, PAD_Z>(N, 2 * N, DEV);                  \
    }                                                                          \
  }

REGISTER_1D_TEST_CASE(-1, true, "CPU", zisa::device_type::cpu);
REGISTER_1D_TEST_CASE(-1, true, "GPU", zisa::device_type::cuda);
REGISTER_1D_TEST_CASE(-1, false, "CPU", zisa::device_type::cpu);
REGISTER_1D_TEST_CASE(-1, false, "GPU", zisa::device_type::cuda);
REGISTER_1D_TEST_CASE(0, true, "CPU", zisa::device_type::cpu);
REGISTER_1D_TEST_CASE(0, true, "GPU", zisa::device_type::cuda);

REGISTER_2D_TEST_CASE(-1, true, true, "CPU", zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(-1, true, true, "GPU", zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(-1, true, false, "CPU", zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(-1, true, false, "GPU", zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(-1, false, true, "CPU", zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(-1, false, true, "GPU", zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(-1, false, false, "CPU", zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(-1, false, false, "GPU", zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(0, true, true, "CPU", zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(0, true, true, "GPU", zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(0, true, false, "CPU", zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(0, true, false, "GPU", zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(1, true, true, "CPU", zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(1, true, true, "GPU", zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(1, false, true, "CPU", zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(1, false, true, "GPU", zisa::device_type::cuda);

REGISTER_3D_TEST_CASE(-1, true, true, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(-1, true, true, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(-1, true, true, false, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(-1, true, true, false, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(-1, true, false, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(-1, true, false, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(-1, true, false, false, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(-1, true, false, false, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(-1, false, true, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(-1, false, true, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(-1, false, true, false, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(-1, false, true, false, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(-1, false, false, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(-1, false, false, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(-1, false, false, false, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(-1, false, false, false, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(0, true, true, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(0, true, true, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(0, true, true, false, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(0, true, true, false, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(0, true, false, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(0, true, false, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(0, true, false, false, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(0, true, false, false, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(1, true, true, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(1, true, true, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(1, true, true, false, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(1, true, true, false, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(1, false, true, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(1, false, true, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(1, false, true, false, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(1, false, true, false, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(2, true, true, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(2, true, true, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(2, true, false, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(2, true, false, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(2, false, true, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(2, false, true, true, "GPU", zisa::device_type::cuda)
REGISTER_3D_TEST_CASE(2, false, false, true, "CPU", zisa::device_type::cpu)
REGISTER_3D_TEST_CASE(2, false, false, true, "GPU", zisa::device_type::cuda)

#undef REGISTER_1D_TEST_CASE
#undef REGISTER_2D_TEST_CASE
#undef REGISTER_3D_TEST_CASE
