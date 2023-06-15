#include <azeban/operations/second_order_structure_function.hpp>
#include <cmath>
#include <iostream>
#include <omp.h>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/mpi_types.hpp>
#endif

namespace azeban {

namespace detail {

template <int Dim>
struct IExact;

template <>
struct IExact<1> {
  static real_t eval(real_t k, real_t r) {
    const real_t kr = k * r;
    return 2 * (1 - sin(kr) / kr);
  }
};

template <>
struct IExact<2> {
  static real_t eval(real_t k, real_t r) {
    const real_t kr = k * r;
    return 2 - (4 * std::cyl_bessel_j(real_t(1), kr)) / kr;
  }
};

template <>
struct IExact<3> {
  static real_t eval(real_t k, real_t r) {
    const real_t kr = k * r;
    const real_t kr2 = kr * kr;
    const real_t kr3 = kr2 * kr;
    return 2 + 6 * (std::cos(kr) / kr2 - std::sin(kr) / kr3);
  }
};

template <int Dim>
struct IApprox;

template <>
struct IApprox<1> {
  static real_t eval(real_t k, real_t r) { return 2 * std::sin(k * r) / k; }
};

template <>
struct IApprox<2> {
  static real_t eval(real_t k, real_t r) {
    const real_t v = std::min(k * r / 2, static_cast<real_t>(std::sqrt(2)));
    return v * v;
  }
};

template <>
struct IApprox<3> {
  static real_t eval(real_t k, real_t r) {
    const real_t kr = k * r;
    return std::min(kr * kr / 5, real_t(2));
  }
};

}

template <typename I_OP>
static std::vector<real_t> second_order_structure_function_cpu(
    const Grid<1> &grid,
    const zisa::array_const_view<complex_t, 2> &u_hat,
    long k1_offset = 0) {
  const size_t Nr = (grid.N_phys + 1) / 2;
  const real_t dx = 1. / grid.N_phys;
  std::vector<real_t> S(Nr, 0);
  for (zisa::int_t r = 0; r < Nr; ++r) {
    for (zisa::int_t d = 0; d < 1; ++d) {
      for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
        long k1 = i + k1_offset;
        if (k1 >= zisa::integer_cast<long>(grid.N_fourier)) {
          k1 -= grid.N_phys;
        }
        const real_t K = 2 * zisa::pi * std::abs(k1);
        if (K == 0) {
          continue;
        }
        const real_t uk2 = abs2(u_hat(d, i) / zisa::pow<1>(grid.N_phys));
        S[r] += I_OP::eval(K, r * dx) * uk2;
      }
    }
  }
  return S;
}

template <typename I_OP>
static std::vector<real_t> second_order_structure_function_cpu(
    const Grid<2> &grid,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    long k1_offset = 0,
    long k2_offset = 0) {
  const size_t Nr = (grid.N_phys + 1) / 2;
  const real_t dx = 1. / grid.N_phys;
  std::vector<real_t> S(Nr, 0);
  for (zisa::int_t r = 0; r < Nr; ++r) {
    real_t Sr = 0;
#pragma omp parallel for collapse(2) reduction(+ : Sr)
    for (zisa::int_t d = 0; d < 2; ++d) {
      for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
        for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
          long k1 = i + k1_offset;
          if (k1 >= zisa::integer_cast<long>(grid.N_fourier)) {
            k1 -= grid.N_phys;
          }
          long k2 = j + k2_offset;
          if (k2 >= zisa::integer_cast<long>(grid.N_fourier)) {
            k2 -= grid.N_phys;
          }
          const long absk2 = k1 * k1 + k2 * k2;
          const real_t K = 2 * zisa::pi * zisa::sqrt(absk2);
          if (K == 0) {
            continue;
          }
          const real_t uk2 = abs2(u_hat(d, i, j) / zisa::pow<2>(grid.N_phys));
          Sr += I_OP::eval(K, r * dx) * uk2 / 2;
        }
      }
    }
    S[r] = Sr;
  }
  return S;
}

template <typename I_OP>
static std::vector<real_t> second_order_structure_function_cpu(
    const Grid<3> &grid,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    long k1_offset = 0,
    long k2_offset = 0,
    long k3_offset = 0) {
  const size_t Nr = (grid.N_phys + 1) / 2;
  const real_t dx = 1. / grid.N_phys;
  int N_threads;
#pragma omp parallel
  {
#pragma omp master
    N_threads = omp_get_num_threads();
  }
  std::vector<std::vector<real_t>> S(N_threads, std::vector<real_t>(Nr, 0));
  const zisa::int_t N1 = u_hat.shape(1);
  const zisa::int_t N2 = u_hat.shape(2);
  const zisa::int_t N3 = u_hat.shape(3);
#pragma omp parallel for collapse(4)
  for (zisa::int_t d = 0; d < 3; ++d) {
    for (zisa::int_t i = 0; i < N1; ++i) {
      for (zisa::int_t j = 0; j < N2; ++j) {
        for (zisa::int_t k = 0; k < N3; ++k) {
          long k1 = i + k1_offset;
          if (k1 >= zisa::integer_cast<long>(grid.N_fourier)) {
            k1 -= grid.N_phys;
          }
          long k2 = j + k2_offset;
          if (k2 >= zisa::integer_cast<long>(grid.N_fourier)) {
            k2 -= grid.N_phys;
          }
          long k3 = k + k3_offset;
          if (k3 >= zisa::integer_cast<long>(grid.N_fourier)) {
            k3 -= grid.N_phys;
          }
          const long absk2 = k1 * k1 + k2 * k2 + k3 * k3;
          const real_t K = 2 * zisa::pi * zisa::sqrt(absk2);
          if (K == 0) {
            continue;
          }
          const real_t uk2
              = abs2(u_hat(d, i, j, k) / zisa::pow<3>(grid.N_phys));
          const int tn = omp_get_thread_num();
          for (zisa::int_t r = 0; r < Nr; ++r) {
            S[tn][r] += I_OP::eval(K, r * dx) * uk2 / 3;
          }
        }
      }
    }
  }
  for (int i = 1; i < N_threads; ++i) {
    for (size_t j = 0; j < Nr; ++j) {
      S[0][j] += S[i][j];
    }
  }
  return S[0];
}

#if AZEBAN_HAS_MPI
template <int Dim, typename I_OP>
static std::vector<real_t> second_order_structure_function_cpu(
    const Grid<Dim> &grid,
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    MPI_Comm comm) {
  const long k1_offset = grid.i_fourier(0, comm);
  const std::vector<real_t> local_S
      = second_order_structure_function_cpu<I_OP>(grid, u_hat, k1_offset);
  std::vector<real_t> S(local_S.size(), 0);
  MPI_Reduce(local_S.data(),
             S.data(),
             local_S.size(),
             mpi_type<real_t>(),
             MPI_SUM,
             0,
             comm);
  return S;
}
#endif

template <int Dim, typename I_OP>
static std::vector<real_t> second_order_structure_function(
    const Grid<Dim> &grid,
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    auto S = second_order_structure_function_cpu<I_OP>(grid, u_hat);
    for (real_t &Sr : S) {
      Sr = std::sqrt(Sr);
    }
    return S;
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported memory loaction");
  }
}

template <int Dim>
std::vector<real_t> second_order_structure_function_exact(
    const Grid<Dim> &grid,
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
  return second_order_structure_function<Dim, detail::IExact<Dim>>(grid, u_hat);
}

template <int Dim>
std::vector<real_t> second_order_structure_function_approx(
    const Grid<Dim> &grid,
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
  return second_order_structure_function<Dim, detail::IApprox<Dim>>(grid,
                                                                    u_hat);
}

template std::vector<real_t> second_order_structure_function_exact(
    const Grid<1> &, const zisa::array_const_view<complex_t, 2> &);
template std::vector<real_t> second_order_structure_function_exact(
    const Grid<2> &, const zisa::array_const_view<complex_t, 3> &);
template std::vector<real_t> second_order_structure_function_exact(
    const Grid<3> &, const zisa::array_const_view<complex_t, 4> &);
template std::vector<real_t> second_order_structure_function_approx(
    const Grid<1> &, const zisa::array_const_view<complex_t, 2> &);
template std::vector<real_t> second_order_structure_function_approx(
    const Grid<2> &, const zisa::array_const_view<complex_t, 3> &);
template std::vector<real_t> second_order_structure_function_approx(
    const Grid<3> &, const zisa::array_const_view<complex_t, 4> &);

#if AZEBAN_HAS_MPI
template <int Dim, typename I_OP>
static std::vector<real_t> second_order_structure_function(
    const Grid<Dim> &grid,
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    MPI_Comm comm) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    auto S = second_order_structure_function_cpu<Dim, I_OP>(grid, u_hat, comm);
    for (real_t &Sr : S) {
      Sr = std::sqrt(Sr);
    }
    return S;
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported memory loaction");
  }
}

template <int Dim>
std::vector<real_t> second_order_structure_function_exact(
    const Grid<Dim> &grid,
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    MPI_Comm comm) {
  return second_order_structure_function<Dim, detail::IExact<Dim>>(
      grid, u_hat, comm);
}

template <int Dim>
std::vector<real_t> second_order_structure_function_approx(
    const Grid<Dim> &grid,
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    MPI_Comm comm) {
  return second_order_structure_function<Dim, detail::IApprox<Dim>>(
      grid, u_hat, comm);
}

template std::vector<real_t> second_order_structure_function_exact(
    const Grid<1> &, const zisa::array_const_view<complex_t, 2> &, MPI_Comm);
template std::vector<real_t> second_order_structure_function_exact(
    const Grid<2> &, const zisa::array_const_view<complex_t, 3> &, MPI_Comm);
template std::vector<real_t> second_order_structure_function_exact(
    const Grid<3> &, const zisa::array_const_view<complex_t, 4> &, MPI_Comm);
template std::vector<real_t> second_order_structure_function_approx(
    const Grid<1> &, const zisa::array_const_view<complex_t, 2> &, MPI_Comm);
template std::vector<real_t> second_order_structure_function_approx(
    const Grid<2> &, const zisa::array_const_view<complex_t, 3> &, MPI_Comm);
template std::vector<real_t> second_order_structure_function_approx(
    const Grid<3> &, const zisa::array_const_view<complex_t, 4> &, MPI_Comm);
#endif

}
