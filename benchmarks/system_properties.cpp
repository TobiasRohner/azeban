#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fmt/core.h>
#include <memory>
#if ZISA_HAS_CUDA
#include <cuda_runtime.h>
#endif
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

void measure_bandwidth_host(size_t bytes) {
  auto from = std::make_unique<uint8_t[]>(bytes);
  auto to = std::make_unique<uint8_t[]>(bytes);

  auto start = std::chrono::steady_clock::now();
  std::copy(from.get(), from.get() + bytes, to.get());
  auto stop = std::chrono::steady_clock::now();

  std::chrono::duration<double> time = stop - start;
  fmt::print("Host Bandwidth: {}bytes/s\n", bytes / time.count());
}

#if ZISA_HAS_CUDA
void measure_bandwidth_device(size_t bytes) {
  uint8_t *from, *to;
  cudaMalloc(&from, bytes);
  cudaMalloc(&to, bytes);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  auto stop = std::chrono::steady_clock::now();

  cudaFree(to);
  cudaFree(from);

  std::chrono::duration<double> time = stop - start;
  fmt::print("Device Bandwidth: {}bytes/s\n", bytes / time.count());
}

void measure_bandwidth_transfer(size_t bytes) {
  auto from = std::make_unique<uint8_t[]>(bytes);
  uint8_t *to;
  cudaMalloc(&to, bytes);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  cudaMemcpy(to, from.get(), bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  auto stop = std::chrono::steady_clock::now();

  cudaFree(to);

  std::chrono::duration<double> time = stop - start;
  fmt::print("Transfer Bandwidth: {}bytes/s\n", bytes / time.count());
}
#endif

#if AZEBAN_HAS_MPI
void measure_bandwidth_mpi_send(size_t bytes) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    return;
  }
  auto buffer = std::make_unique<uint8_t[]>(bytes);

  MPI_Barrier(MPI_COMM_WORLD);
  auto start = std::chrono::steady_clock::now();
  if (rank == 0) {
    MPI_Send(buffer.get(), bytes, MPI_UINT8_T, 1, 0, MPI_COMM_WORLD);
  }
  if (rank == 1) {
    MPI_Recv(buffer.get(),
             bytes,
             MPI_UINT8_T,
             0,
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  auto stop = std::chrono::steady_clock::now();

  std::chrono::duration<double> time = stop - start;
  fmt::print("MPI Bandwidth: {}bytes/s\n", bytes / time.count());
}
#endif

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // 4GB
  static constexpr size_t bytes = 4ull * 1024ull * 1024ull * 1024ull;

  measure_bandwidth_host(bytes);
#if ZISA_HAS_CUDA
  measure_bandwidth_device(bytes);
  measure_bandwidth_transfer(bytes);
#endif
#if AZEBAN_HAS_MPI
  measure_bandwidth_mpi_send(bytes);
#endif

  MPI_Finalize();
  return 0;
}
