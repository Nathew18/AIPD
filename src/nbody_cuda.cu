// Wariant 3: GPU + CUDA - wersja podstawowa (Krok 6, wersja 1).
//   * jeden wątek GPU = jedno ciało,
//   * dane kopiowane na GPU RAZ, wszystkie kroki liczone na karcie,
//   * osobny pomiar transferów i obliczeń (cudaEvent).
//
// TODO (Krok 6, wersja 2): kernel z tilingiem w pamięci __shared__
//      i porównanie z tą wersją.
#include "simulation.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess)                                                \
            throw std::runtime_error(std::string("CUDA: ") +                    \
                                     cudaGetErrorString(err_) + " w " #call);   \
    } while (0)

namespace nbody {

namespace {

constexpr int kBlockSize = 256;

struct DeviceBodies {
    real *x, *y, *z, *vx, *vy, *vz, *ax, *ay, *az, *m;
};

// Liczy nowe przyspieszenie ciała i, potem v += a * h.
// Przy h = 0 tylko liczy przyspieszenia (start symulacji).
__global__ void accel_kick_kernel(int n, DeviceBodies d, real G, real eps2, real h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real ax, ay, az;
    accel_on_body(std::size_t(i), std::size_t(n), d.x, d.y, d.z, d.m, G, eps2, ax, ay, az);
    d.ax[i] = ax; d.ay[i] = ay; d.az[i] = az;
    d.vx[i] += ax * h; d.vy[i] += ay * h; d.vz[i] += az * h;
}

// v += a * h_kick, potem x += v * h_drift.
__global__ void kick_drift_kernel(int n, DeviceBodies d, real h_kick, real h_drift) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d.vx[i] += d.ax[i] * h_kick; d.vy[i] += d.ay[i] * h_kick; d.vz[i] += d.az[i] * h_kick;
    d.x[i] += d.vx[i] * h_drift; d.y[i] += d.vy[i] * h_drift; d.z[i] += d.vz[i] * h_drift;
}

} // namespace

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cuda_device_name() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "brak GPU";
    return prop.name;
}

void run_cuda(Bodies& b, const Params& p, int steps, CudaTimings* timings) {
    const int n = int(b.size());
    const std::size_t bytes = std::size_t(n) * sizeof(real);
    const real eps2 = p.eps * p.eps;
    const real half = p.dt * real(0.5);
    const int grid = (n + kBlockSize - 1) / kBlockSize;

    // Alokacja - poza pomiarami transferu/obliczeń.
    DeviceBodies d{};
    real** ptrs[] = {&d.x, &d.y, &d.z, &d.vx, &d.vy, &d.vz, &d.ax, &d.ay, &d.az, &d.m};
    for (real** ptr : ptrs) CUDA_CHECK(cudaMalloc(ptr, bytes));

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

    // --- host -> device
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d.x, b.x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.y, b.y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.z, b.z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vx, b.vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vy, b.vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vz, b.vz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.m, b.m.data(), bytes, cudaMemcpyHostToDevice));

    // --- obliczenia (leapfrog KDK, jak w wersji CPU)
    CUDA_CHECK(cudaEventRecord(e1));
    accel_kick_kernel<<<grid, kBlockSize>>>(n, d, p.G, eps2, real(0));
    for (int s = 0; s < steps; ++s) {
        kick_drift_kernel<<<grid, kBlockSize>>>(n, d, half, p.dt);
        accel_kick_kernel<<<grid, kBlockSize>>>(n, d, p.G, eps2, half);
    }
    CUDA_CHECK(cudaGetLastError());

    // --- device -> host
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaMemcpy(b.x.data(), d.x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.y.data(), d.y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.z.data(), d.z, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.vx.data(), d.vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.vy.data(), d.vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.vz.data(), d.vz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.ax.data(), d.ax, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.ay.data(), d.ay, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.az.data(), d.az, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));

    if (timings) {
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1); timings->h2d_s = ms / 1000.0;
        cudaEventElapsedTime(&ms, e1, e2); timings->kernel_s = ms / 1000.0;
        cudaEventElapsedTime(&ms, e2, e3); timings->d2h_s = ms / 1000.0;
    }

    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    for (real** ptr : ptrs) cudaFree(*ptr);
}

} // namespace nbody
