// Warianty GPU: kernel podstawowy oraz tiling w pamieci wspoldzielonej.
//   * jeden wątek GPU = jedno ciało,
//   * dane kopiowane na GPU RAZ, wszystkie kroki liczone na karcie,
//   * osobny pomiar transferów i obliczeń (cudaEvent).
//
#include "simulation.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <limits>

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

// Jeden bufor na wszystkie tablice; sprzatanie takze po wyjatku CUDA_CHECK.
struct Resources {
    real* data = nullptr;
    cudaEvent_t events[4]{};
    ~Resources() {
        for (auto event : events) if (event) cudaEventDestroy(event);
        if (data) cudaFree(data);
    }
    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
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

__global__ void accel_kick_tiled_kernel(int n, DeviceBodies d, real G, real eps2, real h) {
    struct Particle { real x, y, z, m; };
    __shared__ Particle tile[kBlockSize];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + lane;
    const bool active = i < n;
    const real xi = active ? d.x[i] : real(0);
    const real yi = active ? d.y[i] : real(0);
    const real zi = active ? d.z[i] : real(0);
    real sx = 0, sy = 0, sz = 0;
    // Nieaktywne watki ostatniego bloku tez musza dojsc do obu barier.
    // Indeks size_t zapobiega przepelnieniu dla ostatniego kafla.
    for (std::size_t base = 0; base < std::size_t(n); base += kBlockSize) {
        const std::size_t j = base + lane;
        if (j < std::size_t(n)) tile[lane] = {d.x[j], d.y[j], d.z[j], d.m[j]};
        __syncthreads();
        const int count = int(std::size_t(n) - base < kBlockSize ?
                              std::size_t(n) - base : kBlockSize);
        if (active) {
            for (int k = 0; k < count; ++k)
                accumulate_pair(tile[k].x - xi, tile[k].y - yi, tile[k].z - zi,
                                tile[k].m, eps2, sx, sy, sz);
        }
        __syncthreads();
    }
    if (active) {
        const real ax = G * sx, ay = G * sy, az = G * sz;
        d.ax[i] = ax; d.ay[i] = ay; d.az[i] = az;
        d.vx[i] += ax * h; d.vy[i] += ay * h; d.vz[i] += az * h;
    }
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

void run_cuda(Bodies& b, const Params& p, int steps, CudaTimings* timings, CudaKernel kernel) {
    validate_simulation(b, p, steps);
    if (timings) *timings = {};
    if (b.size() == 0) return;
    if (b.size() > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("CUDA: za duzo cial (limit INT_MAX)");
    if (kernel != CudaKernel::naive && kernel != CudaKernel::tiled)
        throw std::invalid_argument("CUDA: nieznany kernel");
    const int n = int(b.size());
    const std::size_t bytes = std::size_t(n) * sizeof(real);
    const real eps2 = p.eps * p.eps;
    const real half = p.dt * real(0.5);
    const int grid = (n - 1) / kBlockSize + 1;

    // Alokacja - poza pomiarami transferu/obliczeń.
    DeviceBodies d{};
    Resources resources;
    CUDA_CHECK(cudaMalloc(&resources.data, bytes * 10));
    real** ptrs[] = {&d.x, &d.y, &d.z, &d.vx, &d.vy, &d.vz, &d.ax, &d.ay, &d.az, &d.m};
    for (int i = 0; i < 10; ++i) *ptrs[i] = resources.data + std::size_t(i) * n;

    for (auto& event : resources.events) CUDA_CHECK(cudaEventCreate(&event));
    const auto e0 = resources.events[0], e1 = resources.events[1];
    const auto e2 = resources.events[2], e3 = resources.events[3];

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
    auto accel_kick = [&](real h) {
        if (kernel == CudaKernel::tiled)
            accel_kick_tiled_kernel<<<grid, kBlockSize>>>(n, d, p.G, eps2, h);
        else
            accel_kick_kernel<<<grid, kBlockSize>>>(n, d, p.G, eps2, h);
        CUDA_CHECK(cudaGetLastError());
    };
    accel_kick(real(0));
    for (int s = 0; s < steps; ++s) {
        kick_drift_kernel<<<grid, kBlockSize>>>(n, d, half, p.dt);
        CUDA_CHECK(cudaGetLastError());
        accel_kick(half);
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
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); timings->h2d_s = ms / 1000.0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e1, e2)); timings->kernel_s = ms / 1000.0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e2, e3)); timings->d2h_s = ms / 1000.0;
    }

}

} // namespace nbody
