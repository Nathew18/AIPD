#pragma once
// Wspólny interfejs dla wszystkich trzech wariantów implementacji.
// Każdy wariant realizuje TEN SAM algorytm: bezpośrednie O(N^2)
// + integrator leapfrog (kick-drift-kick).

#include "body.hpp"

#include <cmath>
#include <cstdint>
#include <string>

#if defined(__CUDACC__)
#define NBODY_HD __host__ __device__
#else
#define NBODY_HD
#endif

namespace nbody {

// warunki początkowe
// Losowa chmura N ciał w kuli o promieniu 1, masa całkowita = 1,
// identyczne dane na każdym komputerze.
Bodies make_random(std::size_t n, std::uint64_t seed);

// Dwa ciała o równych masach na orbicie kołowej (okres = 2*pi przy G = 1).
Bodies make_two_body();

// fizyka
// Przyspieszenie działające na ciało i od wszystkich ciał j z [0, n).
// Człon j == i daje 0 (dx = 0), więc nie trzeba go pomijać - brak rozgałęzień.
// Funkcja jest __host__ __device__, więc kernel CUDA może jej użyć 1:1.
NBODY_HD inline void accel_on_body(std::size_t i, std::size_t n,
                                   const real* x, const real* y, const real* z,
                                   const real* m, real G, real eps2,
                                   real& ax, real& ay, real& az) {
    const real xi = x[i], yi = y[i], zi = z[i];
    real sx = 0, sy = 0, sz = 0;
    for (std::size_t j = 0; j < n; ++j) {
        const real dx = x[j] - xi;
        const real dy = y[j] - yi;
        const real dz = z[j] - zi;
        const real r2 = dx * dx + dy * dy + dz * dz + eps2;
        const real inv_r = real(1) / std::sqrt(r2);
        const real s = m[j] * inv_r * inv_r * inv_r;
        sx += dx * s;
        sy += dy * s;
        sz += dz * s;
    }
    ax = G * sx;
    ay = G * sy;
    az = G * sz;
}

// warianty
// Każda funkcja wykonuje `steps` kroków symulacji na `b` (w miejscu).
void run_seq(Bodies& b, const Params& p, int steps);
void run_omp(Bodies& b, const Params& p, int steps, int threads);

struct CudaTimings {
    double h2d_s    = 0; // transfer host -> device
    double kernel_s = 0; // obliczenia na GPU
    double d2h_s    = 0; // transfer device -> host
};
// Rzuca std::runtime_error, jeśli program zbudowano bez CUDA.
void run_cuda(Bodies& b, const Params& p, int steps, CudaTimings* timings = nullptr);
bool cuda_available();
std::string cuda_device_name();

// Liczba wątków OpenMP dostępna domyślnie (1, jeśli brak OpenMP).
int omp_max_threads();

// poprawność
double total_energy(const Bodies& b, const Params& p);   // kinetyczna + potencjalna
double momentum_magnitude(const Bodies& b);              // |suma m*v|
double max_position_diff(const Bodies& a, const Bodies& b);

} // namespace nbody
