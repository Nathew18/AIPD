// Wariant 2: CPU + OpenMP.
// Ten sam algorytm co nbody_seq.cpp; pętle po ciałach dzielone między wątki.
// Każdy wątek zapisuje tylko przyspieszenia "swoich" ciał => brak wyścigów.
#include "simulation.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nbody {

namespace {

// Indeks ze znakiem: MSVC obsługuje tylko OpenMP 2.0, które tego wymaga.
using idx = long long;

void compute_accelerations(Bodies& b, const Params& p) {
    const idx n = idx(b.size());
    const real eps2 = p.eps * p.eps;
    const real *x = b.x.data(), *y = b.y.data(), *z = b.z.data(), *m = b.m.data();
    real *ax = b.ax.data(), *ay = b.ay.data(), *az = b.az.data();
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i)
        accel_on_body(std::size_t(i), std::size_t(n), x, y, z, m, p.G, eps2,
                      ax[i], ay[i], az[i]);
}

void kick(Bodies& b, real h) {
    const idx n = idx(b.size());
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        b.vx[i] += b.ax[i] * h;
        b.vy[i] += b.ay[i] * h;
        b.vz[i] += b.az[i] * h;
    }
}

void drift(Bodies& b, real h) {
    const idx n = idx(b.size());
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        b.x[i] += b.vx[i] * h;
        b.y[i] += b.vy[i] * h;
        b.z[i] += b.vz[i] * h;
    }
}

} // namespace

int omp_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void run_omp(Bodies& b, const Params& p, int steps, int threads) {
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    const real half = p.dt * real(0.5);
    compute_accelerations(b, p);
    for (int s = 0; s < steps; ++s) {
        kick(b, half);
        drift(b, p.dt);
        compute_accelerations(b, p);
        kick(b, half);
    }
}

} // namespace nbody
