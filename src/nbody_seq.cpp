// Wariant 1: CPU sekwencyjnie (punkt odniesienia).
#include "simulation.hpp"

namespace nbody {

namespace {

void compute_accelerations(Bodies& b, const Params& p) {
    const std::size_t n = b.size();
    const real eps2 = p.eps * p.eps;
    for (std::size_t i = 0; i < n; ++i)
        accel_on_body(i, n, b.x.data(), b.y.data(), b.z.data(), b.m.data(),
                      p.G, eps2, b.ax[i], b.ay[i], b.az[i]);
}

// v += a * h
void kick(Bodies& b, real h) {
    for (std::size_t i = 0; i < b.size(); ++i) {
        b.vx[i] += b.ax[i] * h;
        b.vy[i] += b.ay[i] * h;
        b.vz[i] += b.az[i] * h;
    }
}

// x += v * h
void drift(Bodies& b, real h) {
    for (std::size_t i = 0; i < b.size(); ++i) {
        b.x[i] += b.vx[i] * h;
        b.y[i] += b.vy[i] * h;
        b.z[i] += b.vz[i] * h;
    }
}

} // namespace

// Leapfrog KDK: pół kroku prędkości, pełny krok położenia,
// nowe przyspieszenia, drugie pół kroku prędkości.
void run_seq(Bodies& b, const Params& p, int steps) {
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
